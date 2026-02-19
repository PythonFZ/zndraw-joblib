# Binary Provider Results

**Date**: 2026-02-19
**Status**: Approved

## Problem

Provider result upload and caching is hardcoded to JSON. The upload endpoint parses the request body via `ProviderResultUploadRequest` (Pydantic), then re-serializes with `json.dumps().encode()` for storage -- a pointless parse/reserialize roundtrip. The read endpoint hardcodes `media_type="application/json"` when returning cached data.

This prevents providers from returning binary formats (e.g. msgpack, arrow, parquet) which are common for scientific data.

## Design

### Principle

The content type is a property of the **provider type**, not per-request. A `FilesystemRead` provider always returns JSON; an `AtomsProvider` always returns msgpack. This is declared once on the class and stored at registration time.

### Changes

#### 1. Provider base class (`provider.py`)

Add `content_type` ClassVar defaulting to JSON for backwards compatibility:

```python
class Provider(BaseModel):
    category: ClassVar[str]
    content_type: ClassVar[str] = "application/json"
```

#### 2. ProviderRecord model (`models.py`)

Add column:

```python
content_type: Mapped[str] = mapped_column(String, default="application/json")
```

#### 3. Registration schema (`schemas.py`)

Add field to `ProviderRegisterRequest`:

```python
content_type: str = "application/json"
```

Delete `ProviderResultUploadRequest` entirely.

#### 4. Registration endpoint (`router.py`)

Store `request.content_type` on the ProviderRecord during registration.

#### 5. Upload endpoint (`router.py`)

Replace Pydantic body parameter with raw `Request`:

```python
@router.post("/providers/{provider_id}/results", status_code=204)
async def upload_provider_result(
    provider_id: UUID,
    request: Request,
    x_request_hash: str = Header(),
    ...
):
    data = await request.body()
    cache_key = f"provider-result:{provider.full_name}:{x_request_hash}"
    await result_backend.store(cache_key, data, ttl)
    ...
```

No parsing, no re-serialization. Body stored as-is.

#### 6. Read endpoint (`router.py`)

Use `provider.content_type` from the DB row (already queried):

```python
cached = await result_backend.get(cache_key)
if cached is not None:
    return Response(content=cached, media_type=provider.content_type, status_code=200)
```

#### 7. Client handler (`client.py`)

`_on_provider_request`: serialize based on `reg.cls.content_type`:

```python
result = instance.read(reg.handler)
if reg.cls.content_type == "application/json":
    data = json.dumps(result).encode()
else:
    data = result  # assumed bytes

resp = self.api.http.post(
    url, content=data,
    headers={**self.api.get_headers(), "X-Request-Hash": event.request_id},
)
```

`register_provider`: send `content_type` from `cls.content_type` in the registration payload.

### What gets deleted

- `ProviderResultUploadRequest` schema class
- `json.dumps(upload.data).encode()` roundtrip in upload endpoint
- Hardcoded `media_type="application/json"` in read endpoint

### Migration

The `content_type` column defaults to `"application/json"`, so existing ProviderRecords (if any in production) get the correct default. No data migration needed.
