# Provider Long-Polling Design

**Date:** 2026-02-19
**Status:** Draft
**Scope:** zndraw-joblib (protocol + router), zndraw-fastapi (implementations + frontend)

## Problem

The current provider read flow has three issues:

1. **RFC 9457 misuse** — `FramePending` returns HTTP 202 with `application/problem+json`.
   Problem JSON is for error responses (4xx/5xx). Using it on a 2xx forces the frontend
   to sniff `Content-Type` on success responses (`throwProblemError` on 2xx).

2. **Three inconsistent patterns** for "data not ready yet":
   - Frames: 202 + problem+json + 3s polling + Socket.IO fast-path
   - Provider reads: 202 + plain JSON + TanStack Query retry (15x at 2s)
   - Task status: `Prefer: wait=N` long-polling (already correct)

3. **Redundant polling** — Socket.IO already notifies when data arrives, yet the
   client polls every 2-3s as the primary mechanism.

## Design: Long-Polling with Redis Pub/Sub

Replace the 202-and-retry pattern with **server-held long-polling**. The client
makes a single GET; the server holds the connection until data arrives or timeout.

### Why Redis Pub/Sub

- **Multi-server safe** — works across uvicorn workers connected via Redis.
  `asyncio.Future` or in-process signaling fails when request lands on worker A
  but the provider result uploads to worker B.
- **Purpose-built** — Redis pub/sub is designed for exactly this: cross-process
  notification with sub-millisecond latency.
- **Already available** — Redis is a required dependency. No new infrastructure.

### Flow: Provider Read (zndraw-joblib)

```
Client                          Server                          Provider Worker
  |                               |                               |
  |  GET /providers/{name}        |                               |
  |  Prefer: wait=5               |                               |
  |------------------------------>|                               |
  |                               |                               |
  |                               |  1. Subscribe to pub/sub      |
  |                               |     channel for cache_key     |
  |                               |  2. Check cache               |
  |                               |  3. Dispatch ProviderRequest  |
  |                               |     (if not inflight)         |
  |                               |------------------------------>|
  |                               |                               |  4. Process
  |                               |     POST /results             |
  |                               |<------------------------------|
  |                               |  5. Store in cache            |
  |                               |  6. PUBLISH to channel        |
  |                               |  7. Pub/sub wakes handler     |
  |                               |  8. Read from cache           |
  |  200 OK + data                |                               |
  |<------------------------------|                               |
```

On timeout (default 5s):

```
  |  404 Not Found                |
  |  Retry-After: 2               |
  |  application/problem+json     |
  |<------------------------------|
```

### Flow: Frame Read (zndraw-fastapi)

Same pattern. `_dispatch_frame()` becomes `_await_provider_frame()` — calls
`result_backend.wait_for_key(cache_key, timeout)` instead of raising `FramePending`.

## API Changes

### `ResultBackend` Protocol (zndraw-joblib/dependencies.py)

Add two methods:

```python
class ResultBackend(Protocol):
    # ... existing methods ...

    async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
        """Wait for a cache key to be populated.

        Subscribes to a notification channel, checks cache (race-safe),
        then awaits notification or timeout.

        Returns
        -------
        bytes | None
            Cached data if it arrives within timeout, None otherwise.
        """
        ...

    async def notify_key(self, key: str) -> None:
        """Notify waiters that a cache key has been populated."""
        ...
```

### `RedisResultBackend` Implementation (zndraw-fastapi/result_backends.py)

Uses `async with` for automatic pub/sub cleanup. `get_message(timeout=N)` natively
awaits up to N seconds — no manual deadline loop needed.
With `ignore_subscribe_messages=True`, only actual published messages are returned;
`None` means timeout. No message type checking needed.

```python
NOTIFY_PREFIX = "notify:"

async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
    channel = f"{NOTIFY_PREFIX}{key}"
    async with self._redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)

        # Check cache AFTER subscribing — handles the race where
        # the result landed between our first check and the subscribe.
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Single await — get_message natively blocks up to timeout seconds.
        msg = await pubsub.get_message(
            ignore_subscribe_messages=True, timeout=timeout
        )
        if msg is not None:
            return await self.get(key)

        return None

async def notify_key(self, key: str) -> None:
    await self._redis.publish(f"{NOTIFY_PREFIX}{key}", b"1")
```

### `CompositeResultBackend` (zndraw-fastapi/result_backends.py)

Pub/sub always goes through Redis (same pattern as `acquire_inflight`).
Data reads use `self.get()` which routes to the correct backend via `_backend_for()`.

```python
async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
    channel = f"{NOTIFY_PREFIX}{key}"
    async with self._redis._redis.pubsub() as pubsub:  # raw Redis client
        await pubsub.subscribe(channel)
        cached = await self.get(key)  # routes via _backend_for()
        if cached is not None:
            return cached
        msg = await pubsub.get_message(
            ignore_subscribe_messages=True, timeout=timeout
        )
        if msg is not None:
            return await self.get(key)  # routes via _backend_for()
        return None

async def notify_key(self, key: str) -> None:
    await self._redis.notify_key(key)  # always Redis
```

### `StorageResultBackend`

Does not support pub/sub (same as inflight locks):

```python
async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
    raise NotImplementedError("Use CompositeResultBackend for long-polling")

async def notify_key(self, key: str) -> None:
    raise NotImplementedError("Use CompositeResultBackend for long-polling")
```

## Router Changes

### Session Lifecycle — Critical

**Problem:** `SessionDep` holds a database session (and with in-memory SQLite,
the `asyncio.Lock`) for the entire request. A 5-30s long-poll would block all
other DB operations.

**Solution:** Use `SessionMakerDep` (the factory) instead of `SessionDep`.
Scope the session to just the provider lookup, close it before long-polling.
This matches the existing pattern in `get_task_status` (line 737).

| Dependency | Lifecycle | During long-poll |
|------------|-----------|------------------|
| `SessionDep` | Entire request | DB connection/lock held 5-30s |
| `SessionMakerDep` | Scoped `async with` blocks | No DB resources held |

### Provider Read Endpoint (zndraw-joblib/router.py)

No `response_model` — returns raw `Response(content=bytes)` with varying
`media_type` per provider. Errors use `ProblemType` exceptions (same as all
other endpoints). Remove the `responses=` dict and `ProviderReadPendingResponse`.

```python
@router.get("/rooms/{room_id}/providers/{provider_name:path}")
async def read_provider(
    room_id: str,
    provider_name: str,
    request: Request,
    response: Response,
    session_maker: SessionMakerDep,  # factory, not session — no lock during long-poll
    result_backend: ResultBackendDep,
    settings: SettingsDep,
    tsio: TsioDep,
    prefer: str | None = Header(None),
):
    """Read data from a provider. Long-polls until result is available."""
    validate_room_id(room_id)

    # Short-lived session — closed before long-poll
    async with session_maker() as session:
        provider = await _resolve_provider(session, provider_name, room_id)

    params = dict(request.query_params)
    rhash = request_hash(params)
    cache_key = f"provider-result:{provider.full_name}:{rhash}"
    inflight_key = f"provider-inflight:{provider.full_name}:{rhash}"

    # 1. Check cache — fast path
    cached = await result_backend.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type=provider.content_type)

    # 2. Dispatch if not already inflight
    acquired = await result_backend.acquire_inflight(
        inflight_key, settings.provider_inflight_ttl_seconds
    )
    if acquired:
        provider_room = f"providers:{provider.full_name}"
        await emit(
            tsio,
            {Emission(
                ProviderRequest.from_dict_params(
                    request_id=rhash,
                    provider_name=provider.full_name,
                    params=params,
                ),
                provider_room,
            )},
        )

    # 3. Long-poll: wait for result via Redis pub/sub
    requested_wait = parse_prefer_wait(prefer)
    timeout = min(
        requested_wait or settings.provider_long_poll_default_seconds,
        settings.provider_long_poll_max_seconds,
    )

    result = await result_backend.wait_for_key(cache_key, timeout)
    if result is not None:
        if requested_wait:
            response.headers["Preference-Applied"] = f"wait={int(timeout)}"
        return Response(content=result, media_type=provider.content_type)

    # 4. Timeout — RFC 9457 error with retry guidance
    raise ProviderTimeout.exception(
        detail=f"Provider '{provider_name}' did not respond within {timeout}s",
        headers={"Retry-After": "2"},
    )
```

### Upload Result Endpoint (zndraw-joblib/router.py)

Add `notify_key` after storing:

```python
@router.post("/providers/{provider_id}/results", status_code=204)
async def upload_provider_result(...):
    # ... existing validation + store ...

    await result_backend.store(cache_key, data, settings.provider_result_ttl_seconds)
    await result_backend.release_inflight(inflight_key)

    # Wake long-polling waiters (Redis pub/sub)
    await result_backend.notify_key(cache_key)

    # Notify frontend (Socket.IO — keep for UI refresh)
    await emit(tsio, {Emission(ProviderResultReady(...), ...)})
```

### Settings (zndraw-joblib/settings.py)

```python
class JobLibSettings(BaseSettings):
    # ... existing ...
    provider_long_poll_default_seconds: int = 5
    provider_long_poll_max_seconds: int = 30
```

### New Exception (zndraw-joblib/exceptions.py)

```python
class ProviderTimeout(ProblemType):
    """The provider did not respond within the requested wait time."""

    title: ClassVar[str] = "Not Found"
    status: ClassVar[int] = 404
```

### Removals in zndraw-joblib

| Remove | Reason |
|--------|--------|
| `ProviderReadPendingResponse` schema | No more 202 responses |
| `responses=` dict on `read_provider` | Errors via ProblemType exceptions |

## Frame Endpoint Changes (zndraw-fastapi)

### Session Lifecycle — Same Fix

Frame endpoints (`get_frame`, `get_frame_metadata`, `list_frames`) currently use
`session: SessionDep` and pass it through `_get_frame_or_dispatch` →
`_find_frames_provider`. With long-polling, this holds the session during the wait.

**Fix:** Use `SessionMakerDep`. Do all SQL work (verify room, get length, find
provider) in a scoped `async with`, close before long-polling.

Add `SessionMakerDep` to zndraw-fastapi dependencies (import `get_session_maker`
from `zndraw_auth.db`, same as zndraw-joblib does).

### Route handler pattern (e.g. `get_frame`)

```python
@router.get("/{index}", response_class=MessagePackResponse, ...)
async def get_frame(
    session_maker: SessionMakerDep,  # factory, not session
    storage: StorageDep,
    current_user: CurrentUserDep,
    sio: SioDep,
    result_backend: ResultBackendDep,
    joblib_settings: JobLibSettingsDep,
    room_id: str,
    index: int,
) -> Response:
    # All SQL in scoped session — closed before any long-poll
    async with session_maker() as session:
        await verify_room(session, room_id)
        total = await storage.get_length(room_id)
        if index < 0 or index >= total:
            _raise_frame_not_found(index, total)

        frame = await storage.get(room_id, index)
        if frame is not None:
            return MessagePackResponse(content=[frame])

        provider = await _find_frames_provider(session, room_id)
    # Session closed here ^

    if provider is None:
        _raise_frame_not_found(index, total)

    # Long-poll — no DB resources held
    frame = await _await_provider_frame(
        result_backend, sio, provider, index,
        timeout=joblib_settings.provider_long_poll_default_seconds,
        inflight_ttl=joblib_settings.provider_inflight_ttl_seconds,
    )
    return MessagePackResponse(content=[frame])
```

### `_dispatch_frame` -> `_await_provider_frame`

No session parameter — takes an already-resolved `ProviderRecord`.

```python
async def _await_provider_frame(
    result_backend: ResultBackend,
    sio: AsyncServerWrapper,
    provider: ProviderRecord,
    index: int,
    timeout: float = 5.0,
    inflight_ttl: int = 30,
) -> RawFrame:
    """Dispatch to provider and wait for result.

    Returns the frame on success. Raises FrameNotFound on timeout.
    """
    params = {"index": str(index)}
    rhash = request_hash(params)
    cache_key = f"provider-result:{provider.full_name}:{rhash}"
    inflight_key = f"provider-inflight:{provider.full_name}:{rhash}"

    # Check cache
    cached = await result_backend.get(cache_key)
    if cached is not None:
        return msgpack.unpackb(cached, raw=True)

    # Dispatch if not inflight
    acquired = await result_backend.acquire_inflight(inflight_key, inflight_ttl)
    if acquired:
        provider_room = f"providers:{provider.full_name}"
        await joblib_emit(sio, {Emission(
            ProviderRequest.from_dict_params(
                request_id=rhash,
                provider_name=provider.full_name,
                params=params,
            ),
            provider_room,
        )})

    # Long-poll — no DB resources held
    result = await result_backend.wait_for_key(cache_key, timeout)
    if result is not None:
        return msgpack.unpackb(result, raw=True)

    raise FrameNotFound.exception(
        detail=f"Frame {index} not available from provider",
        headers={"Retry-After": "2"},
    )
```

### `_get_frame_or_dispatch` — Remove

This function mixed SQL (provider lookup) with dispatch logic. With the new
pattern, the route handler does SQL in a scoped session, then calls
`_await_provider_frame` separately. The function is no longer needed.

## Removals

### Backend (zndraw-fastapi)

| Remove | Reason |
|--------|--------|
| `FramePending` exception class | No longer used — replaced by long-poll + timeout |
| `_resolve_provider_frame()` | Merged into `_await_provider_frame()` |
| `_dispatch_frame()` | Replaced by `_await_provider_frame()` |
| `_get_frame_or_dispatch()` | Split into route handler (SQL) + `_await_provider_frame` (wait) |

### Frontend (zndraw-fastapi)

| Remove | Reason |
|--------|--------|
| `FramePendingError` class | No more 202 responses to handle |
| `ProviderPendingError` class | No more 202 responses to handle |
| `throwProblemError()` + content-type sniffing | Frame endpoint always returns data or 404 |
| 3s retry loop in `getFrameBatched()` | Request either succeeds or fails — no pending state |
| `framePendingSince` store field | No pending state to track |
| `PENDING_RETRY_MS` constant | No retry interval needed |
| TanStack Query `retry` for `ProviderPendingError` | Standard error handling suffices |

### Frontend simplification of `getFrameBatched`

The `attemptFetch` retry loop collapses to a single call:

```typescript
getFrames(roomId, frame, toFetch, controller.signal)
    .then((data) => {
        for (const k of toFetch) {
            queryClient.setQueryData(
                ["frame", roomId, frame, k],
                data?.[k] !== undefined ? { [k]: data[k] } : null,
            );
        }
        resolve(data);
    })
    .catch(reject);
```

No `FramePendingError` branch. No retry. No `framePendingSince`.

## Socket.IO Events — What Changes

| Event | Change |
|-------|--------|
| `ProviderResultReady` | **Keep** — still useful for frontend UI refresh |
| `FramesInvalidate` | **Keep** — still used for non-provider frame mutations |
| `frames_invalidate` handler | Remove `FramePendingError` retry wake-up logic (no longer needed) |

The Socket.IO events remain for **UI invalidation** (e.g. a provider uploads frame 5 ->
the frontend invalidates its React Query cache for frame 5 so the next render fetches fresh).
But they are no longer the primary data delivery mechanism — long-polling handles that.

## Race Condition Safety

The subscribe-then-check pattern eliminates races:

```
T0: GET /providers/foo  -> cache miss
T1: subscribe("notify:provider-result:foo:abc123")
T2: check cache again   -> still miss (but if hit, return immediately)
T3: await get_message(timeout=5)...
    ---- meanwhile ----
T2.5: Provider uploads result -> store() + PUBLISH("notify:...")
    --------------------
T4: get_message returns msg -> read from cache -> return 200
```

If the result lands between T0 and T1 (before subscribe), the T2 cache check catches it.
If it lands between T1 and T2, the pub/sub message is already queued and T3 receives it.

## Abort / Cancellation

For frame scrubbing (rapid index changes), `AbortController` already works:
- Client aborts the in-flight HTTP request
- Server receives disconnect — `async with pubsub` cleans up the subscription
- No leaked subscriptions

## Testing

1. **Cache hit** — returns immediately, no pub/sub involved
2. **Cache miss + fast provider** — long-poll resolves in <100ms
3. **Cache miss + slow provider** — long-poll resolves at ~2s
4. **Timeout** — provider never responds, returns 404 after timeout
5. **Race condition** — result arrives between dispatch and subscribe
6. **Abort** — client disconnects mid-wait, subscription cleaned up
7. **Inflight dedup** — two concurrent requests for same key, only one dispatch
8. **`Prefer: wait=N`** — custom timeout respected, capped by max setting
