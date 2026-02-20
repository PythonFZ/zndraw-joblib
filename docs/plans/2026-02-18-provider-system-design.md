# Provider System Design for zndraw-joblib

## Problem

ZnDraw needs a generic way for connected Python clients to **serve data on demand** to the
server and frontend. Two concrete use cases drive this:

1. **Frame sources** (`vis.mount()`): lazily serve atomistic frames
2. **Filesystem browsing** (`vis.register_fs()`): expose an fsspec filesystem for listing,
   searching, and loading files

Both follow the same pattern: a Python client registers as a **data provider**, the server
dispatches read requests via Socket.IO, the provider processes and POSTs results back via REST,
and the server caches + pushes updates to the frontend.

The current frame source implementation is bespoke code in `zndraw-fastapi`. Adding filesystem
browsing would duplicate the entire dispatch/result/lifecycle pattern. Instead, we generalize
it as a **Provider** abstraction inside `zndraw-joblib`.

### Why zndraw-joblib?

From a user's perspective, providers and jobs are the same concept: "my Python client offers
a capability to the server." Keeping both in one package means one import, one registration
flow, and — critically — **jobs can access provider handlers** (e.g., a `LoadFile` job needs
the fsspec filesystem object that the provider registered).

### Providers vs Jobs

Providers complement jobs — they don't replace them:

| | **Jobs** (existing) | **Providers** (new) |
|---|---|---|
| **Purpose** | User-initiated computation | Remote resource access |
| **Dispatch** | Workers pull/claim (FIFO) | Server pushes to specific provider |
| **Persistence** | SQL (full task lifecycle) | SQL registration, Redis results |
| **Results** | Side effects (modify room state) | Data returned to caller |
| **HTTP** | POST (creates task) | GET (reads resource) → 200 or 202 |
| **Examples** | Run modifier, run analysis | List files, fetch frame |

**Actions** (load file, mount source) are **jobs**, not provider reads. When a provider
registers, its action capabilities auto-register as jobs. The same Python client handles both.

---

## Architecture Overview

### Two Kinds of Capabilities

Every provider registration declares:

1. **Reads** — GET requests that return data (ephemeral, cached in Redis)
2. **Actions** — POST requests that create jobs (persistent, SQL task lifecycle)

```
Provider "local" (type=filesystem, scope=@global)
├── Reads (GET → 200 or 202)
│   GET /v1/joblib/rooms/{room_id}/providers/filesystem/local?path=/data&glob=*.xyz
│
└── Actions (POST → creates job task)
    POST /v1/joblib/rooms/{room_id}/tasks/filesystem:load
      {path: "/data/trajectory.xyz", target_room: "new-room"}
```

The frontend uses **GET for reads** (resource access) and **POST for actions** (via existing
job task submission). No new action infrastructure needed.

---

## Provider Pydantic Model (Client-Side)

Follows the same pattern as `Extension` — a Pydantic BaseModel with `ClassVar` metadata
and fields that define the read parameters.

```python
from pydantic import BaseModel
from typing import ClassVar, Any


class Provider(BaseModel):
    """Base class for provider resource definitions.

    Subclass to define read parameters as Pydantic fields.
    The JSON Schema is auto-generated for frontend discovery
    and query parameter validation.

    The ``type`` ClassVar groups providers by kind (e.g., all
    filesystem providers share ``type = "filesystem"``).
    """

    type: ClassVar[str]

    def read(self, handler: Any) -> Any:
        """Handle a read request.

        Parameters
        ----------
        handler
            The provider-specific handler object (e.g., an fsspec
            filesystem instance). Passed by the client at dispatch time.

        Returns
        -------
        Any
            JSON-serializable result data.
        """
        raise NotImplementedError
```

### Concrete Example: Filesystem (defined in ZnDraw, NOT in joblib)

```python
# In zndraw-fastapi (not joblib — joblib is type-agnostic)

class FilesystemRead(Provider):
    """Read parameters for filesystem providers.

    All fsspec filesystems provide the same interface, so this
    schema is shared across all filesystem provider instances.
    """

    type: ClassVar[str] = "filesystem"

    path: str = "/"
    glob: str | None = None
    recursive: bool = False

    def read(self, handler: AbstractFileSystem) -> list[dict[str, Any]]:
        if self.glob:
            pattern = f"{self.path.rstrip('/')}/{self.glob}"
            matches = handler.glob(pattern)
            return [_file_info(handler, m) for m in matches]
        return [_file_info(handler, f) for f in handler.ls(self.path, detail=True)]
```

### Concrete Example: Frame Source (defined in ZnDraw, NOT in joblib)

```python
class FrameSourceRead(Provider):
    """Read parameters for frame source providers."""

    type: ClassVar[str] = "frame_source"

    frame_id: int | None = None
    start: int | None = None
    stop: int | None = None
    keys: str | None = None  # comma-separated

    def read(self, handler: Any) -> list[dict[bytes, bytes]]:
        if self.frame_id is not None:
            return [handler[self.frame_id]]
        return handler[self.start : self.stop]
```

### JSON Schema Generation

The Provider model auto-generates a JSON Schema from its Pydantic fields:

```python
FilesystemRead.model_json_schema()
# {
#   "properties": {
#     "path": {"type": "string", "default": "/"},
#     "glob": {"type": "string", "default": null},
#     "recursive": {"type": "boolean", "default": false}
#   },
#   "required": []
# }
```

This schema is:
1. Stored on the server when the provider registers
2. Served to the frontend for discovery (`GET /v1/joblib/rooms/{room_id}/providers`)
3. Used to validate query parameters on read requests

---

## Provider SQL Model (Server-Side)

Providers are **persistently registered** with heartbeat-based liveness, just like workers.

```python
class ProviderRecord(SQLModel, table=True):
    """Server-side record of a registered provider."""

    __tablename__ = "provider"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    type: str = Field(index=True)              # "filesystem", "frame_source"
    name: str = Field(index=True)              # "local", "s3-bucket"
    scope: str = Field(index=True)             # "@global" or room_id
    schema_: dict = Field(sa_column=Column(JSON))  # JSON Schema from Provider model
    user_id: UUID = Field(foreign_key="user.id")
    worker_id: UUID | None = Field(            # Links to joblib Worker (for actions)
        default=None, foreign_key="worker.id"
    )
    sid: str                                   # Socket.IO SID for dispatch
    last_heartbeat: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)

    __table_args__ = (
        UniqueConstraint("type", "name", "scope"),
    )
```

### Why SQL (not just Redis)?

- Providers are "more permanently registered" — they outlive individual requests
- Heartbeat + sweeper needs queryable records (same pattern as Worker)
- `worker_id` FK links providers to the job system
- Cleanup on stale heartbeat mirrors worker cleanup

### Result Backend (Dependency-Injectable)

Read results are cached ephemerally — NOT persisted in SQL. The storage backend is
**dependency-injectable** so the host app can provide different backends per provider type.

```python
# In zndraw_joblib/dependencies.py

class ResultBackend(Protocol):
    """Protocol for provider result storage.

    Default implementation uses Redis. Host app can override
    per provider type (e.g., LMDB for large frame data).
    """

    async def store(
        self, provider_type: str, scope: str, name: str,
        request_hash: str, data: bytes, ttl: int,
    ) -> None: ...

    async def get(
        self, provider_type: str, scope: str, name: str,
        request_hash: str,
    ) -> bytes | None: ...

    async def delete(
        self, provider_type: str, scope: str, name: str,
        request_hash: str,
    ) -> None: ...


# Default: Redis (built into joblib)
class RedisResultBackend(ResultBackend):
    """Stores results in Redis with TTL. Suitable for small JSON data."""

    def __init__(self, redis: Redis, default_ttl: int = 300):
        self._redis = redis
        self._default_ttl = default_ttl

    async def store(self, provider_type, scope, name, request_hash, data, ttl):
        key = f"provider-result:{provider_type}:{scope}:{name}:{request_hash}"
        await self._redis.set(key, data, ex=ttl or self._default_ttl)

    async def get(self, provider_type, scope, name, request_hash):
        key = f"provider-result:{provider_type}:{scope}:{name}:{request_hash}"
        return await self._redis.get(key)
```

**Host app overrides per provider type** (e.g., frame data is too large for Redis):

```python
# In zndraw-fastapi lifespan:
class CompositeResultBackend(ResultBackend):
    """Routes to different backends based on provider type."""

    def __init__(self, backends: dict[str, ResultBackend], default: ResultBackend):
        self._backends = backends
        self._default = default

    async def store(self, provider_type, scope, name, request_hash, data, ttl):
        backend = self._backends.get(provider_type, self._default)
        await backend.store(provider_type, scope, name, request_hash, data, ttl)

    async def get(self, provider_type, scope, name, request_hash):
        backend = self._backends.get(provider_type, self._default)
        return await backend.get(provider_type, scope, name, request_hash)

# Wire it up:
app.dependency_overrides[get_result_backend] = lambda: CompositeResultBackend(
    backends={
        "frame_source": StorageResultBackend(lmdb_storage),  # large binary data
    },
    default=RedisResultBackend(redis),  # small JSON data (filesystem listings)
)
```

**Redis keys (for the default RedisResultBackend):**
- `provider-result:{type}:{scope}:{name}:{request_hash}` → cached result (configurable TTL)
- `provider-inflight:{type}:{scope}:{name}:{request_hash}` → inflight coalescing (SET NX, TTL=30s)

---

## REST Endpoints

All under `/v1/joblib/`:

### Provider CRUD

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `PUT` | `/providers` | Register/update provider (idempotent) |
| `GET` | `/rooms/{room_id}/providers` | List providers visible to room (room + @global) |
| `GET` | `/rooms/{room_id}/providers/{type}/{name}` | Get provider details + schema |
| `DELETE` | `/providers/{provider_id}` | Unregister provider |
| `PATCH` | `/providers/{provider_id}` | Heartbeat (updates `last_heartbeat`) |

### Provider Reads (the key new endpoint)

```
GET /v1/joblib/rooms/{room_id}/providers/{type}/{name}?{params}
```

Query parameters are validated against the provider's JSON Schema.

**Response:**
- `200 OK` — cached result available, returned immediately
- `202 Accepted` — dispatched to provider, result pending

```http
HTTP/1.1 202 Accepted
Location: /v1/joblib/rooms/{room_id}/providers/filesystem/local?path=/data
Retry-After: 2
Content-Type: application/json

{"status": "pending"}
```

### Provider Result Upload (worker → server)

```
POST /v1/joblib/providers/{provider_id}/results
  {request_hash: "...", data: {...}}
→ 204 No Content
```

Server stores result in Redis, emits `ProviderResultReady` via Socket.IO.

### Actions (via existing job system)

Actions are standard joblib tasks. No new endpoints needed:

```
POST /v1/joblib/rooms/{room_id}/tasks/filesystem:load
  {path: "/data/traj.xyz", target_room: "new-room", start: 0, stop: 100}
→ 202 Accepted
  Location: /v1/joblib/tasks/{task_id}
```

The job category matches the provider type (e.g., `filesystem`). When the worker claims the
task, it looks up its local provider handler to access the fsspec filesystem.

---

## Socket.IO Events

All frozen Pydantic models (same pattern as existing joblib events):

```python
# Registration lifecycle
class ProvidersInvalidate(FrozenEvent):
    """Provider list changed."""
    # Emitted to: room:{scope}

# Read dispatch
class ProviderRequest(FrozenEvent):
    """Server → Provider: process this read request."""
    request_id: str
    params: dict[str, Any]  # Validated against provider schema
    # Emitted to: provider:{type}:{scope}:{name}

class ProviderResultReady(FrozenEvent):
    """Server → Frontend: result cached, refetch."""
    provider_type: str
    provider_name: str
    request_hash: str
    # Emitted to: room:{scope}

# Client ↔ Server room management
class JoinProviderRoom(FrozenEvent):
    """Client joins provider dispatch room."""
    provider_id: str
    type: str
    name: str
    scope: str

class LeaveProviderRoom(FrozenEvent):
    """Client leaves provider dispatch room."""
    provider_id: str
    type: str
    name: str
    scope: str
```

---

## Client-Side API (in zndraw-joblib)

### ProviderManager

```python
class ProviderManager:
    """Manages provider registrations for a connected client.

    Works alongside JobManager — same client can register both
    providers (for reads) and jobs (for actions).
    """

    def __init__(self, api: ApiManager, tsio: SyncClientWrapper | None = None):
        self._api = api
        self._tsio = tsio
        self._providers: dict[tuple[str, str], _RegisteredProvider] = {}
        # key: (type, name) → handler + provider_id

    def register(
        self,
        provider_cls: type[Provider],
        *,
        name: str,
        handler: Any,
        scope: str = "@global",
        worker_id: UUID | None = None,
    ) -> UUID:
        """Register a provider.

        Parameters
        ----------
        provider_cls
            The Provider subclass (e.g., FilesystemRead).
        name
            Unique name for this provider instance.
        handler
            The handler object passed to ``provider.read(handler)``.
        scope
            "@global" or a room_id.
        worker_id
            Optional link to a joblib Worker (for action access).

        Returns
        -------
        UUID
            The provider ID.
        """
        schema = provider_cls.model_json_schema()
        # PUT /v1/joblib/providers
        response = self._api.http.put(
            f"{self._api.base_url}/v1/joblib/providers",
            json={
                "type": provider_cls.type,
                "name": name,
                "scope": scope,
                "schema": schema,
                "worker_id": str(worker_id) if worker_id else None,
            },
            headers=self._api.get_headers(),
        )
        self._api.raise_for_status(response)
        provider_id = response.json()["id"]

        # Store locally for dispatch handling
        self._providers[(provider_cls.type, name)] = _RegisteredProvider(
            id=provider_id,
            cls=provider_cls,
            handler=handler,
            scope=scope,
        )

        # Join Socket.IO room for dispatch
        if self._tsio is not None:
            self._tsio.emit(JoinProviderRoom(
                provider_id=str(provider_id),
                type=provider_cls.type,
                name=name,
                scope=scope,
            ))

        return provider_id

    def unregister(self, type: str, name: str) -> None:
        """Unregister a provider by type and name."""
        key = (type, name)
        if key not in self._providers:
            return
        reg = self._providers.pop(key)
        # DELETE /v1/joblib/providers/{id}
        self._api.http.delete(
            f"{self._api.base_url}/v1/joblib/providers/{reg.id}",
            headers=self._api.get_headers(),
        )
        # Leave Socket.IO room
        if self._tsio is not None:
            self._tsio.emit(LeaveProviderRoom(
                provider_id=str(reg.id),
                type=type,
                name=name,
                scope=reg.scope,
            ))

    @property
    def handlers(self) -> dict[str, Any]:
        """All registered handlers keyed by provider name.

        Used by job execution — the full dict is passed to Extension.run()
        so the Extension can look up the right handler by provider_name.
        """
        return {name: reg.handler for (_, name), reg in self._providers.items()}

    def disconnect(self) -> None:
        """Unregister all providers."""
        for type, name in list(self._providers):
            self.unregister(type, name)
```

### Handling Read Requests (on ProviderRequest event)

```python
def _on_provider_request(self, data: ProviderRequest) -> None:
    """Handle incoming read request from server."""
    key = (data.type, data.name)  # type and name from the SIO room
    reg = self._providers.get(key)
    if reg is None:
        return

    try:
        # Instantiate Provider model with the params (validates via Pydantic)
        instance = reg.cls(**data.params)
        result = instance.read(reg.handler)

        # POST result back to server
        self._api.http.post(
            f"{self._api.base_url}/v1/joblib/providers/{reg.id}/results",
            json={"request_hash": data.request_hash, "data": result},
            headers=self._api.get_headers(),
        )
    except Exception as e:
        log.exception("Provider request failed: %s", e)
```

---

## Job ↔ Provider Integration

When a provider registers action capabilities (e.g., `LoadFile`, `MountFile`), those are
registered as **standard joblib jobs** with the provider's type as the job category.

The `worker_id` field on `ProviderRecord` links the provider to the worker that handles
these action jobs. When the worker claims a task, it passes the **full providers dict**
so the Extension can look up the right handler by name:

```python
# In ZnDraw client (NOT in joblib)
def _execute_task(self, task: ClaimedTask) -> None:
    # Pass all registered provider handlers to the Extension.
    # The Extension knows which provider to use via its provider_name field.
    task.extension.run(vis, providers=self.providers.handlers)
```

The `ProviderManager` exposes all handlers as a simple dict:

```python
# In ProviderManager
@property
def handlers(self) -> dict[str, Any]:
    """All registered handlers keyed by provider name.

    Returns a dict like {"local": <fsspec.LocalFileSystem>, "s3": <S3FileSystem>}.
    Extensions look up the handler they need by provider_name.
    """
    return {name: reg.handler for (_, name), reg in self._providers.items()}
```

This means a `LoadFile` Extension receives the full dict and picks the right filesystem
by `provider_name` (which comes from the task payload — the user selected it in the UI):

```python
class LoadFile(Extension):
    category: ClassVar[str] = "filesystem"
    provider_name: str       # "local" or "s3-bucket" — set by frontend
    path: str
    target_room: str
    start: int | None = None
    stop: int | None = None
    step: int | None = None

    def run(self, vis: Any, *, providers: dict[str, Any], **kwargs: Any) -> None:
        import ase.io

        fs = providers[self.provider_name]  # look up the right filesystem
        atoms_list = ase.io.read(
            fs.open(self.path),
            index=slice(self.start, self.stop, self.step),
        )
        target = ZnDraw(url=vis.url, room=self.target_room)
        target.extend(atoms_list)
        target.disconnect()
```

This approach works cleanly when multiple providers of the same type are registered
(e.g., "local" and "s3-bucket" filesystem providers on the same client).

---

## Lifecycle & Cleanup

### Heartbeat (same pattern as Worker)

```
PATCH /v1/joblib/providers/{provider_id}
→ updates last_heartbeat
```

Client sends heartbeat on the same interval as worker heartbeat. If a provider shares a
`worker_id`, the client can send one heartbeat for both.

### Sweeper (extends existing sweeper)

Add `cleanup_stale_providers()` alongside `cleanup_stale_workers()`:

```python
async def cleanup_stale_providers(
    session: AsyncSession,
    timeout: timedelta,
) -> tuple[int, set[Emission]]:
    """Find providers with stale heartbeats and remove them."""
    threshold = utcnow() - timeout
    stale = await session.exec(
        select(ProviderRecord)
        .where(ProviderRecord.last_heartbeat < threshold)
    )
    emissions: set[Emission] = set()
    for provider in stale:
        session.delete(provider)
        emissions.add(Emission(
            ProvidersInvalidate(),
            f"room:{provider.scope}",
        ))
    return len(stale), emissions
```

### Disconnect Cleanup

When a Socket.IO connection drops, the host app's disconnect handler calls:

```python
# In zndraw-fastapi socketio.py on_disconnect:
providers = await get_providers_by_sid(session, sid)
for provider in providers:
    await session.delete(provider)
    emissions.add(...)
```

### Internal Providers (`@internal`)

Like `@internal` jobs, internal providers run server-side via taskiq. The server itself
registers as the provider — no external Python client needed.

```python
# In zndraw-fastapi lifespan:
register_internal_providers(app, broker, [
    LocalFilesystemRead,  # Server can list its own files
])
```

Internal providers have no SID, no heartbeat (they're the server itself), and reads are
dispatched to taskiq instead of Socket.IO.

---

## Settings

Extend `JobLibSettings` (not a separate settings class):

```python
class JobLibSettings(BaseSettings):
    # ... existing settings ...

    # Provider settings
    provider_result_ttl_seconds: int = 300
    provider_inflight_ttl_seconds: int = 30
    provider_stale_timeout_seconds: int = 30
```

---

## Read Request Flow (Complete)

```
1. Frontend
   GET /v1/joblib/rooms/{room_id}/providers/filesystem/local?path=/data&glob=*.xyz
                                    │
2. Server                           │
   ├─ Validate params against schema (Pydantic)
   ├─ Build request_hash from params
   ├─ Check Redis cache: provider-result:filesystem:@global:local:{hash}
   │  ├─ HIT → return 200 with cached data
   │  └─ MISS ↓
   ├─ Check inflight: provider-inflight:..:{hash}
   │  ├─ Already inflight → return 202 (someone else dispatched)
   │  └─ New → SET NX, emit ProviderRequest to provider:filesystem:@global:local
   └─ Return 202 Accepted
        Location: (same URL)
        Retry-After: 2
                                    │
3. Provider (Python client)          │
   ├─ Receives ProviderRequest via Socket.IO
   ├─ Instantiates FilesystemRead(path="/data", glob="*.xyz")
   ├─ Calls .read(fs_handler) → list[FileItem]
   └─ POST /v1/joblib/providers/{id}/results
        {request_hash: "...", data: [...]}
                                    │
4. Server                           │
   ├─ Store via ResultBackend (Redis for fs, LMDB for frames, etc.)
   ├─ Clear inflight key
   └─ Emit ProviderResultReady to room:@global
                                    │
5. Frontend                          │
   ├─ Receives ProviderResultReady via Socket.IO
   ├─ Invalidates React Query cache
   └─ GET same URL → 200 with cached data
```

---

## Action Flow (Complete)

```
1. Frontend
   POST /v1/joblib/rooms/{room_id}/tasks/filesystem:load
     {path: "/data/traj.xyz", target_room: "new-room"}
   → 202 Accepted, Location: /v1/joblib/tasks/{task_id}
                                    │
2. Server (existing joblib)          │
   ├─ Creates Task record (PENDING)
   ├─ Emits TaskAvailable to jobs:@global:filesystem:load
   └─ Returns 202
                                    │
3. Worker (same Python client)       │
   ├─ Receives TaskAvailable via Socket.IO
   ├─ Claims task: POST /v1/joblib/tasks/claim
   ├─ Instantiates LoadFile(provider_name="local", path=..., target_room=...)
   ├─ Calls extension.run(vis, providers=self.providers.handlers)
   │  ├─ Extension does: fs = providers["local"]
   │  ├─ Reads file via fsspec
   │  └─ Uploads frames to target room
   └─ PATCH /v1/joblib/tasks/{id} → COMPLETED
                                    │
4. Server                           │
   └─ Emits TaskStatusEvent to room:{room_id}
```

---

## Summary

| Component | Location | What it provides |
|-----------|----------|------------------|
| `Provider` base model | `zndraw_joblib.provider` | Pydantic base class, schema generation |
| `ProviderRecord` SQL model | `zndraw_joblib.models` | Registration persistence, heartbeat |
| Provider REST endpoints | `zndraw_joblib.router` | CRUD, read dispatch, result upload |
| Provider events | `zndraw_joblib.events` | Socket.IO models for dispatch + invalidation |
| `ProviderManager` client | `zndraw_joblib.client` | Register, unregister, handle requests |
| Sweeper extension | `zndraw_joblib.sweeper` | Stale provider cleanup |
| `FilesystemRead` | `zndraw` (host app) | Filesystem-specific provider type |
| `FrameSourceRead` | `zndraw` (host app) | Frame source-specific provider type |
| `LoadFile`, `MountFile` | `zndraw` (host app) | Action Extensions (standard jobs) |
| `vis.register_fs()` | `zndraw` (host app) | Convenience API combining provider + worker |

### Key Design Principles

1. **Joblib is type-agnostic** — knows nothing about filesystems or frames
2. **Pydantic everywhere** — Provider models, event models, request validation
3. **Reads = GET, Actions = POST** — REST-compliant resource access
4. **SQL for registrations** — providers are persistent with heartbeat lifecycle
5. **Injectable result backend** — Redis default, host app overrides per type (e.g., LMDB for frames)
6. **Same heartbeat/sweeper** — mirrors existing Worker pattern
7. **Jobs access all providers** — `providers: dict[str, Any]` passed to Extension.run()
8. **Internal providers** — server-side via taskiq (like `@internal` jobs)
9. **Single registration** — `vis.register_fs()` creates provider + worker + jobs
