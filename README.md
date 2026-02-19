# ZnDraw Job Management Library

A self-contained FastAPI package for distributed job/task management with SQL persistence.

## Integration into your APP

```python
# main.py
from fastapi import FastAPI
from zndraw_auth import current_active_user, current_superuser
from zndraw_auth.db import get_session_maker
from zndraw_joblib.router import router
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.sweeper import run_sweeper
from zndraw_joblib.settings import JobLibSettings

app = FastAPI()

# 1. Override session maker dependency at auth level
#    All database access (from auth and joblib) flows through this
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
engine = create_async_engine("sqlite+aiosqlite:///./app.db")
my_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
app.dependency_overrides[get_session_maker] = lambda: my_session_maker

# 2. Override auth dependencies (from zndraw_auth)
# app.dependency_overrides[current_active_user] = my_get_current_user
# app.dependency_overrides[current_superuser] = my_get_superuser

# 3. Register exception handler and router
app.add_exception_handler(ProblemException, problem_exception_handler)
app.include_router(router)

# 4. Start background sweeper
async def get_session():
    async with my_session_maker() as session:
        yield session

settings = JobLibSettings()
# asyncio.create_task(run_sweeper(get_session=get_session, settings=settings))
```

### Dependency Architecture

All database access flows through `zndraw_auth.db.get_session_maker`:

```
get_session_maker (from zndraw_auth)  ← override this one dependency
  ├─ SessionDep (regular endpoints)
  └─ SessionMakerDep (long-polling endpoints)
```

| Dependency | Override? | Purpose |
|------------|-----------|---------|
| `get_session_maker` | **Yes** | Single source of truth for all DB sessions (from zndraw_auth) |
| `current_active_user` | Yes (from zndraw_auth) | Authenticated user identity |
| `current_superuser` | Yes (from zndraw_auth) | Superuser access control |
| `verify_writable_room` | Optional | Room writability guard for `register_job` and `submit_task` |
| `get_tsio` | Optional | Socket.IO server for real-time events |
| `get_result_backend` | **Yes** (for providers) | Result caching backend for provider reads |
| `get_settings` | Optional | Override `JobLibSettings` defaults |

**Note**: SQLite locking is handled by the host application (zndraw-fastapi). For SQLite databases, wrap the session maker with a lock in your app's lifespan context.

### Room Writability Guard

The `verify_writable_room` dependency guards write endpoints (`register_job`, `submit_task`). By default it only validates the `room_id` format. Host apps can override it to add lock checks:

```python
from fastapi import Path
from zndraw_joblib import verify_writable_room, validate_room_id

async def get_writable_room(
    session: SessionDep,
    current_user: CurrentUserDep,
    redis: RedisDep,
    room_id: str = Path(),
) -> str:
    validate_room_id(room_id)  # format validation (@ and : checks)
    room = await verify_room(session, room_id)
    if room.locked and not current_user.is_superuser:
        raise HTTPException(status_code=423, detail="Room is locked")
    # ... additional checks (edit lock, etc.) ...
    return room_id

app.dependency_overrides[verify_writable_room] = get_writable_room
```

Read endpoints and existing task/worker operations (updates, heartbeats, disconnects) are **not** affected by this guard.

## Configuration

Settings via environment variables with `ZNDRAW_JOBLIB_` prefix:

```python
class JobLibSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZNDRAW_JOBLIB_")

    allowed_categories: list[str] = ["modifiers", "selections", "analysis"]
    worker_timeout_seconds: int = 60
    sweeper_interval_seconds: int = 30
    long_poll_max_wait_seconds: int = 120

    # Provider settings
    provider_result_ttl_seconds: int = 300    # cached result lifetime
    provider_inflight_ttl_seconds: int = 30   # inflight lock lifetime
```

## Job Naming Convention

Jobs use the format: `<room_id>:<category>:<name>`

- `@global:modifiers:Rotate` - global job available to all rooms
- `room_123:modifiers:Rotate` - private job for room_123 only

**Validation rules:**
- `room_id` cannot contain `@` (reserved for `@global`) or `:` (delimiter)
- `category` must be in `settings.allowed_categories`
- Same job name in same room: schema must match (409 Conflict otherwise)
- Different rooms can have same job name with different schemas

## REST Endpoints

All endpoints prefixed with `/v1/joblib` (hardcoded in router).

### Job Registration

```
PUT /v1/joblib/rooms/{room_id}/jobs
```

Register a job for a room. Use `@global` as `room_id` for global jobs.

Request body:
```json
{
  "category": "modifiers",
  "name": "Rotate",
  "schema": { ... }
}
```

- `@global` registration requires admin (`get_is_admin`)
- Re-registering validates schema match (409 if mismatch)
- Creates Worker and WorkerJobLink if not exists

### Job Listing

```
GET /v1/joblib/rooms/{room_id}/jobs     # room + @global jobs
GET /v1/joblib/rooms/@global/jobs       # only global jobs
```

### Job Details

```
GET /v1/joblib/rooms/{room_id}/jobs/{job_name}
```

`job_name` format: `<room_id>:<category>:<name>`

### Task Submission

```
POST /v1/joblib/rooms/{room_id}/tasks/{job_name}
```

Returns `202 Accepted` with `Location` header pointing to task status.

Request body:
```json
{
  "payload": { "angle": 90 }
}
```

### Task Claim (Worker)

```
POST /v1/joblib/tasks/claim
```

Claims oldest pending task (FIFO) across all jobs the worker is registered for:

```sql
SELECT task.* FROM task
JOIN worker_job_link ON task.job_id = worker_job_link.job_id
WHERE worker_job_link.worker_id = :worker_id
  AND task.status = 'pending'
ORDER BY task.created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED
```

### Task Status

```
GET /v1/joblib/tasks/{task_id}
```

Supports long-polling via `Prefer: wait=N` header (max 120s). Returns immediately on terminal states.

### Task Update

```
PATCH /v1/joblib/tasks/{task_id}
```

Valid transitions:
- `pending` → `claimed` (via `/claim`)
- `claimed` → `running`
- `running` → `completed` | `failed`
- Any → `cancelled`


### Worker Heartbeat

```
PATCH /v1/joblib/workers/{worker_id}
```

Updates `last_heartbeat`. Workers must call periodically during execution.

### Worker Deletion

```
DELETE /v1/joblib/workers/{worker_id}
```

Explicit cleanup on graceful shutdown. Also cascades to any providers registered under that worker.

## Providers

Providers are a generic abstraction for connected Python clients to **serve data on demand**. While jobs are user-initiated computation (workers pull tasks), providers handle **server-dispatched read requests** — the server pushes a request to a specific provider and caches the result.

### Providers vs Jobs

| | **Jobs** | **Providers** |
|---|---|---|
| **Purpose** | User-initiated computation | Remote resource access |
| **Dispatch** | Workers pull/claim (FIFO) | Server pushes to specific provider |
| **Results** | Side effects (modify room state) | Data returned to caller (cached) |
| **HTTP** | POST (creates task) | GET (reads resource) → 200 or 202 |

### Provider Base Model

Host apps define provider types by subclassing `Provider`:

```python
from zndraw_joblib import Provider

class FilesystemRead(Provider):
    category: ClassVar[str] = "filesystem"
    path: str = "/"
    glob: str | None = None

    def read(self, handler: Any) -> Any:
        if self.glob:
            return handler.glob(f"{self.path}/{self.glob}")
        return handler.ls(self.path, detail=True)
```

The JSON Schema is auto-generated from Pydantic fields and stored on registration.

### Result Backend

Provider reads require a `ResultBackend` for caching results and inflight coalescing. The host app **must** override `get_result_backend`:

```python
from zndraw_joblib.dependencies import get_result_backend, ResultBackend

class RedisResultBackend:
    """Implements the ResultBackend protocol using Redis."""

    def __init__(self, redis):
        self._redis = redis

    async def store(self, key: str, data: bytes, ttl: int) -> None:
        await self._redis.set(key, data, ex=ttl)

    async def get(self, key: str) -> bytes | None:
        return await self._redis.get(key)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def acquire_inflight(self, key: str, ttl: int) -> bool:
        return await self._redis.set(key, b"1", nx=True, ex=ttl)

    async def release_inflight(self, key: str) -> None:
        await self._redis.delete(key)

backend = RedisResultBackend(redis)
app.dependency_overrides[get_result_backend] = lambda: backend
```

### Provider REST Endpoints

All under `/v1/joblib/`:

#### Registration

```
PUT /v1/joblib/rooms/{room_id}/providers
```

Register or update a provider. Idempotent on `(room_id, category, name)`. Auto-creates a worker if `worker_id` is not provided.

Request body:
```json
{
  "category": "filesystem",
  "name": "local",
  "schema": { "path": {"type": "string"} }
}
```

#### Listing

```
GET /v1/joblib/rooms/{room_id}/providers
```

Lists providers visible from a room (room-scoped + `@global`). Paginated.

#### Provider Info

```
GET /v1/joblib/rooms/{room_id}/providers/{provider_name}/info
```

Returns provider details and JSON Schema. `provider_name` is the full name (e.g., `@global:filesystem:local`).

#### Data Read

```
GET /v1/joblib/rooms/{room_id}/providers/{provider_name}?params
```

Query parameters are passed to the provider's `read()` method. `provider_name` is the full name.

**Responses:**
- `200 OK` — cached result available, returned immediately
- `202 Accepted` — dispatched to provider, result pending. Includes `Location` and `Retry-After` headers.

#### Result Upload

```
POST /v1/joblib/providers/{provider_id}/results
```

Provider worker posts read results back to the server. Stores in `ResultBackend`, clears inflight lock, emits `ProviderResultReady`.

```json
{"request_hash": "abc123...", "data": [...]}
```

#### Deletion

```
DELETE /v1/joblib/providers/{provider_id}
```

Unregister a provider. Must be owned by authenticated user or superuser.

### Read Request Flow

```
1. Frontend: GET /rooms/room-42/providers/@global:filesystem:local?path=/data
2. Server: check cache → HIT: return 200
                        → MISS: acquire inflight, emit ProviderRequest → return 202
3. Provider client: receives ProviderRequest via Socket.IO
                    calls provider.read(handler)
                    POST /providers/{id}/results
4. Server: store in ResultBackend, emit ProviderResultReady
5. Frontend: receives ProviderResultReady, re-fetches → 200
```

### Client SDK

Provider methods are integrated into `JobManager`:

```python
with JobManager(api, tsio=tsio) as manager:
    # Register a provider
    provider_id = manager.register_provider(
        FilesystemRead,
        name="local",
        handler=fsspec.filesystem("file"),
        room="@global",
    )

    # Access all handlers (used during job execution)
    print(manager.handlers)  # {"@global:filesystem:local": <LocalFileSystem>}

    # Unregister by full_name
    manager.unregister_provider("@global:filesystem:local")
# disconnect() cleans up both providers and jobs
```

## SQLModel

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Any
from uuid import uuid4
import uuid

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON
from sqlmodel import SQLModel, Field, Relationship


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerJobLink(SQLModel, table=True):
    """Bare M:N link between Worker and Job."""
    worker_id: str = Field(foreign_key="worker.id", primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="job.id", primary_key=True)


class Job(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("room_id", "category", "name", name="unique_job"),
    )

    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: str = Field(index=True)      # "@global" or "room_123"
    category: str = Field(index=True)     # validated against allowed list
    name: str = Field(index=True)         # e.g., "Rotate"
    schema: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Relationships
    tasks: list["Task"] = Relationship(back_populates="job")
    workers: list["Worker"] = Relationship(back_populates="jobs", link_model=WorkerJobLink)

    @property
    def full_name(self) -> str:
        return f"{self.room_id}:{self.category}:{self.name}"


class Worker(SQLModel, table=True):
    id: str = Field(primary_key=True)  # from JWT identity
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Relationships
    jobs: list[Job] = Relationship(back_populates="workers", link_model=WorkerJobLink)
    tasks: list["Task"] = Relationship(back_populates="worker")

    def is_alive(self, threshold: timedelta) -> bool:
        return datetime.utcnow() - self.last_heartbeat < threshold


class Task(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)

    job_id: uuid.UUID = Field(foreign_key="job.id", index=True)
    job: Job = Relationship(back_populates="tasks")

    worker_id: Optional[str] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship(back_populates="tasks")

    room_id: str = Field(index=True)  # room where task was submitted
    created_by_id: Optional[str] = Field(default=None, index=True)

    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: TaskStatus = Field(default=TaskStatus.PENDING, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
```

## Long-Polling

`GET /v1/joblib/tasks/{task_id}` with `Prefer: wait=N`:
- Server polls database until task reaches terminal state or timeout
- Returns immediately on terminal state (`completed`, `failed`, `cancelled`)
- Returns `Preference-Applied: wait` header if honored
- Maximum wait capped by `settings.long_poll_max_wait_seconds`

## Socket.IO Real-Time Events

The package emits real-time events via [zndraw-socketio](https://github.com/zincware/zndraw-socketio). The host app provides its `AsyncServerWrapper` through dependency injection:

```python
from zndraw_socketio import wrap, AsyncServerWrapper
from zndraw_joblib.dependencies import get_tsio

tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

app.dependency_overrides[get_tsio] = lambda: tsio
```

When `get_tsio` returns `None` (default), all event emissions are skipped.

### Event Models

All models are frozen Pydantic `BaseModel`s (hashable for set-based deduplication).

| Event | Payload | Room Target | Trigger |
|-------|---------|-------------|---------|
| `JobsInvalidate` | *(none)* | `room:{room_id}` | Job registered/deleted, worker connected/disconnected |
| `TaskAvailable` | `job_name`, `room_id`, `task_id` | `jobs:{full_name}` | Task submitted (non-`@internal` only) |
| `TaskStatusEvent` | `id`, `name`, `room_id`, `status`, timestamps, `worker_id`, `error` | `room:{room_id}` | Any task status transition |
| `JoinJobRoom` | `job_name`, `worker_id` | *(client → server)* | Worker joins a job's notification room |
| `LeaveJobRoom` | `job_name`, `worker_id` | *(client → server)* | Worker leaves a job's notification room |
| `ProvidersInvalidate` | *(none)* | `room:{room_id}` | Provider registered/deleted, worker disconnected |
| `ProviderRequest` | `request_id`, `provider_name`, `params` | `providers:{full_name}` | Server dispatches read to provider |
| `ProviderResultReady` | `provider_name`, `request_hash` | `room:{room_id}` | Provider result cached, frontend should refetch |
| `JoinProviderRoom` | `provider_name`, `worker_id` | *(client → server)* | Client joins provider dispatch room |
| `LeaveProviderRoom` | `provider_name`, `worker_id` | *(client → server)* | Client leaves provider dispatch room |

Event names are auto-derived as snake_case by zndraw-socketio: `jobs_invalidate`, `task_available`, `task_status_event`, `providers_invalidate`, `provider_request`, etc.

### Worker Notification Pattern

Workers (e.g., the `ZnDraw` client class) register jobs via REST, then join the socketio room to receive task notifications:

```python
# 1. Register job via REST
client.put("/v1/joblib/rooms/@global/jobs", json={...})

# 2. Join the job's socketio room
await sio.emit(JoinJobRoom(job_name="@global:modifiers:Rotate", worker_id="..."))

# 3. Receive TaskAvailable when tasks are submitted
@sio.on(TaskAvailable)
async def on_task_available(sid: str, data: TaskAvailable):
    await worker.claim_and_run(data.job_name)
```

The host app registers the room join/leave handlers and stores the `worker_id` in the SIO session for disconnect cleanup:

```python
@tsio.on(JoinJobRoom)
async def handle_join(sid: str, data: JoinJobRoom):
    await tsio.enter_room(sid, f"jobs:{data.job_name}")
    session = await tsio.get_session(sid)
    session["worker_id"] = data.worker_id
    await tsio.save_session(sid, session)

@tsio.on(LeaveJobRoom)
async def handle_leave(sid: str, data: LeaveJobRoom):
    await tsio.leave_room(sid, f"jobs:{data.job_name}")
```

### Server-Side Disconnect Cleanup

When a worker's Socket.IO connection drops (crash, network loss), the host app can
immediately clean up by calling `cleanup_worker` from its existing disconnect handler.
The `worker_id` stored in the SIO session during `JoinJobRoom` enables the mapping:

```python
from zndraw_joblib import cleanup_worker, emit

@tsio.on("disconnect")
async def on_disconnect(sid: str, reason: str):
    session = await tsio.get_session(sid)

    # ... existing cleanup (presence, locks, etc.) ...

    # Worker cleanup
    worker_id = session.get("worker_id")
    if worker_id:
        async with get_session() as db:
            worker = await db.get(Worker, UUID(worker_id))
            if worker:
                emissions = await cleanup_worker(db, worker)
                await db.commit()
                await emit(tsio, emissions)
```

This provides immediate cleanup (fail stuck tasks, remove job links, soft-delete
orphan jobs) without waiting for the background sweeper's heartbeat timeout.

| Disconnect Scenario | Handler |
|---------------------|---------|
| Network drop / process kill | Server-side SIO disconnect (immediate) |
| Graceful shutdown (`with manager:`) | Client `disconnect()` emits `LeaveJobRoom` + calls `DELETE /workers` |
| REST-only workers (no SIO) | Background sweeper heartbeat timeout |

### Emission Deduplication

Internally, emissions are `Emission(NamedTuple)` pairs of `(event, room)`. Functions that modify state return `set[Emission]`, and callers emit **after commit**. Frozen models ensure duplicate events (e.g., multiple workers disconnecting from the same job) are deduplicated automatically.

## Background Sweeper

Host app starts explicitly:

```python
from zndraw_joblib import run_sweeper, get_settings

asyncio.create_task(
    run_sweeper(get_session=my_session_factory, settings=get_settings(), tsio=tsio)
)
```

The sweeper:
1. Finds workers with stale `last_heartbeat`
2. Marks their `running`/`claimed` tasks as `FAILED`
3. Removes orphan jobs (no workers, no pending tasks)
4. Emits `TaskStatusEvent` and `JobsInvalidate` events after each cleanup cycle

## Error Handling (RFC 9457)

```python
class JobNotFound(ProblemType):
    title = "Not Found"
    status = 404

class SchemaConflict(ProblemType):
    title = "Conflict"
    status = 409

class InvalidCategory(ProblemType):
    title = "Bad Request"
    status = 400

class WorkerNotFound(ProblemType):
    title = "Not Found"
    status = 404

class TaskNotFound(ProblemType):
    title = "Not Found"
    status = 404

class InvalidTaskTransition(ProblemType):
    title = "Conflict"
    status = 409

class ProviderNotFound(ProblemType):
    title = "Not Found"
    status = 404
```

## Client

```python
import time
import httpx
from datetime import datetime
from typing import Any, Protocol
from pydantic import BaseModel


class ApiManager(Protocol):
    http: httpx.Client
    def get_headers(self) -> dict[str, str]: ...
    @property
    def base_url(self) -> str: ...


class ClaimedTask(BaseModel):
    """Pydantic model for a claimed task."""
    id: str
    job_name: str  # full name: room_id:category:name
    room_id: str
    payload: dict[str, Any]
    created_at: datetime


class TaskStream:
    """Iterator for consuming tasks."""
    def __init__(self, api: ApiManager, worker_id: str, polling_interval: float = 2.0):
        self.api = api
        self.worker_id = worker_id
        self.interval = polling_interval
        self._stop_event = False

    def __iter__(self):
        return self

    def __next__(self) -> ClaimedTask:
        if self._stop_event:
            raise StopIteration

        while True:
            response = self.api.http.post(
                f"{self.api.base_url}/v1/joblib/tasks/claim",
                headers=self.api.get_headers(),
            )
            data = response.json()
            if data.get("task"):
                return ClaimedTask.model_validate(data["task"])
            time.sleep(self.interval)

    def stop(self):
        self._stop_event = True


class JobManager:
    """Main entry point. Behaves like a dictionary of registered jobs."""

    def __init__(self, api: ApiManager):
        self.api = api
        self._registry: dict[str, type[BaseModel]] = {}

    def __getitem__(self, key: str):
        return self._registry[key]

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def register(self, task_class: type[BaseModel] | None = None, *, room: str | None = None):
        """
        Register a job. room=None defaults to @global.

        Extension classes are Pydantic BaseModels. Schema is derived via model_json_schema().

        Usage:
            @vis.jobs.register
            class Rotate(BaseModel):
                angle: float = 0.0

            @vis.jobs.register(room="room_123")
            class PrivateRotate(BaseModel):
                angle: float = 0.0

            vis.jobs.register(MyJob, room="room_123")
        """
        def decorator(cls: type[BaseModel]):
            self._register_impl(cls, room)
            return cls

        if task_class is None:
            return decorator
        return decorator(task_class)

    def _register_impl(self, cls: type[BaseModel], room: str | None):
        room_id = room if room is not None else "@global"
        category = getattr(cls, "category", "modifiers")
        name = cls.__name__

        # Schema from Pydantic model
        schema = cls.model_json_schema()

        self.api.http.put(
            f"{self.api.base_url}/v1/joblib/rooms/{room_id}/jobs",
            headers=self.api.get_headers(),
            json={"category": category, "name": name, "schema": schema},
        )

        full_name = f"{room_id}:{category}:{name}"
        self._registry[full_name] = cls

    def listen(self, interval: float = 2.0) -> TaskStream:
        """Returns an iterator to process tasks."""
        worker_id = self._get_worker_id()
        return TaskStream(self.api, worker_id, interval)

    def _get_worker_id(self) -> str:
        response = self.api.http.get(
            f"{self.api.base_url}/v1/me",
            headers=self.api.get_headers(),
        )
        return response.json()["id"]

    def heartbeat(self):
        """Manually send worker heartbeat. Call during long task execution."""
        worker_id = self._get_worker_id()
        self.api.http.patch(
            f"{self.api.base_url}/v1/joblib/workers/{worker_id}",
            headers=self.api.get_headers(),
        )
```

### Usage Example

```python
from pydantic import BaseModel

# Context manager for automatic cleanup on shutdown
with JobManager(api, tsio=tsio) as manager:
    # 1. Register jobs
    @manager.register  # @global by default
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    @manager.register(room="room_123")
    class PrivateRotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    print(len(manager))  # 2 Jobs Registered

    # 2. Consume tasks
    for task in manager.listen(stop_event=shutdown):
        manager.heartbeat()  # keep alive during processing
        task.extension.run()
# disconnect() called automatically: LeaveJobRoom emitted, DELETE /workers called
```

### Notes

- Extension classes are **Pydantic BaseModels** with a `category` ClassVar and `run()` method
- Schema is generated via `cls.model_json_schema()`
- Re-registering validates schema match (raises on mismatch)
- `room=None` (default) registers to `@global`
- Worker must call `heartbeat()` explicitly during long-running tasks
- Use `with JobManager(...) as manager:` for automatic cleanup on shutdown
- Or call `manager.disconnect()` explicitly for manual cleanup
