# ZnDraw Job Management Library

A self-contained FastAPI package for distributed job/task management with SQL persistence. Provides a pluggable router, ORM models, a client SDK with auto-serve, provider-based data reads, and server-side taskiq workers.

## Integration into your APP

```python
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
get_session_maker (from zndraw_auth)  <- override this one dependency
  +- SessionDep (regular endpoints)
  +- SessionMakerDep (long-polling endpoints)
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

**Note**: SQLite locking is handled by the host application. For SQLite databases, wrap the session maker with a lock in your app's lifespan context.

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
    return room_id

app.dependency_overrides[verify_writable_room] = get_writable_room
```

Read endpoints and existing task/worker operations (updates, heartbeats, disconnects) are **not** affected by this guard.

## Configuration

Settings via environment variables with `ZNDRAW_JOBLIB_` prefix:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZNDRAW_JOBLIB_ALLOWED_CATEGORIES` | `["modifiers", "selections", "analysis"]` | Valid job categories |
| `ZNDRAW_JOBLIB_WORKER_TIMEOUT_SECONDS` | `60` | Stale heartbeat threshold |
| `ZNDRAW_JOBLIB_SWEEPER_INTERVAL_SECONDS` | `30` | Sweeper cycle interval |
| `ZNDRAW_JOBLIB_LONG_POLL_MAX_WAIT_SECONDS` | `60` | Max long-poll wait |
| `ZNDRAW_JOBLIB_CLAIM_MAX_ATTEMPTS` | `10` | Retries for concurrent claim contention |
| `ZNDRAW_JOBLIB_CLAIM_BASE_DELAY_SECONDS` | `0.01` | Exponential backoff base delay |
| `ZNDRAW_JOBLIB_INTERNAL_TASK_TIMEOUT_SECONDS` | `3600` | Timeout for stuck `@internal` tasks |
| `ZNDRAW_JOBLIB_ALLOWED_PROVIDER_CATEGORIES` | `None` (unrestricted) | Valid provider categories |
| `ZNDRAW_JOBLIB_PROVIDER_RESULT_TTL_SECONDS` | `300` | Cached provider result lifetime |
| `ZNDRAW_JOBLIB_PROVIDER_INFLIGHT_TTL_SECONDS` | `30` | Inflight lock lifetime |

## Job Naming Convention

Jobs use the format: `<room_id>:<category>:<name>`

- `@global:modifiers:Rotate` - global job available to all rooms
- `room_123:modifiers:Rotate` - private job for room_123 only
- `@internal:modifiers:Rotate` - server-side job executed via taskiq

**Validation rules:**
- `room_id` cannot contain `@` (reserved for `@global`/`@internal`) or `:` (delimiter)
- `category` must be in `settings.allowed_categories`
- Same job name in same room: schema must match (409 Conflict otherwise)
- Different rooms can have same job name with different schemas

## Client SDK

The `JobManager` is the main entry point for Python workers. It handles job registration, task claiming, provider dispatch, and background lifecycle management.

### Basic Usage

```python
from zndraw_joblib import JobManager, Extension, Category

# Auto-serve mode: background threads claim and execute tasks
with JobManager(api, tsio=tsio, execute=my_execute) as manager:
    @manager.register
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

        def run(self, vis, **kwargs):
            # modify vis based on self.angle
            pass

    manager.wait()  # blocks until SIGINT/SIGTERM or disconnect()
# disconnect() called automatically: threads joined, worker deleted
```

### Extension Classes

Extensions are Pydantic models with a `category` ClassVar and a `run()` method:

```python
from typing import ClassVar, Any
from zndraw_joblib import Extension, Category

class Rotate(Extension):
    category: ClassVar[Category] = Category.MODIFIER  # or SELECTION, ANALYSIS
    angle: float = 0.0
    axis: str = "z"

    def run(self, vis: Any, **kwargs: Any) -> None:
        # Implementation here
        pass
```

The JSON Schema is auto-generated from Pydantic fields and sent to the server on registration.

### Auto-Serve Mode

When an `execute` callback is provided, `JobManager` runs background threads that automatically claim and execute tasks:

```python
from zndraw_joblib import JobManager, ClaimedTask

def execute(task: ClaimedTask) -> None:
    """Called for each claimed task."""
    task.extension.run(vis)

manager = JobManager(
    api,
    tsio=tsio,
    execute=execute,
    polling_interval=2.0,      # how often to poll for tasks (seconds)
    heartbeat_interval=30.0,   # heartbeat frequency (seconds)
)
```

Background threads start on the first `register()` or `register_provider()` call:
- **Heartbeat thread** - periodic keep-alives to prevent sweeper cleanup
- **Claim loop thread** - polls for tasks, calls `execute`, marks completed/failed

The lifecycle is fully managed: `start()` is called before execute, `complete()` or `fail()` after. Exceptions in `execute` mark the task as failed with the error message.

### Manual Mode

Without `execute`, tasks must be claimed and processed manually:

```python
manager = JobManager(api, tsio=tsio)

@manager.register
class Rotate(Extension):
    category: ClassVar[Category] = Category.MODIFIER
    angle: float = 0.0
    def run(self, vis, **kwargs): ...

# Manual claim-execute loop
for task in manager.listen(polling_interval=2.0):
    manager.start(task)
    try:
        task.extension.run(vis)
        manager.complete(task)
    except Exception as e:
        manager.fail(task, str(e))
```

### Task Submission

```python
task_id = manager.submit(
    Rotate(angle=90.0),
    room="room_123",
    job_room="@global",  # room where the job is registered
)
```

### Provider Registration

Providers handle server-dispatched read requests (see [Providers](#providers)):

```python
from zndraw_joblib import Provider

class FilesystemRead(Provider):
    category: ClassVar[str] = "filesystem"
    path: str = "/"

    def read(self, handler):
        return handler.ls(self.path, detail=True)

manager.register_provider(
    FilesystemRead,
    name="local",
    handler=fsspec.filesystem("file"),
    room="@global",
)

# Access handlers during job execution
print(manager.handlers)  # {"@global:filesystem:local": <LocalFileSystem>}
```

### Lifecycle Management

```python
# Context manager (recommended)
with JobManager(api, execute=execute) as manager:
    # ... register jobs/providers ...
    manager.wait()  # blocks until disconnect

# Manual lifecycle
manager = JobManager(api, execute=execute)
# ... register jobs/providers ...
manager.disconnect()  # idempotent, safe to call multiple times
```

`disconnect()` is idempotent and handles:
1. Signaling background threads to stop
2. Joining all threads (waits for in-flight tasks to finish)
3. Emitting `LeaveJobRoom`/`LeaveProviderRoom` events
4. Calling `DELETE /workers/{id}` for server-side cleanup

Signal handlers (SIGINT/SIGTERM) call `disconnect()` automatically.

## REST Endpoints

All endpoints prefixed with `/v1/joblib`.

### Workers

```
POST   /workers                         # Create worker (201)
GET    /workers                         # List workers (paginated)
PATCH  /workers/{worker_id}             # Heartbeat
DELETE /workers/{worker_id}             # Delete + cascade cleanup (204)
```

### Jobs

```
PUT    /rooms/{room_id}/jobs            # Register job (idempotent, 201/200)
GET    /rooms/{room_id}/jobs            # List jobs (room + @global, paginated)
GET    /rooms/{room_id}/jobs/{job_name} # Job details
GET    /rooms/{room_id}/jobs/{job_name}/tasks  # Tasks for job (paginated)
```

### Tasks

```
POST   /rooms/{room_id}/tasks/{job_name}  # Submit task (202 Accepted)
POST   /tasks/claim                        # Claim oldest pending (FIFO)
GET    /tasks/{task_id}                    # Status (supports Prefer: wait=N)
PATCH  /tasks/{task_id}                    # Update status
GET    /rooms/{room_id}/tasks              # List room tasks (paginated)
```

### Task Lifecycle

```
PENDING -> CLAIMED -> RUNNING -> COMPLETED
                              -> FAILED
                   -> CANCELLED
         -> CANCELLED
```

Claiming uses optimistic locking with exponential backoff for concurrent safety.

Long-polling: `GET /tasks/{id}` with `Prefer: wait=N` header (max `long_poll_max_wait_seconds`). Returns immediately on terminal states.

## Providers

Providers are a generic abstraction for connected Python clients to **serve data on demand**. While jobs are user-initiated computation (workers pull tasks), providers handle **server-dispatched read requests** with result caching.

| | **Jobs** | **Providers** |
|---|---|---|
| **Purpose** | User-initiated computation | Remote resource access |
| **Dispatch** | Workers pull/claim (FIFO) | Server pushes to specific provider |
| **Results** | Side effects (modify room state) | Data returned to caller (cached) |
| **HTTP** | POST (creates task) | GET (reads resource) -> 200 or 202 |

### Provider Endpoints

```
PUT    /rooms/{room_id}/providers                        # Register (201/200)
GET    /rooms/{room_id}/providers                        # List (paginated)
GET    /rooms/{room_id}/providers/{name}/info             # Schema + metadata
GET    /rooms/{room_id}/providers/{name}?params           # Read (200 cached / 202 dispatched)
POST   /providers/{provider_id}/results                   # Upload result (204)
DELETE /providers/{provider_id}                            # Unregister (204)
```

### Read Request Flow

```
1. Frontend: GET /rooms/room-42/providers/@global:filesystem:local?path=/data
2. Server:   check cache -> HIT: return 200
                         -> MISS: acquire inflight, emit ProviderRequest -> return 202
3. Client:   receives ProviderRequest via Socket.IO
             calls provider.read(handler)
             POST /providers/{id}/results
4. Server:   store in ResultBackend, emit ProviderResultReady
5. Frontend: receives ProviderResultReady, re-fetches -> 200
```

### Result Backend

Provider reads require a `ResultBackend` for caching and inflight coalescing. The host app **must** override `get_result_backend`:

```python
from zndraw_joblib.dependencies import get_result_backend

class RedisResultBackend:
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

app.dependency_overrides[get_result_backend] = lambda: RedisResultBackend(redis)
```

## Internal TaskIQ Workers

For server-side jobs that should execute without an external Python client, use the `@internal` room with taskiq:

```python
from zndraw_joblib import register_internal_jobs

# In your FastAPI app lifespan:
await register_internal_jobs(
    app=app,
    broker=redis_broker,
    extensions=[MyServerSideJob],
    executor=my_executor,
    session_factory=my_session_maker,
)
```

This registers extensions as taskiq tasks, creates `@internal:category:name` job rows in the database, and stores the `InternalRegistry` on `app.state.internal_registry`.

For external taskiq worker processes (no FastAPI app):

```python
from zndraw_joblib import register_internal_tasks

registry = register_internal_tasks(
    broker=redis_broker,
    extensions=[MyServerSideJob],
    executor=my_executor,
)
```

Internal tasks that exceed `internal_task_timeout_seconds` are automatically failed by the sweeper.

## Socket.IO Real-Time Events

The package emits real-time events via [zndraw-socketio](https://github.com/zincware/zndraw-socketio). The host app provides its `AsyncServerWrapper` through dependency injection:

```python
from zndraw_socketio import wrap
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
| `ProvidersInvalidate` | *(none)* | `room:{room_id}` | Provider registered/deleted, worker disconnected |
| `ProviderRequest` | `request_id`, `provider_name`, `params` | `providers:{full_name}` | Server dispatches read to provider |
| `ProviderResultReady` | `provider_name`, `request_hash` | `room:{room_id}` | Provider result cached |
| `JoinJobRoom` | `job_name`, `worker_id` | *(client -> server)* | Worker joins notification room |
| `LeaveJobRoom` | `job_name`, `worker_id` | *(client -> server)* | Worker leaves notification room |
| `JoinProviderRoom` | `provider_name`, `worker_id` | *(client -> server)* | Client joins provider dispatch room |
| `LeaveProviderRoom` | `provider_name`, `worker_id` | *(client -> server)* | Client leaves provider dispatch room |

### Emission Deduplication

Internally, emissions are `Emission(NamedTuple)` pairs of `(event, room)`. Functions that modify state return `set[Emission]`, and callers emit **after commit**. Frozen models ensure duplicate events are deduplicated automatically.

### Worker Notification Pattern

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

### Server-Side Disconnect Cleanup

When a worker's Socket.IO connection drops, the host app can immediately clean up:

```python
from zndraw_joblib import cleanup_worker, emit

@tsio.on("disconnect")
async def on_disconnect(sid: str, reason: str):
    session = await tsio.get_session(sid)
    worker_id = session.get("worker_id")
    if worker_id:
        async with get_session() as db:
            worker = await db.get(Worker, UUID(worker_id))
            if worker:
                emissions = await cleanup_worker(db, worker)
                await db.commit()
                await emit(tsio, emissions)
```

| Disconnect Scenario | Handler |
|---------------------|---------|
| Network drop / process kill | Server-side SIO disconnect (immediate) |
| Graceful shutdown (`with manager:`) | Client `disconnect()` emits leave events + `DELETE /workers` |
| REST-only workers (no SIO) | Background sweeper heartbeat timeout |

## Background Sweeper

Host app starts explicitly:

```python
from zndraw_joblib import run_sweeper

asyncio.create_task(
    run_sweeper(get_session=my_session_factory, settings=settings, tsio=tsio)
)
```

The sweeper runs periodically (`sweeper_interval_seconds`) and:
1. Finds workers with stale `last_heartbeat` (beyond `worker_timeout_seconds`)
2. Marks their `running`/`claimed` tasks as `FAILED`
3. Removes orphan jobs (no workers, no pending tasks, not `@internal`)
4. Cleans up `@internal` tasks stuck beyond `internal_task_timeout_seconds`
5. Emits `TaskStatusEvent` and `JobsInvalidate` events after each cleanup cycle

## Error Handling (RFC 9457)

All errors use [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457) format:

| Exception | Status | Description |
|-----------|--------|-------------|
| `JobNotFound` | 404 | Job does not exist |
| `SchemaConflict` | 409 | Job schema differs from existing registration |
| `InvalidCategory` | 400 | Category not in allowed list |
| `WorkerNotFound` | 404 | Worker does not exist |
| `TaskNotFound` | 404 | Task does not exist |
| `InvalidTaskTransition` | 409 | Invalid status transition |
| `InvalidRoomId` | 400 | Room ID contains `@` or `:` |
| `Forbidden` | 403 | Admin privileges required |
| `InternalJobNotConfigured` | 503 | Internal job has no executor |
| `ProviderNotFound` | 404 | Provider does not exist |

## ORM Models

Models use SQLAlchemy 2.0 ORM inheriting from `zndraw_auth.Base`:

- **Job** - `(room_id, category, name)` unique, soft-deleted via `deleted` flag
- **Worker** - Tracks `last_heartbeat`, linked to user via `user_id`
- **Task** - Status state machine, linked to job and claiming worker
- **WorkerJobLink** - M:N bridge between Worker and Job
- **ProviderRecord** - `(room_id, category, name)` unique, linked to worker
