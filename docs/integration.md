# Integration Guide

This guide walks through mounting `zndraw-joblib` into a host FastAPI application. Each section is self-contained -- implement only the parts you need. For the reasoning behind these design decisions, see the linked concept pages.

## Minimal Setup

A working integration requires four things: a database session, auth overrides, the error handler, and settings.

```python
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from zndraw_auth import current_active_user, current_superuser
from zndraw_auth.db import get_session_maker
from zndraw_joblib.router import router
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.settings import JobLibSettings

app = FastAPI()

# 1. Database -- override the single session maker dependency
engine = create_async_engine("sqlite+aiosqlite:///./app.db")
session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
app.dependency_overrides[get_session_maker] = lambda: session_maker

# 2. Auth -- override user dependencies (from zndraw_auth)
# app.dependency_overrides[current_active_user] = my_get_current_user
# app.dependency_overrides[current_superuser] = my_get_superuser

# 3. Register error handler + router
app.add_exception_handler(ProblemException, problem_exception_handler)
app.include_router(router)

# 4. Configure settings
app.state.joblib_settings = JobLibSettings()
```

The router reads `app.state.joblib_settings` via the `get_joblib_settings` dependency. All endpoints live under the `/v1/joblib` prefix.

See [Architecture: Dependency Injection](concepts/architecture.md#dependency-injection) for details on override tiers and why `get_session_maker` is the single point for all DB access.

## Adding Socket.IO

When `app.state.tsio` is set to an `AsyncServerWrapper` instance, the router emits real-time events on job registration, task status changes, and provider updates. When it is `None` (the default), all emissions are silently skipped.

```python
from zndraw_socketio import wrap
import socketio

sio = socketio.AsyncServer(async_mode="asgi")
tsio = wrap(sio)
app.state.tsio = tsio
```

The `get_tsio` dependency reads `app.state.tsio` and returns `None` if it is not set. No override is needed -- just assign the attribute.

See [Events](concepts/events.md) for the room topology and event model details.

## Adding the Sweeper

The sweeper is a background coroutine that periodically cleans up stale workers and stuck internal tasks. It expects an async generator that yields a database session.

```python
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from zndraw_joblib import run_sweeper

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_maker() as session:
        yield session

@asynccontextmanager
async def lifespan(app):
    settings = app.state.joblib_settings
    task = asyncio.create_task(
        run_sweeper(get_session=get_session, settings=settings, tsio=tsio)
    )
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
```

The `run_sweeper` function runs two cleanup passes per cycle:

1. `cleanup_stale_workers` -- removes workers whose heartbeat exceeds `WORKER_TIMEOUT_SECONDS`, fails their in-progress tasks, and soft-deletes orphan jobs.
2. `cleanup_stuck_internal_tasks` -- fails `@internal` tasks stuck in RUNNING or CLAIMED beyond `INTERNAL_TASK_TIMEOUT_SECONDS`.

See [Sweeper](concepts/sweeper.md) for cleanup logic details.

## SQLite Locking

SQLite only allows one writer at a time. For production with SQLite, wrap the session maker with an `asyncio.Lock` to serialize DB access:

```python
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

db_lock = asyncio.Lock()

@asynccontextmanager
async def locked_session():
    async with db_lock:
        async with session_maker() as session:
            yield session
```

Then override the session maker dependency to return a wrapper that uses this locked session maker.

For PostgreSQL or MySQL, no app-level lock is needed -- the database engine handles concurrent writes. The optimistic locking in task claiming (see [Jobs & Tasks: Optimistic Locking](concepts/jobs-and-tasks.md#optimistic-locking-task-claiming)) handles application-level race conditions for all backends.

## Room Writability Guard

The `verify_writable_room` dependency guards the `register_job` and `submit_task` endpoints. The default implementation only validates room_id format (no `@` or `:` characters, except for `@global` and `@internal`). Override it to add lock or permission checks:

```python
from fastapi import Path, HTTPException
from zndraw_auth import current_active_user
from zndraw_auth.db import SessionDep
from zndraw_joblib import verify_writable_room, validate_room_id

async def get_writable_room(
    session: SessionDep,
    current_user: ...,  # your CurrentUserDep
    room_id: str = Path(),
) -> str:
    validate_room_id(room_id)
    room = await get_room(session, room_id)
    if room.locked and not current_user.is_superuser:
        raise HTTPException(status_code=423, detail="Room is locked")
    return room_id

app.dependency_overrides[verify_writable_room] = get_writable_room
```

The dependency must accept `room_id` as a `Path()` parameter and return the validated room_id string.

## Adding Providers

Provider endpoints require a `ResultBackend` for caching and coordinating read results. The default `get_result_backend` dependency raises `NotImplementedError` -- host apps must override it.

```python
from zndraw_joblib import get_result_backend, ResultBackend

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

    async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
        # Subscribe first, then check cache (race-safe ordering)
        ...

    async def notify_key(self, key: str) -> None:
        # Publish notification via Redis pub/sub
        ...

backend = RedisResultBackend(redis)
app.dependency_overrides[get_result_backend] = lambda: backend
```

The `ResultBackend` is a `typing.Protocol` defined in `zndraw_joblib.dependencies`. Any class implementing the eight methods above satisfies it at runtime.

See [Providers: ResultBackend Protocol](concepts/providers.md#resultbackend-protocol) for the full protocol specification.

## Adding Internal Workers (TaskIQ)

Internal workers execute `@internal` jobs server-side via taskiq and a message broker (e.g., Redis). Use `register_internal_jobs` inside your FastAPI lifespan to wire up the broker, database rows, and app state in one call:

```python
from zndraw_joblib import register_internal_jobs

# In your FastAPI lifespan:
await register_internal_jobs(
    app=app,
    broker=redis_broker,
    extensions=[MyServerSideJob],
    executor=my_executor,
    session_factory=session_maker,
)
```

This does three things:

1. Registers each `Extension` subclass as a taskiq task on the broker.
2. Creates or reactivates `@internal:category:name` Job rows in the database.
3. Stores the `InternalRegistry` on `app.state.internal_registry`.

For external taskiq worker processes (no FastAPI app or database access):

```python
from zndraw_joblib import register_internal_tasks

registry = register_internal_tasks(
    broker=redis_broker,
    extensions=[MyServerSideJob],
    executor=my_executor,
)
```

This only registers tasks on the broker -- no database interaction occurs. Use this in standalone worker processes that consume from the same broker.

See [Jobs & Tasks: Internal Job Dispatch](concepts/jobs-and-tasks.md#internal-job-dispatch-internal) for the dispatch flow.

## Configuration Reference

All settings use the `ZNDRAW_JOBLIB_` environment variable prefix and are managed by `JobLibSettings` (pydantic-settings). Store the instance on `app.state.joblib_settings`.

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_CATEGORIES` | `["modifiers", "selections", "analysis"]` | Valid job categories |
| `WORKER_TIMEOUT_SECONDS` | `60` | Stale heartbeat threshold |
| `SWEEPER_INTERVAL_SECONDS` | `30` | Sweeper cycle interval |
| `LONG_POLL_MAX_WAIT_SECONDS` | `60` | Max long-poll wait for task status |
| `CLAIM_MAX_ATTEMPTS` | `10` | Retries for concurrent claim contention |
| `CLAIM_BASE_DELAY_SECONDS` | `0.01` | Exponential backoff base delay |
| `INTERNAL_TASK_TIMEOUT_SECONDS` | `3600` | Timeout for stuck `@internal` tasks |
| `ALLOWED_PROVIDER_CATEGORIES` | `None` (unrestricted) | Valid provider categories |
| `PROVIDER_RESULT_TTL_SECONDS` | `300` | Cached provider result lifetime |
| `PROVIDER_INFLIGHT_TTL_SECONDS` | `30` | Inflight lock lifetime |
| `PROVIDER_LONG_POLL_DEFAULT_SECONDS` | `5` | Default provider long-poll wait |
| `PROVIDER_LONG_POLL_MAX_SECONDS` | `30` | Max provider long-poll wait |
