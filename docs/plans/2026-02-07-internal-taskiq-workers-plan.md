# Internal TaskIQ Workers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `@internal` job support so the server can execute tasks via taskiq + Redis, alongside existing external workers.

**Architecture:** Internal jobs use `room_id="@internal"` (like `@global`). Host app registers Extension classes + an executor callback at startup. On task submission, the router dispatches to taskiq instead of waiting for a worker claim. Status updates flow through HTTP like external workers.

**Tech Stack:** taskiq, taskiq-redis, FastAPI, SQLAlchemy 2.0, pydantic

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add taskiq and taskiq-redis**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv add taskiq taskiq-redis
```

**Step 2: Verify install**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run python -c "from taskiq_redis import ListQueueBroker; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add pyproject.toml uv.lock && git commit -m "build: add taskiq and taskiq-redis dependencies"
```

---

### Task 2: Add InternalJobNotConfigured exception

**Files:**
- Modify: `src/zndraw_joblib/exceptions.py:144-149` (append after Forbidden)
- Test: `tests/test_exceptions.py`

**Step 1: Write the failing test**

Add to `tests/test_exceptions.py`:

```python
def test_internal_job_not_configured_problem():
    from zndraw_joblib.exceptions import InternalJobNotConfigured

    problem = InternalJobNotConfigured.create(detail="Job '@internal:modifiers:Rotate' not configured")
    assert problem.status == 503
    assert problem.title == "Service Unavailable"
    assert problem.type == "/v1/problems/internal-job-not-configured"
    assert "@internal:modifiers:Rotate" in problem.detail
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_exceptions.py::test_internal_job_not_configured_problem -v
```

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Append to `src/zndraw_joblib/exceptions.py` after the `Forbidden` class:

```python
class InternalJobNotConfigured(ProblemType):
    """Internal job is registered but no executor is available."""

    title: ClassVar[str] = "Service Unavailable"
    status: ClassVar[int] = 503
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_exceptions.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/exceptions.py tests/test_exceptions.py && git commit -m "feat: add InternalJobNotConfigured exception (503)"
```

---

### Task 3: Add internal_task_timeout_seconds setting

**Files:**
- Modify: `src/zndraw_joblib/settings.py:8-20`
- Test: `tests/test_settings.py`

**Step 1: Write the failing test**

Add to `tests/test_settings.py`:

```python
def test_internal_task_timeout_default():
    from zndraw_joblib.settings import JobLibSettings

    settings = JobLibSettings()
    assert settings.internal_task_timeout_seconds == 3600
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_settings.py::test_internal_task_timeout_default -v
```

Expected: FAIL with `AttributeError`

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/settings.py` in `JobLibSettings` class, after `db_lock_timeout_seconds`:

```python
    # Internal taskiq worker settings
    internal_task_timeout_seconds: int = 3600  # 1 hour
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_settings.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/settings.py tests/test_settings.py && git commit -m "feat: add internal_task_timeout_seconds setting"
```

---

### Task 4: Create registry.py with InternalExecutor, InternalRegistry, and register_internal_tasks

**Files:**
- Create: `src/zndraw_joblib/registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write the failing tests**

Create `tests/test_registry.py`:

```python
"""Tests for internal job registry."""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from zndraw_joblib.client import Extension, Category
from zndraw_joblib.registry import InternalRegistry, register_internal_tasks


class FakeRotate(Extension):
    category: ClassVar[Category] = Category.MODIFIER
    angle: float = 0.0


class FakeScale(Extension):
    category: ClassVar[Category] = Category.MODIFIER
    factor: float = 1.0


def make_mock_broker():
    """Create a mock broker with register_task that returns a mock task handle."""
    broker = MagicMock()
    task_handles = {}

    def mock_register_task(fn, task_name, **kwargs):
        handle = MagicMock()
        handle.kiq = AsyncMock()
        task_handles[task_name] = handle
        return handle

    broker.register_task = mock_register_task
    broker._task_handles = task_handles  # for test inspection
    return broker


async def mock_executor(
    extension_cls: type[Extension],
    payload: dict[str, Any],
    room_id: str,
    task_id: str,
    base_url: str,
) -> None:
    pass


def test_register_internal_tasks_returns_registry():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [FakeRotate, FakeScale], executor=mock_executor)

    assert isinstance(registry, InternalRegistry)
    assert "@internal:modifiers:Rotate" in registry.tasks
    assert "@internal:modifiers:Scale" in registry.tasks
    assert len(registry.tasks) == 2


def test_register_internal_tasks_stores_extension_classes():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [FakeRotate], executor=mock_executor)

    assert registry.extensions["@internal:modifiers:Rotate"] is FakeRotate


def test_register_internal_tasks_registers_on_broker():
    broker = make_mock_broker()
    register_internal_tasks(broker, [FakeRotate], executor=mock_executor)

    assert "@internal:modifiers:Rotate" in broker._task_handles


def test_register_internal_tasks_empty_list():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [], executor=mock_executor)

    assert len(registry.tasks) == 0
    assert len(registry.extensions) == 0


def test_internal_registry_executor_stored():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [FakeRotate], executor=mock_executor)

    assert registry.executor is mock_executor
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/zndraw_joblib/registry.py`:

```python
"""Internal job registry for taskiq-based server-side execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from taskiq import AsyncBroker

from zndraw_joblib.client import Extension

logger = logging.getLogger(__name__)


class InternalExecutor(Protocol):
    """Protocol for the host-provided executor callback."""

    async def __call__(
        self,
        extension_cls: type[Extension],
        payload: dict[str, Any],
        room_id: str,
        task_id: str,
        base_url: str,
    ) -> None: ...


@dataclass
class InternalRegistry:
    """Holds taskiq task handles and extension class mappings."""

    tasks: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, type[Extension]] = field(default_factory=dict)
    executor: InternalExecutor | None = None


def register_internal_tasks(
    broker: AsyncBroker,
    extensions: list[type[Extension]],
    executor: InternalExecutor,
) -> InternalRegistry:
    """Register extension classes as taskiq tasks on the broker.

    For external taskiq worker processes that have no FastAPI app or DB.
    Also used internally by register_internal_jobs.

    Parameters
    ----------
    broker : AsyncBroker
        The taskiq broker to register tasks on.
    extensions : list[type[Extension]]
        Extension classes to register.
    executor : InternalExecutor
        Callback that executes the extension (provided by host app).

    Returns
    -------
    InternalRegistry
        Registry containing task handles and extension mappings.
    """
    registry = InternalRegistry(executor=executor)

    for ext_cls in extensions:
        category = ext_cls.category.value
        name = ext_cls.__name__
        full_name = f"@internal:{category}:{name}"

        def _make_task_fn(cls: type[Extension] = ext_cls, ex: InternalExecutor = executor):
            async def _execute(task_id: str, room_id: str, payload: dict, base_url: str) -> None:
                await ex(cls, payload, room_id, task_id, base_url)

            return _execute

        task_handle = broker.register_task(
            _make_task_fn(),
            task_name=full_name,
        )

        registry.tasks[full_name] = task_handle
        registry.extensions[full_name] = ext_cls
        logger.debug("Registered internal task: %s", full_name)

    logger.info("Registered %d internal task(s)", len(extensions))
    return registry


async def register_internal_jobs(
    app: Any,
    broker: AsyncBroker,
    extensions: list[type[Extension]],
    executor: InternalExecutor,
    session_factory: Any,
) -> None:
    """Register internal extensions for server-side execution.

    Does three things:
    1. Registers each Extension as a taskiq task on the broker
    2. Creates/reactivates @internal:category:name Job rows in the DB
    3. Stores the InternalRegistry on app.state.internal_registry

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    broker : AsyncBroker
        The taskiq broker to register tasks on.
    extensions : list[type[Extension]]
        Extension classes to register.
    executor : InternalExecutor
        Callback that executes the extension (provided by host app).
    session_factory : async_sessionmaker
        SQLAlchemy async session factory for DB operations.
    """
    from zndraw_joblib.models import Job

    registry = register_internal_tasks(broker, extensions, executor)

    async with session_factory() as session:
        for ext_cls in extensions:
            category = ext_cls.category.value
            name = ext_cls.__name__
            schema = ext_cls.model_json_schema()

            from sqlalchemy import select

            result = await session.execute(
                select(Job).where(
                    Job.room_id == "@internal",
                    Job.category == category,
                    Job.name == name,
                )
            )
            existing = result.scalar_one_or_none()

            if existing and existing.deleted:
                existing.deleted = False
                existing.schema_ = schema
            elif existing:
                existing.schema_ = schema
            else:
                job = Job(
                    room_id="@internal",
                    category=category,
                    name=name,
                    schema_=schema,
                )
                session.add(job)

        await session.commit()

    app.state.internal_registry = registry
    logger.info("Internal jobs registered in DB and attached to app.state")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_registry.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/registry.py tests/test_registry.py && git commit -m "feat: add InternalRegistry, register_internal_tasks, register_internal_jobs"
```

---

### Task 5: Test register_internal_jobs (DB + app.state integration)

**Files:**
- Modify: `tests/test_registry.py`

**Step 1: Write the failing test**

Add to `tests/test_registry.py`:

```python
from zndraw_joblib.registry import register_internal_jobs
from zndraw_joblib.models import Job
from sqlalchemy import select


async def test_register_internal_jobs_creates_db_rows(async_session_factory):
    """register_internal_jobs creates Job rows in the DB."""
    from unittest.mock import MagicMock

    broker = make_mock_broker()
    app = MagicMock()
    app.state = MagicMock()

    await register_internal_jobs(app, broker, [FakeRotate, FakeScale], executor=mock_executor, session_factory=async_session_factory)

    async with async_session_factory() as session:
        result = await session.execute(
            select(Job).where(Job.room_id == "@internal")
        )
        jobs = result.scalars().all()

    assert len(jobs) == 2
    names = {j.full_name for j in jobs}
    assert "@internal:modifiers:Rotate" in names
    assert "@internal:modifiers:Scale" in names
    assert all(not j.deleted for j in jobs)


async def test_register_internal_jobs_sets_app_state(async_session_factory):
    """register_internal_jobs stores registry on app.state."""
    from unittest.mock import MagicMock

    broker = make_mock_broker()
    app = MagicMock()
    app.state = MagicMock()

    await register_internal_jobs(app, broker, [FakeRotate], executor=mock_executor, session_factory=async_session_factory)

    assert hasattr(app.state, "internal_registry")
    registry = app.state.internal_registry
    assert isinstance(registry, InternalRegistry)
    assert "@internal:modifiers:Rotate" in registry.tasks


async def test_register_internal_jobs_reactivates_deleted(async_session_factory):
    """register_internal_jobs reactivates soft-deleted jobs."""
    from unittest.mock import MagicMock

    # First: create a deleted job manually
    async with async_session_factory() as session:
        job = Job(room_id="@internal", category="modifiers", name="Rotate", schema_={}, deleted=True)
        session.add(job)
        await session.commit()

    broker = make_mock_broker()
    app = MagicMock()
    app.state = MagicMock()

    await register_internal_jobs(app, broker, [FakeRotate], executor=mock_executor, session_factory=async_session_factory)

    async with async_session_factory() as session:
        result = await session.execute(
            select(Job).where(Job.room_id == "@internal", Job.name == "Rotate")
        )
        job = result.scalar_one()
        assert not job.deleted
```

Note: these tests use the `async_session_factory` fixture from `conftest.py`.

**Step 2: Run tests to verify they pass**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_registry.py -v
```

Expected: ALL PASS (implementation was already written in Task 4)

**Step 3: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add tests/test_registry.py && git commit -m "test: add integration tests for register_internal_jobs"
```

---

### Task 6: Add get_internal_registry dependency

**Files:**
- Modify: `src/zndraw_joblib/dependencies.py:1-76`
- Test: `tests/test_dependencies.py`

**Step 1: Write the failing test**

Add to `tests/test_dependencies.py`:

```python
def test_get_internal_registry_import():
    from zndraw_joblib.dependencies import get_internal_registry
    assert callable(get_internal_registry)
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py::test_get_internal_registry_import -v
```

Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/zndraw_joblib/dependencies.py` at the end:

```python
from fastapi import Request

from zndraw_joblib.registry import InternalRegistry


async def get_internal_registry(request: Request) -> InternalRegistry | None:
    """Return the internal registry from app.state, or None if not configured."""
    return getattr(request.app.state, "internal_registry", None)
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/dependencies.py tests/test_dependencies.py && git commit -m "feat: add get_internal_registry dependency"
```

---

### Task 7: Update _resolve_job to include @internal fallback

**Files:**
- Modify: `src/zndraw_joblib/router.py:92-113`
- Test: `tests/test_router_jobs.py`

**Step 1: Write the failing test**

Add to `tests/test_router_jobs.py`:

```python
def test_resolve_internal_job_from_room(client):
    """@internal jobs are accessible from any room."""
    # Register an @internal job (requires superuser)
    resp = client.put(
        "/v1/joblib/rooms/@internal/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code in (200, 201)

    # Should be accessible when listing jobs for any room
    resp = client.get("/v1/joblib/rooms/test-room/jobs")
    assert resp.status_code == 200
    names = [j["full_name"] for j in resp.json()["items"]]
    assert "@internal:modifiers:Rotate" in names
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_jobs.py::test_resolve_internal_job_from_room -v
```

Expected: FAIL — `@internal` contains `@` which is rejected by `validate_room_id`

**Step 3: Update validate_room_id to allow @internal**

In `src/zndraw_joblib/router.py`, change `validate_room_id`:

```python
def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global and @internal)."""
    if room_id in ("@global", "@internal"):
        return
    if "@" in room_id or ":" in room_id:
        raise InvalidRoomId.exception(
            detail=f"Room ID '{room_id}' contains invalid characters (@ or :)"
        )
```

**Step 4: Update _resolve_job fallback chain**

In `src/zndraw_joblib/router.py`, replace the `_resolve_job` function:

```python
async def _resolve_job(
    session: AsyncSession, job_name: str, room_id: str
) -> Job:
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")
    job_room_id, category, name = parts

    # For non-special rooms, the job must be in the room, @global, or @internal
    if room_id not in ("@global", "@internal") and job_room_id not in (
        "@global",
        "@internal",
        room_id,
    ):
        raise JobNotFound.exception(
            detail=f"Job '{job_name}' not accessible from room '{room_id}'"
        )

    result = await session.execute(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    )
    job = result.scalar_one_or_none()
    if not job or job.deleted:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")
    return job
```

**Step 5: Update list_jobs to include @internal**

In `src/zndraw_joblib/router.py`, update `list_jobs` base_filter:

```python
    base_filter = (
        Job.room_id == "@global" if room_id == "@global"
        else Job.room_id == "@internal" if room_id == "@internal"
        else (Job.room_id.in_(["@global", "@internal"])) | (Job.room_id == room_id)
    )
```

**Step 6: Update list_workers_for_room similarly**

In `src/zndraw_joblib/router.py`, update `list_workers_for_room` base_filter:

```python
    base_filter = (
        Job.room_id == "@global" if room_id == "@global"
        else Job.room_id == "@internal" if room_id == "@internal"
        else (Job.room_id.in_(["@global", "@internal"])) | (Job.room_id == room_id)
    )
```

**Step 7: Update register_job to require superuser for @internal**

In `src/zndraw_joblib/router.py`, update the admin check in `register_job`:

```python
    # Check admin for @global and @internal
    if room_id in ("@global", "@internal") and not user.is_superuser:
        raise Forbidden.exception(detail="Admin required for @global/@internal job registration")
```

**Step 8: Run all tests**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 9: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/router.py tests/test_router_jobs.py && git commit -m "feat: add @internal to _resolve_job fallback and job listing"
```

---

### Task 8: Add taskiq dispatch to submit_task endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py:510-544`
- Test: `tests/test_router_task_submit.py`

**Step 1: Write the failing test**

Add to `tests/test_router_task_submit.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from zndraw_joblib.registry import InternalRegistry


def test_submit_internal_task_dispatches_to_taskiq(app, client):
    """Submitting a task for an @internal job dispatches via taskiq."""
    # Register an @internal job
    resp = client.put(
        "/v1/joblib/rooms/@internal/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code in (200, 201)

    # Set up mock internal registry on app.state
    mock_task_handle = MagicMock()
    mock_task_handle.kiq = AsyncMock()
    registry = InternalRegistry(
        tasks={"@internal:modifiers:Rotate": mock_task_handle},
        extensions={},
    )
    app.state.internal_registry = registry

    # Submit a task
    resp = client.post(
        "/v1/joblib/rooms/test-room/tasks/@internal:modifiers:Rotate",
        json={"payload": {"angle": 90}},
    )
    assert resp.status_code == 202

    # Verify taskiq dispatch was called
    mock_task_handle.kiq.assert_called_once()
    call_kwargs = mock_task_handle.kiq.call_args.kwargs
    assert call_kwargs["room_id"] == "test-room"
    assert "task_id" in call_kwargs


def test_submit_internal_task_no_registry_returns_503(app, client):
    """Submitting to @internal job without registry returns 503."""
    # Register an @internal job
    resp = client.put(
        "/v1/joblib/rooms/@internal/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code in (200, 201)

    # Ensure no registry is set
    if hasattr(app.state, "internal_registry"):
        delattr(app.state, "internal_registry")

    resp = client.post(
        "/v1/joblib/rooms/test-room/tasks/@internal:modifiers:Rotate",
        json={"payload": {}},
    )
    assert resp.status_code == 503


def test_submit_external_task_unchanged(seeded_client):
    """External task submission still works as before (no taskiq dispatch)."""
    resp = seeded_client.post(
        "/v1/joblib/rooms/test-room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_submit.py::test_submit_internal_task_dispatches_to_taskiq tests/test_router_task_submit.py::test_submit_internal_task_no_registry_returns_503 -v
```

Expected: FAIL — submit_task doesn't check for internal dispatch

**Step 3: Write implementation**

Update `submit_task` in `src/zndraw_joblib/router.py`. Add imports at the top of the file:

```python
from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
```

And add `InternalJobNotConfigured` to the exception imports:

```python
from zndraw_joblib.exceptions import (
    InvalidCategory,
    InvalidRoomId,
    SchemaConflict,
    Forbidden,
    JobNotFound,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
    InternalJobNotConfigured,
)
```

Add dependency import:

```python
from zndraw_joblib.dependencies import get_settings, get_locked_async_session, get_session_factory, get_internal_registry
from zndraw_joblib.registry import InternalRegistry
```

Add type alias:

```python
InternalRegistryDep = Annotated[InternalRegistry | None, Depends(get_internal_registry)]
```

Replace `submit_task`:

```python
@router.post(
    "/rooms/{room_id}/tasks/{job_name:path}",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task(
    room_id: str,
    job_name: str,
    request: TaskSubmitRequest,
    response: Response,
    session: LockedSessionDep,
    user: CurrentUserDep,
    internal_registry: InternalRegistryDep,
):
    """Submit a task for processing."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    # Create task
    task = Task(
        job_id=job.id,
        room_id=room_id,
        created_by_id=user.id,
        payload=request.payload,
        status=TaskStatus.PENDING,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Dispatch to taskiq for @internal jobs
    if job.room_id == "@internal":
        if internal_registry is None or job.full_name not in internal_registry.tasks:
            raise InternalJobNotConfigured.exception(
                detail=f"Internal job '{job.full_name}' is registered but no executor is available"
            )
        await internal_registry.tasks[job.full_name].kiq(
            task_id=str(task.id),
            room_id=room_id,
            payload=request.payload,
            base_url="",  # Host executor knows its own base_url
        )

    # Set Location header
    response.headers["Location"] = f"/v1/joblib/tasks/{task.id}"
    response.headers["Retry-After"] = "1"

    return await _task_response(session, task)
```

**Step 4: Run all tests**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/router.py tests/test_router_task_submit.py && git commit -m "feat: dispatch @internal tasks to taskiq on submission"
```

---

### Task 9: Add stuck internal task cleanup to sweeper

**Files:**
- Modify: `src/zndraw_joblib/sweeper.py`
- Test: `tests/test_sweeper.py`

**Step 1: Write the failing test**

Add to `tests/test_sweeper.py`:

```python
from datetime import datetime, timedelta, timezone
from zndraw_joblib.models import Job, Task, TaskStatus
from zndraw_joblib.sweeper import cleanup_stuck_internal_tasks


async def test_cleanup_stuck_internal_tasks(async_session_factory):
    """Stuck @internal tasks in RUNNING are marked FAILED after timeout."""
    async with async_session_factory() as session:
        job = Job(room_id="@internal", category="modifiers", name="Rotate", schema_={})
        session.add(job)
        await session.flush()

        # Task started 2 hours ago (exceeds 1h timeout)
        task = Task(
            job_id=job.id,
            room_id="test-room",
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        session.add(task)
        await session.commit()
        task_id = task.id

    async with async_session_factory() as session:
        count = await cleanup_stuck_internal_tasks(
            session, timeout=timedelta(hours=1)
        )
        assert count == 1

    async with async_session_factory() as session:
        from sqlalchemy import select
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one()
        assert task.status == TaskStatus.FAILED
        assert task.error == "Internal worker timeout"
        assert task.completed_at is not None


async def test_cleanup_stuck_internal_tasks_skips_recent(async_session_factory):
    """Recently started @internal tasks are not cleaned up."""
    async with async_session_factory() as session:
        job = Job(room_id="@internal", category="modifiers", name="Scale", schema_={})
        session.add(job)
        await session.flush()

        # Task started 10 minutes ago (within 1h timeout)
        task = Task(
            job_id=job.id,
            room_id="test-room",
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        session.add(task)
        await session.commit()

    async with async_session_factory() as session:
        count = await cleanup_stuck_internal_tasks(
            session, timeout=timedelta(hours=1)
        )
        assert count == 0


async def test_cleanup_stuck_skips_external_tasks(async_session_factory):
    """External (@global) RUNNING tasks are NOT cleaned up by this function."""
    async with async_session_factory() as session:
        job = Job(room_id="@global", category="modifiers", name="Rotate", schema_={})
        session.add(job)
        await session.flush()

        task = Task(
            job_id=job.id,
            room_id="test-room",
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        session.add(task)
        await session.commit()

    async with async_session_factory() as session:
        count = await cleanup_stuck_internal_tasks(
            session, timeout=timedelta(hours=1)
        )
        assert count == 0
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_sweeper.py::test_cleanup_stuck_internal_tasks tests/test_sweeper.py::test_cleanup_stuck_internal_tasks_skips_recent tests/test_sweeper.py::test_cleanup_stuck_skips_external_tasks -v
```

Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/zndraw_joblib/sweeper.py`:

```python
async def cleanup_stuck_internal_tasks(
    session: AsyncSession, timeout: timedelta
) -> int:
    """Find and fail @internal tasks stuck in RUNNING beyond timeout.

    Args:
        session: Async database session
        timeout: How long a RUNNING internal task can run before being considered stuck

    Returns:
        Count of tasks failed
    """
    cutoff = datetime.now(timezone.utc) - timeout
    now = datetime.now(timezone.utc)

    result = await session.execute(
        select(Task)
        .join(Job)
        .where(
            Job.room_id == "@internal",
            Task.status == TaskStatus.RUNNING,
            Task.started_at < cutoff,
        )
    )
    stuck_tasks = result.scalars().all()

    count = 0
    for task in stuck_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Internal worker timeout"
        session.add(task)
        count += 1

    if count > 0:
        await session.commit()
        logger.info("Failed %d stuck internal task(s)", count)

    return count
```

Also add `Job` to the imports at the top of `sweeper.py` if not already present (it already imports `Job`).

**Step 4: Integrate into run_sweeper**

Update `run_sweeper` in `src/zndraw_joblib/sweeper.py` to also call `cleanup_stuck_internal_tasks`:

```python
async def run_sweeper(
    get_session: Callable[[], AsyncGenerator[AsyncSession, None]],
    settings: JobLibSettings,
) -> None:
    """Background task that runs cleanup periodically."""
    timeout = timedelta(seconds=settings.worker_timeout_seconds)
    internal_timeout = timedelta(seconds=settings.internal_task_timeout_seconds)
    interval = settings.sweeper_interval_seconds

    logger.info(
        "Starting sweeper with interval=%ss, worker_timeout=%ss, internal_task_timeout=%ss",
        interval,
        settings.worker_timeout_seconds,
        settings.internal_task_timeout_seconds,
    )

    while True:
        await asyncio.sleep(interval)
        try:
            async for session in get_session():
                count = await cleanup_stale_workers(session, timeout)
                if count > 0:
                    logger.info("Cleaned up %s stale worker(s)", count)

            async for session in get_session():
                count = await cleanup_stuck_internal_tasks(session, internal_timeout)
                if count > 0:
                    logger.info("Failed %s stuck internal task(s)", count)
        except Exception as e:
            logger.exception("Error in sweeper: %s", e)
```

**Step 5: Run all tests**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 6: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/sweeper.py tests/test_sweeper.py && git commit -m "feat: add stuck internal task cleanup to sweeper"
```

---

### Task 10: Update __init__.py exports

**Files:**
- Modify: `src/zndraw_joblib/__init__.py`
- Test: `tests/test_init.py`

**Step 1: Write the failing test**

Add to `tests/test_init.py`:

```python
def test_internal_registry_exports():
    from zndraw_joblib import (
        register_internal_jobs,
        register_internal_tasks,
        InternalExecutor,
        InternalRegistry,
        InternalJobNotConfigured,
    )
    assert callable(register_internal_jobs)
    assert callable(register_internal_tasks)
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_init.py::test_internal_registry_exports -v
```

Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Update `src/zndraw_joblib/__init__.py` — add imports and __all__ entries:

Add to imports:

```python
from zndraw_joblib.registry import (
    register_internal_jobs,
    register_internal_tasks,
    InternalExecutor,
    InternalRegistry,
)
from zndraw_joblib.exceptions import InternalJobNotConfigured
```

Add to `__all__`:

```python
    # Internal registry
    "register_internal_jobs",
    "register_internal_tasks",
    "InternalExecutor",
    "InternalRegistry",
    "InternalJobNotConfigured",
```

Also add `cleanup_stuck_internal_tasks` to the sweeper imports and __all__:

```python
from zndraw_joblib.sweeper import run_sweeper, cleanup_stale_workers, cleanup_stuck_internal_tasks
```

```python
    "cleanup_stuck_internal_tasks",
```

**Step 4: Run all tests**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add src/zndraw_joblib/__init__.py tests/test_init.py && git commit -m "feat: export internal registry public API"
```

---

### Task 11: Run full test suite and lint

**Step 1: Run all tests**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 2: Run linter**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run ruff check
```

**Step 3: Fix any lint issues**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run ruff check --fix && uv run ruff format
```

**Step 4: Run tests again after formatting**

```bash
cd /Users/fzills/tools/zndraw-joblib && uv run pytest -v
```

Expected: ALL PASS

**Step 5: Commit if any formatting changes**

```bash
cd /Users/fzills/tools/zndraw-joblib && git add -A && git commit -m "style: lint and format"
```
