# Socket.IO Events Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real-time Socket.IO event emissions to zndraw-joblib so frontends and workers get push notifications for job/task state changes.

**Architecture:** Three frozen Pydantic event models (`JobsInvalidate`, `TaskAvailable`, `TaskStatus`) in a new `events.py` module. A dependency-injected `tsio` wrapper (defaulting to `None`) lets the host app provide Socket.IO. Sweeper functions return `set[Emission]` instead of emitting directly, keeping them pure/testable. Router endpoints emit inline after DB commits.

**Tech Stack:** Pydantic v2 (frozen models), FastAPI dependency injection, zndraw-socketio wrapper (optional dep), NamedTuple for hashable emission pairs.

---

### Task 1: Create events module

**Files:**
- Create: `src/zndraw_joblib/events.py`
- Test: `tests/test_events.py`

**Step 1: Write the failing test**

```python
# tests/test_events.py
"""Tests for Socket.IO event models and Emission dedup."""

from datetime import datetime, timezone

from zndraw_joblib.events import (
    JobsInvalidate,
    TaskAvailable,
    TaskStatusEvent,
    Emission,
)


def test_jobs_invalidate_frozen():
    """Frozen model is hashable and equal to itself."""
    a = JobsInvalidate()
    b = JobsInvalidate()
    assert a == b
    assert hash(a) == hash(b)


def test_task_available_frozen():
    ev = TaskAvailable(job_name="@global:modifiers:Rotate", room_id="room1", task_id="abc")
    assert ev.job_name == "@global:modifiers:Rotate"
    # Frozen: assignment should raise
    try:
        ev.job_name = "x"
        assert False, "Should have raised"
    except Exception:
        pass


def test_task_status_event_frozen():
    now = datetime.now(timezone.utc)
    ev = TaskStatusEvent(
        id="abc",
        name="@global:modifiers:Rotate",
        room_id="room1",
        status="pending",
        created_at=now,
    )
    assert ev.status == "pending"


def test_emission_dedup_jobs_invalidate():
    """Duplicate JobsInvalidate for same room should dedup in a set."""
    emissions = {
        Emission(JobsInvalidate(), "room:@global"),
        Emission(JobsInvalidate(), "room:@global"),
        Emission(JobsInvalidate(), "room:test"),
    }
    assert len(emissions) == 2


def test_emission_dedup_task_status():
    """Distinct TaskStatusEvent (different id) should NOT dedup."""
    now = datetime.now(timezone.utc)
    emissions = {
        Emission(
            TaskStatusEvent(id="a", name="j", room_id="r", status="failed", created_at=now),
            "room:r",
        ),
        Emission(
            TaskStatusEvent(id="b", name="j", room_id="r", status="failed", created_at=now),
            "room:r",
        ),
    }
    assert len(emissions) == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zndraw_joblib.events'`

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/events.py
"""Socket.IO event models for real-time notifications.

All models are frozen for hashability, enabling set-based deduplication
of emissions via the Emission NamedTuple.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict


class JobsInvalidate(BaseModel):
    """Frontend should refetch the job list.

    Emitted to room:{room_id} when jobs change (registered, deleted,
    worker connected/disconnected).
    """

    model_config = ConfigDict(frozen=True)


class TaskAvailable(BaseModel):
    """A new task is available for claiming.

    Emitted to jobs:{full_name} when a task is submitted for a
    non-@internal job.
    """

    model_config = ConfigDict(frozen=True)

    job_name: str
    room_id: str
    task_id: str


class TaskStatusEvent(BaseModel):
    """A task's status changed.

    Emitted to room:{room_id} on any task status transition.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    room_id: str
    status: Literal["pending", "claimed", "running", "completed", "failed", "cancelled"] # TODO: import from TaskStatus so they are shared!
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_position: int | None = None
    worker_id: str | None = None
    error: str | None = None


class Emission(NamedTuple):
    """Hashable (event, room) pair for set-based deduplication."""

    event: BaseModel
    room: str
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_events.py -v`
Expected: all 5 PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/events.py tests/test_events.py
git commit -m "feat: add frozen Socket.IO event models and Emission NamedTuple"
```

---

### Task 2: Add tsio dependency

**Files:**
- Modify: `src/zndraw_joblib/dependencies.py` (add `get_tsio` + `TsioDep`)
- Test: `tests/test_dependencies.py` (add test for default `get_tsio`)

**Step 1: Write the failing test**

Append to `tests/test_dependencies.py`:

```python
async def test_get_tsio_returns_none_by_default():
    """Default tsio dependency returns None (no socketio configured)."""
    from zndraw_joblib.dependencies import get_tsio

    result = await get_tsio()
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dependencies.py::test_get_tsio_returns_none_by_default -v`
Expected: FAIL — `ImportError: cannot import name 'get_tsio'`

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/dependencies.py` at the end:

```python
async def get_tsio():
    """Return the Socket.IO server wrapper for emitting events.

    Override this dependency in the host app to provide a real
    zndraw-socketio AsyncServerWrapper. Returns None by default,
    which disables all real-time event emissions.
    """
    return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dependencies.py::test_get_tsio_returns_none_by_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/dependencies.py tests/test_dependencies.py
git commit -m "feat: add get_tsio dependency stub for Socket.IO injection"
```

---

### Task 3: Update sweeper to return Emission sets

**Files:**
- Modify: `src/zndraw_joblib/sweeper.py`
- Modify: `tests/test_sweeper.py`

This is the largest task. The sweeper functions change signatures:
- `_soft_delete_orphan_job()` → returns `set[Emission]`
- `_cleanup_worker()` → returns `set[Emission]`
- `cleanup_stale_workers()` → returns `tuple[int, set[Emission]]`
- `cleanup_stuck_internal_tasks()` → returns `tuple[int, set[Emission]]`

**Step 1: Write the failing tests**

Append to `tests/test_sweeper.py`:

```python
from zndraw_joblib.events import Emission, JobsInvalidate, TaskStatusEvent


@pytest.mark.asyncio
async def test_cleanup_worker_returns_task_status_emissions(
    async_session_factory, test_user_id
):
    """_cleanup_worker should return TaskStatusEvent emissions for failed tasks."""
    worker_id = uuid.uuid4()

    async with async_session_factory() as session:
        worker = Worker(
            id=worker_id,
            user_id=test_user_id,
            last_heartbeat=datetime.now(timezone.utc),
        )
        session.add(worker)
        job = Job(room_id="@global", category="modifiers", name="EmitTest", schema_={})
        session.add(job)
        await session.flush()
        link = WorkerJobLink(worker_id=worker_id, job_id=job.id)
        session.add(link)
        task = Task(
            job_id=job.id,
            room_id="room1",
            status=TaskStatus.RUNNING,
            worker_id=worker_id,
        )
        session.add(task)
        await session.commit()
        task_id = task.id

    async with async_session_factory() as session:
        result = await session.execute(select(Worker).where(Worker.id == worker_id))
        worker = result.scalar_one()
        emissions = await _cleanup_worker(session, worker)
        await session.commit()

    # Should have TaskStatusEvent for the failed task + JobsInvalidate for orphan job
    task_events = [e for e in emissions if isinstance(e.event, TaskStatusEvent)]
    job_events = [e for e in emissions if isinstance(e.event, JobsInvalidate)]
    assert len(task_events) == 1
    assert task_events[0].event.id == str(task_id)
    assert task_events[0].event.status == "failed"
    assert task_events[0].room == "room:room1"
    assert len(job_events) == 1
    assert job_events[0].room == "room:@global"


@pytest.mark.asyncio
async def test_cleanup_stale_workers_returns_emissions(
    async_session_factory, test_user_id
):
    """cleanup_stale_workers should return count and emissions."""
    worker_id = uuid.uuid4()

    async with async_session_factory() as session:
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        worker = Worker(id=worker_id, user_id=test_user_id, last_heartbeat=stale_time)
        session.add(worker)
        await session.commit()

    async with async_session_factory() as session:
        count, emissions = await cleanup_stale_workers(session, timedelta(seconds=60))
        assert count == 1
        assert isinstance(emissions, set)


@pytest.mark.asyncio
async def test_cleanup_stuck_internal_returns_emissions(async_session_factory):
    """cleanup_stuck_internal_tasks should return count and emissions."""
    async with async_session_factory() as session:
        job = Job(room_id="@internal", category="modifiers", name="EmitInternal", schema_={})
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
        task_id = task.id

    async with async_session_factory() as session:
        count, emissions = await cleanup_stuck_internal_tasks(
            session, timeout=timedelta(hours=1)
        )
        assert count == 1
        task_events = [e for e in emissions if isinstance(e.event, TaskStatusEvent)]
        assert len(task_events) == 1
        assert task_events[0].event.id == str(task_id)
        assert task_events[0].event.status == "failed"
        assert task_events[0].room == "room:test-room"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sweeper.py::test_cleanup_worker_returns_task_status_emissions tests/test_sweeper.py::test_cleanup_stale_workers_returns_emissions tests/test_sweeper.py::test_cleanup_stuck_internal_returns_emissions -v`
Expected: FAIL — return value mismatch (functions currently return `None`/`int`)

**Step 3: Update sweeper implementation**

Update `src/zndraw_joblib/sweeper.py` to:

```python
# src/zndraw_joblib/sweeper.py
"""Background sweeper for cleaning up stale workers."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zndraw_joblib.events import Emission, JobsInvalidate, TaskStatusEvent
from zndraw_joblib.models import (
    Worker,
    Task,
    TaskStatus,
    Job,
    WorkerJobLink,
    TERMINAL_STATUSES,
)
from zndraw_joblib.settings import JobLibSettings

logger = logging.getLogger(__name__)


def _task_status_emission(task: Task, job_full_name: str) -> Emission:
    """Build a TaskStatusEvent emission from a task."""
    return Emission(
        TaskStatusEvent(
            id=str(task.id),
            name=job_full_name,
            room_id=task.room_id,
            status=task.status.value,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            worker_id=str(task.worker_id) if task.worker_id else None,
            error=task.error,
        ),
        f"room:{task.room_id}",
    )


async def _soft_delete_orphan_job(
    session: AsyncSession, job_id: uuid.UUID
) -> set[Emission]:
    """Soft-delete a job if it has no workers and no non-terminal tasks.

    Returns emissions to broadcast (JobsInvalidate if deleted).
    Note: Does NOT commit the transaction - caller must commit.
    """
    # Check if job has any remaining workers
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job_id).limit(1)
    )
    if result.scalar_one_or_none():
        return set()  # Job still has workers

    # Check if job has any non-terminal tasks
    result = await session.execute(
        select(Task)
        .where(
            Task.job_id == job_id,
            Task.status.not_in(TERMINAL_STATUSES),
        )
        .limit(1)
    )
    if result.scalar_one_or_none():
        return set()  # Job has pending/running tasks

    # Job is orphan - soft delete (tasks remain as historical records)
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job:
        job.deleted = True
        session.add(job)
        return {Emission(JobsInvalidate(), f"room:{job.room_id}")}
    return set()


async def _cleanup_worker(
    session: AsyncSession, worker: Worker
) -> set[Emission]:
    """Clean up a worker by failing tasks, removing links, and soft-deleting orphan jobs.

    Returns emissions to broadcast after commit.
    Note: Does NOT commit the transaction - caller must commit.
    """
    emissions: set[Emission] = set()
    now = datetime.now(timezone.utc)

    # Fail any claimed/running tasks owned by this worker
    result = await session.execute(
        select(Task)
        .where(
            Task.worker_id == worker.id,
            Task.status.in_({TaskStatus.CLAIMED, TaskStatus.RUNNING}),
        )
        .options(selectinload(Task.job))
    )
    worker_tasks = result.scalars().all()
    for task in worker_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Worker disconnected"
        session.add(task)
        job_full_name = task.job.full_name if task.job else ""
        emissions.add(_task_status_emission(task, job_full_name))

    # Get job IDs this worker is linked to before deleting links
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)
    )
    links = result.scalars().all()
    job_ids = [link.job_id for link in links]

    # Delete all links
    for link in links:
        await session.delete(link)

    # Delete worker
    await session.delete(worker)
    await session.flush()

    # Clean up orphan jobs (no workers and no non-terminal tasks)
    for job_id in job_ids:
        emissions |= await _soft_delete_orphan_job(session, job_id)

    return emissions


async def cleanup_stale_workers(
    session: AsyncSession, timeout: timedelta
) -> tuple[int, set[Emission]]:
    """Find and clean up workers with stale heartbeats.

    Args:
        session: Async database session
        timeout: How long since last heartbeat before a worker is considered stale

    Returns:
        Tuple of (count of workers cleaned up, set of emissions to broadcast)
    """
    cutoff = datetime.now(timezone.utc) - timeout

    # Find all stale workers
    result = await session.execute(select(Worker).where(Worker.last_heartbeat < cutoff))
    stale_workers = result.scalars().all()

    count = 0
    all_emissions: set[Emission] = set()
    for worker in stale_workers:
        logger.info("Cleaning up stale worker: %s", worker.id)
        emissions = await _cleanup_worker(session, worker)
        all_emissions |= emissions
        count += 1

    if count > 0:
        await session.commit()

    return count, all_emissions


async def cleanup_stuck_internal_tasks(
    session: AsyncSession, timeout: timedelta
) -> tuple[int, set[Emission]]:
    """Find and fail @internal tasks stuck in RUNNING beyond timeout.

    Args:
        session: Async database session
        timeout: How long a RUNNING internal task can run before being considered stuck

    Returns:
        Tuple of (count of tasks failed, set of emissions to broadcast)
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
        .options(selectinload(Task.job))
    )
    stuck_tasks = result.scalars().all()

    count = 0
    emissions: set[Emission] = set()
    for task in stuck_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Internal worker timeout"
        session.add(task)
        job_full_name = task.job.full_name if task.job else ""
        emissions.add(_task_status_emission(task, job_full_name))
        count += 1

    if count > 0:
        await session.commit()
        logger.info("Failed %d stuck internal task(s)", count)

    return count, emissions


async def run_sweeper(
    get_session: Callable[[], AsyncGenerator[AsyncSession, None]],
    settings: JobLibSettings,
    tsio: Any = None,
) -> None:
    """Background task that runs cleanup periodically.

    Args:
        get_session: Session factory generator
        settings: JobLib settings
        tsio: Optional zndraw-socketio AsyncServerWrapper for emitting events
    """
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
                count, emissions = await cleanup_stale_workers(session, timeout)
                if count > 0:
                    logger.info("Cleaned up %s stale worker(s)", count)
                if tsio and emissions:
                    for emission in emissions:
                        await tsio.emit(emission.event, room=emission.room)

            async for session in get_session():
                count, emissions = await cleanup_stuck_internal_tasks(
                    session, internal_timeout
                )
                if count > 0:
                    logger.info("Failed %s stuck internal task(s)", count)
                if tsio and emissions:
                    for emission in emissions:
                        await tsio.emit(emission.event, room=emission.room)
        except Exception as e:
            logger.exception("Error in sweeper: %s", e)
```

Note: add `from sqlalchemy.orm import selectinload` to the imports.

**Step 4: Fix existing tests**

The existing sweeper tests call `cleanup_stale_workers()` and expect just `int`. Update them to unpack:

In `tests/test_sweeper.py`, change every `count = await cleanup_stale_workers(session, ...)` to `count, _ = await cleanup_stale_workers(session, ...)`. Similarly change `count = await cleanup_stuck_internal_tasks(session, ...)` to `count, _ = await cleanup_stuck_internal_tasks(session, ...)`.

Affected tests:
- `test_cleanup_stale_workers_finds_stale` (line 34)
- `test_cleanup_stale_workers_ignores_alive` (line 63)
- `test_cleanup_fails_running_tasks` (line 125)
- `test_cleanup_soft_deletes_orphan_jobs` (line 183)
- `test_cleanup_keeps_job_with_pending_tasks` (line 231)
- `test_cleanup_multiple_stale_workers` (line 265)
- `test_cleanup_job_keeps_other_workers` (line 344)
- `test_cleanup_stuck_internal_tasks` (line 381)
- `test_cleanup_stuck_internal_tasks_skips_recent` (line 410)
- `test_cleanup_stuck_skips_external_tasks` (line 432)

**Step 5: Run all sweeper tests**

Run: `uv run pytest tests/test_sweeper.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/zndraw_joblib/sweeper.py tests/test_sweeper.py
git commit -m "feat: sweeper returns Emission sets for post-commit broadcasting"
```

---

### Task 4: Add TsioDep to router and emit events

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_events.py` (new file)

This task wires the `get_tsio` dependency into router endpoints and adds emit calls.

**Step 1: Write the failing tests**

```python
# tests/test_router_events.py
"""Tests that router endpoints emit Socket.IO events."""

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zndraw_joblib.router import router
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.dependencies import get_tsio


@pytest.fixture
def mock_tsio():
    """Create a mock tsio wrapper that records emit calls."""
    tsio = AsyncMock()
    tsio.emit = AsyncMock()
    return tsio


@pytest.fixture
def app_with_tsio(
    db_session,
    locked_db_session,
    test_session_factory,
    mock_current_user,
    mock_tsio,
):
    """App with tsio dependency overridden."""
    from zndraw_auth import current_active_user, current_superuser, get_async_session
    from zndraw_joblib.dependencies import get_locked_async_session, get_session_factory

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_async_session] = db_session
    app.dependency_overrides[get_locked_async_session] = locked_db_session
    app.dependency_overrides[get_session_factory] = test_session_factory
    app.dependency_overrides[current_active_user] = mock_current_user
    app.dependency_overrides[current_superuser] = mock_current_user
    app.dependency_overrides[get_tsio] = lambda: mock_tsio

    return app


@pytest.fixture
def client_with_tsio(app_with_tsio):
    return TestClient(app_with_tsio)


def test_register_job_emits_jobs_invalidate(client_with_tsio, mock_tsio):
    """PUT /rooms/{room_id}/jobs should emit JobsInvalidate."""
    resp = client_with_tsio.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code in (200, 201)

    # Check that emit was called with a JobsInvalidate event
    from zndraw_joblib.events import JobsInvalidate

    calls = mock_tsio.emit.call_args_list
    invalidate_calls = [c for c in calls if isinstance(c[0][0], JobsInvalidate)]
    assert len(invalidate_calls) >= 1
    assert invalidate_calls[0].kwargs["room"] == "room:@global"


def test_submit_task_emits_task_available(client_with_tsio, mock_tsio):
    """POST submit should emit TaskAvailable and TaskStatusEvent."""
    # Register a job first
    client_with_tsio.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    mock_tsio.emit.reset_mock()

    # Submit a task
    resp = client_with_tsio.post(
        "/v1/joblib/rooms/test-room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    assert resp.status_code == 202

    from zndraw_joblib.events import TaskAvailable, TaskStatusEvent

    calls = mock_tsio.emit.call_args_list
    available_calls = [c for c in calls if isinstance(c[0][0], TaskAvailable)]
    status_calls = [c for c in calls if isinstance(c[0][0], TaskStatusEvent)]

    assert len(available_calls) == 1
    assert available_calls[0].kwargs["room"] == "jobs:@global:modifiers:Rotate"
    assert len(status_calls) == 1
    assert status_calls[0].kwargs["room"] == "room:test-room"


def test_submit_internal_task_no_task_available(client_with_tsio, mock_tsio):
    """@internal task submission should NOT emit TaskAvailable."""
    # Register an internal job
    client_with_tsio.put(
        "/v1/joblib/rooms/@internal/jobs",
        json={"category": "modifiers", "name": "InternalOp", "schema": {}},
    )

    # Set up internal registry mock
    from unittest.mock import MagicMock
    from zndraw_joblib.dependencies import get_internal_registry
    from zndraw_joblib.registry import InternalRegistry

    mock_task = AsyncMock()
    registry = InternalRegistry(tasks={"@internal:modifiers:InternalOp": mock_task})
    client_with_tsio.app.dependency_overrides[get_internal_registry] = lambda: registry

    mock_tsio.emit.reset_mock()

    resp = client_with_tsio.post(
        "/v1/joblib/rooms/test-room/tasks/@internal:modifiers:InternalOp",
        json={"payload": {}},
    )
    assert resp.status_code == 202

    from zndraw_joblib.events import TaskAvailable

    calls = mock_tsio.emit.call_args_list
    available_calls = [c for c in calls if isinstance(c[0][0], TaskAvailable)]
    assert len(available_calls) == 0


def test_claim_task_emits_task_status(client_with_tsio, mock_tsio):
    """POST /tasks/claim should emit TaskStatusEvent with status=claimed."""
    # Register job + get worker_id
    resp = client_with_tsio.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    worker_id = resp.json()["worker_id"]

    # Submit task
    client_with_tsio.post(
        "/v1/joblib/rooms/test-room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    mock_tsio.emit.reset_mock()

    # Claim task
    resp = client_with_tsio.post(
        "/v1/joblib/tasks/claim",
        json={"worker_id": worker_id},
    )
    assert resp.status_code == 200
    assert resp.json()["task"] is not None

    from zndraw_joblib.events import TaskStatusEvent

    calls = mock_tsio.emit.call_args_list
    status_calls = [c for c in calls if isinstance(c[0][0], TaskStatusEvent)]
    assert len(status_calls) == 1
    assert status_calls[0].args[0].status == "claimed"
    assert status_calls[0].kwargs["room"] == "room:test-room"


def test_update_task_emits_task_status(client_with_tsio, mock_tsio):
    """PATCH /tasks/{id} should emit TaskStatusEvent."""
    resp = client_with_tsio.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    worker_id = resp.json()["worker_id"]

    resp = client_with_tsio.post(
        "/v1/joblib/rooms/test-room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = resp.json()["id"]

    # Claim
    client_with_tsio.post(
        "/v1/joblib/tasks/claim",
        json={"worker_id": worker_id},
    )
    mock_tsio.emit.reset_mock()

    # Update to RUNNING
    resp = client_with_tsio.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert resp.status_code == 200

    from zndraw_joblib.events import TaskStatusEvent

    calls = mock_tsio.emit.call_args_list
    status_calls = [c for c in calls if isinstance(c[0][0], TaskStatusEvent)]
    assert len(status_calls) == 1
    assert status_calls[0].args[0].status == "running"


def test_delete_worker_emits_events(client_with_tsio, mock_tsio):
    """DELETE /workers/{id} should emit JobsInvalidate + TaskStatusEvent for failed tasks."""
    resp = client_with_tsio.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    worker_id = resp.json()["worker_id"]

    # Submit and claim a task
    client_with_tsio.post(
        "/v1/joblib/rooms/test-room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    client_with_tsio.post(
        "/v1/joblib/tasks/claim",
        json={"worker_id": worker_id},
    )
    mock_tsio.emit.reset_mock()

    # Delete worker
    resp = client_with_tsio.delete(f"/v1/joblib/workers/{worker_id}")
    assert resp.status_code == 204

    from zndraw_joblib.events import JobsInvalidate, TaskStatusEvent

    calls = mock_tsio.emit.call_args_list
    invalidate_calls = [c for c in calls if isinstance(c[0][0], JobsInvalidate)]
    status_calls = [c for c in calls if isinstance(c[0][0], TaskStatusEvent)]
    assert len(invalidate_calls) >= 1
    assert len(status_calls) >= 1
    assert status_calls[0].args[0].status == "failed"


def test_no_tsio_does_not_break(client):
    """Endpoints work fine without tsio (default None)."""
    resp = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "NoTsio", "schema": {}},
    )
    assert resp.status_code == 201
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_router_events.py -v`
Expected: FAIL — `get_tsio` not used by router, no emit calls

**Step 3: Update router.py**

Changes to `src/zndraw_joblib/router.py`:

1. Import `get_tsio` and event models:
```python
from zndraw_joblib.dependencies import (
    get_settings,
    get_locked_async_session,
    get_session_factory,
    get_internal_registry,
    get_tsio,
)
from zndraw_joblib.events import (
    Emission,
    JobsInvalidate,
    TaskAvailable,
    TaskStatusEvent,
)
```

2. Add type alias:
```python
TsioDep = Annotated[object | None, Depends(get_tsio)]
```

3. Add helper to build TaskStatusEvent from a Task + session:
```python
async def _task_status_emission(
    session: AsyncSession, task: Task
) -> Emission:
    """Build a TaskStatusEvent emission from a task."""
    result = await session.execute(select(Job).where(Job.id == task.job_id))
    job = result.scalar_one_or_none()
    return Emission(
        TaskStatusEvent(
            id=str(task.id),
            name=job.full_name if job else "",
            room_id=task.room_id,
            status=task.status.value,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            queue_position=await _queue_position(session, task),
            worker_id=str(task.worker_id) if task.worker_id else None,
            error=task.error,
        ),
        f"room:{task.room_id}",
    )


async def _emit(tsio: object | None, emissions: set[Emission]) -> None:
    """Emit a set of events if tsio is available."""
    if not tsio:
        return
    for emission in emissions:
        await tsio.emit(emission.event, room=emission.room)
```

4. Add `tsio: TsioDep` parameter to these endpoints and emit after commit:
   - `register_job` — emit `JobsInvalidate` to `room:{room_id}`
   - `submit_task` — emit `TaskAvailable` to `jobs:{full_name}` (non-@internal) + `TaskStatusEvent` to `room:{room_id}`
   - `claim_task` — emit `TaskStatusEvent` to `room:{room_id}` if claimed
   - `update_task_status` — emit `TaskStatusEvent` to `room:{room_id}`
   - `delete_worker` — emit accumulated emissions from `_cleanup_worker`

See the detailed changes for each endpoint below.

**`register_job`** — add `tsio: TsioDep` param, after `await session.commit()` (line 311):
```python
    await _emit(tsio, {Emission(JobsInvalidate(), f"room:{room_id}")})
```

**`submit_task`** — add `tsio: TsioDep` param, after `await session.refresh(task)` and before internal dispatch (line 567):
```python
    emissions = {await _task_status_emission(session, task)}
    if job.room_id != "@internal":
        emissions.add(
            Emission(
                TaskAvailable(
                    job_name=job.full_name,
                    room_id=room_id,
                    task_id=str(task.id),
                ),
                f"jobs:{job.full_name}",
            )
        )
    await _emit(tsio, emissions)
```

**`claim_task`** — add `tsio: TsioDep` param, after claiming succeeds and task is fetched (line 668), before return:
```python
    if claimed_task_id:
        result = await session.execute(select(Task).where(Task.id == claimed_task_id))
        task = result.scalar_one()
        await _emit(tsio, {await _task_status_emission(session, task)})
```

**`update_task_status`** — add `tsio: TsioDep` param, after `await session.refresh(task)` (line 758):
```python
    emissions = {await _task_status_emission(session, task)}
    if request.status in TERMINAL_STATUSES:
        emissions |= await _soft_delete_orphan_job(session, task.job_id)
        await session.commit()
    await _emit(tsio, emissions)
```

**`delete_worker`** — add `tsio: TsioDep` param, after `await session.commit()` (line 838):
```python
    emissions = await _cleanup_worker(session, worker)
    await session.commit()
    await _emit(tsio, emissions)
```

**Step 4: Run all tests**

Run: `uv run pytest tests/test_router_events.py -v`
Expected: all PASS

Run: `uv run pytest -v`
Expected: all existing tests still PASS (tsio defaults to None, no emissions)

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_events.py
git commit -m "feat: emit Socket.IO events from router endpoints"
```

---

### Task 5: Export public API

**Files:**
- Modify: `src/zndraw_joblib/__init__.py`
- Modify: `tests/test_init.py`

**Step 1: Write the failing test**

Append to `tests/test_init.py`:

```python
def test_events_exported():
    """Event models and Emission should be importable from the package."""
    from zndraw_joblib import (
        JobsInvalidate,
        TaskAvailable,
        TaskStatusEvent,
        Emission,
    )

    assert JobsInvalidate is not None
    assert TaskAvailable is not None
    assert TaskStatusEvent is not None
    assert Emission is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init.py::test_events_exported -v`
Expected: FAIL — `ImportError`

**Step 3: Update `__init__.py`**

Add imports and `__all__` entries:

```python
from zndraw_joblib.events import (
    JobsInvalidate,
    TaskAvailable,
    TaskStatusEvent,
    Emission,
)
```

Add to `__all__`:
```python
    # Events
    "JobsInvalidate",
    "TaskAvailable",
    "TaskStatusEvent",
    "Emission",
```

Also add `get_tsio` to imports and `__all__`:
```python
from zndraw_joblib.dependencies import get_settings, get_tsio
```

```python
    # Dependencies
    "get_settings",
    "get_tsio",
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/__init__.py tests/test_init.py
git commit -m "feat: export event models and get_tsio in public API"
```

---

### Task 6: Lint, format, final check

**Step 1: Run lint**

Run: `uv run ruff check --fix`

**Step 2: Run format**

Run: `uv run ruff format`

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS

**Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style: fix lint and format"
```
