# Worker & Task Listing Endpoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add REST endpoints to list workers by room, list all workers, list tasks by room/job, and include queue position in task responses.

**Architecture:** Extend existing FastAPI router with new GET endpoints. Workers are linked to jobs via `WorkerJobLink`, so listing workers for a room means finding workers linked to jobs in that room. Queue position is calculated by counting pending tasks created before the given task for the same job.

**Tech Stack:** FastAPI, SQLModel, Pydantic, pytest

---

## Task 1: Add WorkerSummary Schema

**Files:**
- Modify: `src/zndraw_joblib/schemas.py`
- Test: `tests/test_schemas.py` (if exists, otherwise skip test)

**Step 1: Add WorkerSummary schema**

Add to `src/zndraw_joblib/schemas.py`:

```python
class WorkerSummary(BaseModel):
    id: str
    last_heartbeat: datetime
    job_count: int
```

**Step 2: Commit**

```bash
git add src/zndraw_joblib/schemas.py
git commit -m "feat: add WorkerSummary schema for worker listing endpoints"
```

---

## Task 2: Add GET /workers Endpoint (List All Workers)

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Modify: `src/zndraw_joblib/schemas.py` (import)
- Test: `tests/test_router_workers.py`

**Step 1: Write the failing test**

Create/add to `tests/test_router_workers.py`:

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_list_workers_empty(client: AsyncClient):
    """List workers returns empty list when no workers exist."""
    response = await client.get("/v1/joblib/workers")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_workers_returns_all(client: AsyncClient, register_job):
    """List workers returns all workers with job counts."""
    # Register jobs from two different workers
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    await register_job("room1", "modifiers", "job2", worker_id="worker-a")
    await register_job("room2", "modifiers", "job3", worker_id="worker-b")

    response = await client.get("/v1/joblib/workers")
    assert response.status_code == 200
    workers = response.json()
    assert len(workers) == 2

    worker_map = {w["id"]: w for w in workers}
    assert worker_map["worker-a"]["job_count"] == 2
    assert worker_map["worker-b"]["job_count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_workers.py::test_list_workers_empty -v`
Expected: FAIL (404 or endpoint not found)

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
# Add import at top
from zndraw_joblib.schemas import (
    JobRegisterRequest,
    JobResponse,
    JobSummary,
    TaskSubmitRequest,
    TaskResponse,
    TaskClaimResponse,
    TaskUpdateRequest,
    WorkerSummary,  # Add this
)

# Add endpoint before the PATCH /workers/{worker_id}
@router.get("/workers", response_model=list[WorkerSummary])
async def list_workers(
    db: Session = Depends(get_db_session),
):
    """List all workers with their job counts."""
    workers = db.exec(select(Worker)).all()
    result = []
    for worker in workers:
        job_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)).all()
        )
        result.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_workers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py src/zndraw_joblib/schemas.py tests/test_router_workers.py
git commit -m "feat: add GET /workers endpoint to list all workers"
```

---

## Task 3: Add GET /rooms/{room_id}/workers Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_workers.py`

**Step 1: Write the failing test**

Add to `tests/test_router_workers.py`:

```python
@pytest.mark.asyncio
async def test_list_workers_for_room_empty(client: AsyncClient):
    """List workers for room returns empty list when no workers."""
    response = await client.get("/v1/joblib/rooms/my-room/workers")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_workers_for_room_filters_by_room(client: AsyncClient, register_job):
    """List workers for room only returns workers serving that room."""
    # Worker A serves room1 and @global
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    await register_job("@global", "modifiers", "global-job", worker_id="worker-a", is_admin=True)

    # Worker B serves room2
    await register_job("room2", "modifiers", "job2", worker_id="worker-b")

    # Worker C serves room1
    await register_job("room1", "selections", "job3", worker_id="worker-c")

    # List workers for room1 - should include A (has room1 job) and C, plus workers with @global jobs
    response = await client.get("/v1/joblib/rooms/room1/workers")
    assert response.status_code == 200
    workers = response.json()
    worker_ids = {w["id"] for w in workers}

    # Should include worker-a (room1 + @global), worker-c (room1)
    # Should also include worker-a due to @global
    assert "worker-a" in worker_ids
    assert "worker-c" in worker_ids
    assert "worker-b" not in worker_ids  # Only serves room2


@pytest.mark.asyncio
async def test_list_workers_for_global_room(client: AsyncClient, register_job):
    """List workers for @global room only returns workers serving @global jobs."""
    await register_job("@global", "modifiers", "global-job", worker_id="worker-a", is_admin=True)
    await register_job("room1", "modifiers", "job1", worker_id="worker-b")

    response = await client.get("/v1/joblib/rooms/@global/workers")
    assert response.status_code == 200
    workers = response.json()
    worker_ids = {w["id"] for w in workers}

    assert "worker-a" in worker_ids
    assert "worker-b" not in worker_ids
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_workers.py::test_list_workers_for_room_empty -v`
Expected: FAIL (404)

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py` (near other room-scoped endpoints):

```python
@router.get("/rooms/{room_id}/workers", response_model=list[WorkerSummary])
async def list_workers_for_room(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    """List workers serving jobs in a room. Includes @global workers unless room_id is @global."""
    validate_room_id(room_id)

    # Find jobs for this room (and @global if not requesting @global specifically)
    if room_id == "@global":
        jobs = db.exec(
            select(Job).where(Job.room_id == "@global", Job.deleted == False)
        ).all()
    else:
        jobs = db.exec(
            select(Job).where(
                (Job.room_id == "@global") | (Job.room_id == room_id),
                Job.deleted == False,
            )
        ).all()

    job_ids = [job.id for job in jobs]
    if not job_ids:
        return []

    # Find workers linked to these jobs
    worker_ids = db.exec(
        select(WorkerJobLink.worker_id).where(WorkerJobLink.job_id.in_(job_ids)).distinct()
    ).all()

    if not worker_ids:
        return []

    workers = db.exec(select(Worker).where(Worker.id.in_(worker_ids))).all()
    result = []
    for worker in workers:
        job_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)).all()
        )
        result.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_workers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_workers.py
git commit -m "feat: add GET /rooms/{room_id}/workers endpoint"
```

---

## Task 4: Add TaskSummary Schema with Queue Position

**Files:**
- Modify: `src/zndraw_joblib/schemas.py`

**Step 1: Add TaskSummary schema**

Add to `src/zndraw_joblib/schemas.py`:

```python
class TaskSummary(BaseModel):
    id: UUID
    job_name: str
    status: TaskStatus
    created_at: datetime
    queue_position: Optional[int] = None  # Only set for PENDING tasks
```

**Step 2: Commit**

```bash
git add src/zndraw_joblib/schemas.py
git commit -m "feat: add TaskSummary schema with queue_position field"
```

---

## Task 5: Add GET /rooms/{room_id}/tasks Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_tasks.py`

**Step 1: Write the failing test**

Add to `tests/test_router_tasks.py`:

```python
@pytest.mark.asyncio
async def test_list_tasks_for_room_empty(client: AsyncClient):
    """List tasks for room returns empty list when no tasks exist."""
    response = await client.get("/v1/joblib/rooms/my-room/tasks")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_tasks_for_room_returns_tasks(client: AsyncClient, register_job, submit_task):
    """List tasks for room returns all tasks submitted to that room."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    await register_job("room2", "modifiers", "job2", worker_id="worker-b")

    # Submit tasks to different rooms
    task1 = await submit_task("room1", "room1:modifiers:job1", {"data": 1})
    task2 = await submit_task("room1", "room1:modifiers:job1", {"data": 2})
    task3 = await submit_task("room2", "room2:modifiers:job2", {"data": 3})

    response = await client.get("/v1/joblib/rooms/room1/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 2

    task_ids = {t["id"] for t in tasks}
    assert str(task1["id"]) in task_ids
    assert str(task2["id"]) in task_ids
    assert str(task3["id"]) not in task_ids


@pytest.mark.asyncio
async def test_list_tasks_for_room_with_status_filter(client: AsyncClient, register_job, submit_task):
    """List tasks for room can filter by status."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    await submit_task("room1", "room1:modifiers:job1", {"data": 1})
    task2 = await submit_task("room1", "room1:modifiers:job1", {"data": 2})

    # Cancel second task
    await client.patch(f"/v1/joblib/tasks/{task2['id']}", json={"status": "cancelled"})

    # Filter for pending only
    response = await client.get("/v1/joblib/rooms/room1/tasks?status=pending")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_list_tasks_for_room_includes_queue_position(client: AsyncClient, register_job, submit_task):
    """List tasks includes queue_position for pending tasks."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    task1 = await submit_task("room1", "room1:modifiers:job1", {"data": 1})
    task2 = await submit_task("room1", "room1:modifiers:job1", {"data": 2})
    task3 = await submit_task("room1", "room1:modifiers:job1", {"data": 3})

    response = await client.get("/v1/joblib/rooms/room1/tasks")
    assert response.status_code == 200
    tasks = response.json()

    # Sort by queue_position to verify ordering
    task_map = {t["id"]: t for t in tasks}
    assert task_map[str(task1["id"])]["queue_position"] == 1
    assert task_map[str(task2["id"])]["queue_position"] == 2
    assert task_map[str(task3["id"])]["queue_position"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_tasks.py::test_list_tasks_for_room_empty -v`
Expected: FAIL (404)

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
# Add import
from zndraw_joblib.schemas import (
    # ... existing imports ...
    TaskSummary,
)
from typing import Optional

# Add endpoint
@router.get("/rooms/{room_id}/tasks", response_model=list[TaskSummary])
async def list_tasks_for_room(
    room_id: str,
    status: Optional[TaskStatus] = None,
    db: Session = Depends(get_db_session),
):
    """List tasks for a room, optionally filtered by status. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    query = select(Task).where(Task.room_id == room_id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    tasks = db.exec(query).all()

    # Calculate queue positions for pending tasks (per job)
    # Queue position = count of pending tasks for same job created before this task + 1
    result = []
    for task in tasks:
        job = db.exec(select(Job).where(Job.id == task.job_id)).first()

        queue_position = None
        if task.status == TaskStatus.PENDING:
            # Count pending tasks for same job created before this task
            count = db.exec(
                select(Task).where(
                    Task.job_id == task.job_id,
                    Task.status == TaskStatus.PENDING,
                    Task.created_at < task.created_at,
                )
            ).all()
            queue_position = len(count) + 1

        result.append(
            TaskSummary(
                id=task.id,
                job_name=job.full_name if job else "",
                status=task.status,
                created_at=task.created_at,
                queue_position=queue_position,
            )
        )
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_tasks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py src/zndraw_joblib/schemas.py tests/test_router_tasks.py
git commit -m "feat: add GET /rooms/{room_id}/tasks endpoint with queue position"
```

---

## Task 6: Add GET /rooms/{room_id}/jobs/{job_name}/tasks Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_tasks.py`

**Step 1: Write the failing test**

Add to `tests/test_router_tasks.py`:

```python
@pytest.mark.asyncio
async def test_list_tasks_for_job_empty(client: AsyncClient, register_job):
    """List tasks for job returns empty list when no tasks exist."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    response = await client.get("/v1/joblib/rooms/room1/jobs/room1:modifiers:job1/tasks")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_tasks_for_job_filters_by_job(client: AsyncClient, register_job, submit_task):
    """List tasks for job only returns tasks for that specific job."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    await register_job("room1", "modifiers", "job2", worker_id="worker-a")

    task1 = await submit_task("room1", "room1:modifiers:job1", {"data": 1})
    task2 = await submit_task("room1", "room1:modifiers:job2", {"data": 2})

    response = await client.get("/v1/joblib/rooms/room1/jobs/room1:modifiers:job1/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == str(task1["id"])


@pytest.mark.asyncio
async def test_list_tasks_for_job_not_found(client: AsyncClient):
    """List tasks for non-existent job returns 404."""
    response = await client.get("/v1/joblib/rooms/room1/jobs/room1:modifiers:nonexistent/tasks")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_tasks_for_global_job_from_room(client: AsyncClient, register_job, submit_task):
    """Can list tasks for @global job from any room."""
    await register_job("@global", "modifiers", "global-job", worker_id="worker-a", is_admin=True)

    # Submit task from room1 to global job
    task1 = await submit_task("room1", "@global:modifiers:global-job", {"data": 1})

    response = await client.get("/v1/joblib/rooms/room1/jobs/@global:modifiers:global-job/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == str(task1["id"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_tasks.py::test_list_tasks_for_job_empty -v`
Expected: FAIL (404)

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
@router.get("/rooms/{room_id}/jobs/{job_name:path}/tasks", response_model=list[TaskSummary])
async def list_tasks_for_job(
    room_id: str,
    job_name: str,
    status: Optional[TaskStatus] = None,
    db: Session = Depends(get_db_session),
):
    """List tasks for a specific job. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    # Parse job_name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # Validate access (same logic as get_job)
    if room_id != "@global" and job_room_id not in ("@global", room_id):
        raise JobNotFound.exception(detail=f"Job '{job_name}' not accessible from room '{room_id}'")

    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job or job.deleted:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    query = select(Task).where(Task.job_id == job.id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    tasks = db.exec(query).all()

    result = []
    for task in tasks:
        queue_position = None
        if task.status == TaskStatus.PENDING:
            count = db.exec(
                select(Task).where(
                    Task.job_id == task.job_id,
                    Task.status == TaskStatus.PENDING,
                    Task.created_at < task.created_at,
                )
            ).all()
            queue_position = len(count) + 1

        result.append(
            TaskSummary(
                id=task.id,
                job_name=job.full_name,
                status=task.status,
                created_at=task.created_at,
                queue_position=queue_position,
            )
        )
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_tasks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_tasks.py
git commit -m "feat: add GET /rooms/{room_id}/jobs/{job_name}/tasks endpoint"
```

---

## Task 7: Add queue_position to GET /tasks/{task_id} Response

**Files:**
- Modify: `src/zndraw_joblib/schemas.py` (TaskResponse)
- Modify: `src/zndraw_joblib/router.py` (get_task_status)
- Test: `tests/test_router_tasks.py`

**Step 1: Write the failing test**

Add to `tests/test_router_tasks.py`:

```python
@pytest.mark.asyncio
async def test_get_task_includes_queue_position(client: AsyncClient, register_job, submit_task):
    """GET /tasks/{task_id} includes queue_position for pending tasks."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    task1 = await submit_task("room1", "room1:modifiers:job1", {"data": 1})
    task2 = await submit_task("room1", "room1:modifiers:job1", {"data": 2})
    task3 = await submit_task("room1", "room1:modifiers:job1", {"data": 3})

    # Check queue positions
    resp1 = await client.get(f"/v1/joblib/tasks/{task1['id']}")
    resp2 = await client.get(f"/v1/joblib/tasks/{task2['id']}")
    resp3 = await client.get(f"/v1/joblib/tasks/{task3['id']}")

    assert resp1.json()["queue_position"] == 1
    assert resp2.json()["queue_position"] == 2
    assert resp3.json()["queue_position"] == 3


@pytest.mark.asyncio
async def test_get_task_queue_position_null_for_non_pending(client: AsyncClient, register_job, submit_task):
    """GET /tasks/{task_id} returns null queue_position for non-pending tasks."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    task = await submit_task("room1", "room1:modifiers:job1", {"data": 1})

    # Cancel the task
    await client.patch(f"/v1/joblib/tasks/{task['id']}", json={"status": "cancelled"})

    response = await client.get(f"/v1/joblib/tasks/{task['id']}")
    assert response.json()["queue_position"] is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_tasks.py::test_get_task_includes_queue_position -v`
Expected: FAIL (queue_position not in response)

**Step 3: Modify TaskResponse schema**

Update `src/zndraw_joblib/schemas.py`:

```python
class TaskResponse(BaseModel):
    id: UUID
    job_name: str
    room_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    payload: dict[str, Any] = {}
    queue_position: Optional[int] = None  # Add this line
```

**Step 4: Update get_task_status endpoint**

Modify `get_task_status` in `src/zndraw_joblib/router.py`:

```python
@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: UUID,
    db: Session = Depends(get_db_session),
):
    """Get task status."""
    task = db.exec(select(Task).where(Task.id == task_id)).first()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    # Calculate queue position for pending tasks
    queue_position = None
    if task.status == TaskStatus.PENDING:
        count = db.exec(
            select(Task).where(
                Task.job_id == task.job_id,
                Task.status == TaskStatus.PENDING,
                Task.created_at < task.created_at,
            )
        ).all()
        queue_position = len(count) + 1

    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        payload=task.payload,
        queue_position=queue_position,
    )
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_router_tasks.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/zndraw_joblib/schemas.py src/zndraw_joblib/router.py tests/test_router_tasks.py
git commit -m "feat: add queue_position to TaskResponse"
```

---

## Task 8: Update Other Endpoints That Return TaskResponse

**Files:**
- Modify: `src/zndraw_joblib/router.py` (submit_task, claim_task, update_task_status)
- Test: Run existing tests

**Step 1: Update submit_task**

The `submit_task` endpoint should return `queue_position=1` for newly submitted tasks (or calculate if other pending tasks exist).

Update in `src/zndraw_joblib/router.py`:

```python
# In submit_task, before return:
    # Calculate queue position
    count = db.exec(
        select(Task).where(
            Task.job_id == job.id,
            Task.status == TaskStatus.PENDING,
            Task.created_at < task.created_at,
        )
    ).all()
    queue_position = len(count) + 1

    return TaskResponse(
        id=task.id,
        job_name=job.full_name,
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        payload=task.payload,
        queue_position=queue_position,
    )
```

**Step 2: Update claim_task**

Claimed tasks should have `queue_position=None` (no longer pending).

```python
# In claim_task, the task is CLAIMED so queue_position=None (already default)
```

**Step 3: Update update_task_status**

Add `queue_position=None` for non-pending, or calculate if still pending.

```python
# In update_task_status, before each return:
    queue_position = None
    if task.status == TaskStatus.PENDING:
        count = db.exec(
            select(Task).where(
                Task.job_id == task.job_id,
                Task.status == TaskStatus.PENDING,
                Task.created_at < task.created_at,
            )
        ).all()
        queue_position = len(count) + 1

    return TaskResponse(
        # ... existing fields ...
        queue_position=queue_position,
    )
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py
git commit -m "feat: add queue_position to all TaskResponse returns"
```

---

## Task 9: Change worker_count to workers list in Job Responses

**Files:**
- Modify: `src/zndraw_joblib/schemas.py` (JobResponse, JobSummary)
- Modify: `src/zndraw_joblib/router.py` (register_job, list_jobs, get_job)
- Test: `tests/test_router_jobs.py`

**Step 1: Write the failing test**

Add to `tests/test_router_jobs.py`:

```python
@pytest.mark.asyncio
async def test_register_job_returns_worker_ids(client: AsyncClient, register_job):
    """Register job returns list of worker IDs instead of count."""
    result = await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    assert "workers" in result
    assert isinstance(result["workers"], list)
    assert "worker-a" in result["workers"]
    assert "worker_count" not in result


@pytest.mark.asyncio
async def test_register_job_multiple_workers(client: AsyncClient, register_job):
    """Multiple workers registering same job shows all worker IDs."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    result = await register_job("room1", "modifiers", "job1", worker_id="worker-b")

    assert set(result["workers"]) == {"worker-a", "worker-b"}


@pytest.mark.asyncio
async def test_list_jobs_returns_worker_ids(client: AsyncClient, register_job):
    """List jobs returns worker IDs for each job."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")
    await register_job("room1", "modifiers", "job1", worker_id="worker-b")

    response = await client.get("/v1/joblib/rooms/room1/jobs")
    assert response.status_code == 200
    jobs = response.json()

    job = jobs[0]
    assert "workers" in job
    assert set(job["workers"]) == {"worker-a", "worker-b"}
    assert "worker_count" not in job


@pytest.mark.asyncio
async def test_get_job_returns_worker_ids(client: AsyncClient, register_job):
    """Get job returns worker IDs."""
    await register_job("room1", "modifiers", "job1", worker_id="worker-a")

    response = await client.get("/v1/joblib/rooms/room1/jobs/room1:modifiers:job1")
    assert response.status_code == 200
    job = response.json()

    assert "workers" in job
    assert "worker-a" in job["workers"]
    assert "worker_count" not in job
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_jobs.py::test_register_job_returns_worker_ids -v`
Expected: FAIL (workers not in response, worker_count is)

**Step 3: Update schemas**

Modify `src/zndraw_joblib/schemas.py`:

```python
class JobResponse(BaseModel):
    id: UUID
    room_id: str
    category: str
    name: str
    full_name: str
    schema: dict[str, Any]
    workers: list[str]  # Changed from worker_count: int


class JobSummary(BaseModel):
    full_name: str
    category: str
    name: str
    workers: list[str]  # Changed from worker_count: int
```

**Step 4: Update register_job endpoint**

Modify in `src/zndraw_joblib/router.py`:

```python
@router.put(
    "/rooms/{room_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_job(
    # ... existing parameters ...
):
    # ... existing logic until return ...

    # Get worker IDs for this job
    worker_links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    ).all()
    worker_ids = [link.worker_id for link in worker_links]

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        workers=worker_ids,
    )
```

**Step 5: Update list_jobs endpoint**

Modify in `src/zndraw_joblib/router.py`:

```python
@router.get("/rooms/{room_id}/jobs", response_model=list[JobSummary])
async def list_jobs(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    # ... existing query logic ...

    result = []
    for job in jobs:
        worker_links = db.exec(
            select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
        ).all()
        worker_ids = [link.worker_id for link in worker_links]
        result.append(
            JobSummary(
                full_name=job.full_name,
                category=job.category,
                name=job.name,
                workers=worker_ids,
            )
        )
    return result
```

**Step 6: Update get_job endpoint**

Modify in `src/zndraw_joblib/router.py`:

```python
@router.get("/rooms/{room_id}/jobs/{job_name:path}", response_model=JobResponse)
async def get_job(
    # ... existing parameters ...
):
    # ... existing logic until return ...

    worker_links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    ).all()
    worker_ids = [link.worker_id for link in worker_links]

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        workers=worker_ids,
    )
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_router_jobs.py -v`
Expected: PASS

**Step 8: Run all tests to check for regressions**

Run: `pytest tests/ -v`
Expected: PASS (may need to update existing tests that check for worker_count)

**Step 9: Commit**

```bash
git add src/zndraw_joblib/schemas.py src/zndraw_joblib/router.py tests/test_router_jobs.py
git commit -m "feat: change worker_count to workers list in JobResponse/JobSummary"
```

---

## Summary of New Endpoints

After implementation, these new endpoints will be available:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/workers` | List all workers with job counts |
| GET | `/rooms/{room_id}/workers` | List workers serving a room (includes @global) |
| GET | `/rooms/{room_id}/tasks` | List tasks for a room with queue position |
| GET | `/rooms/{room_id}/jobs/{job_name}/tasks` | List tasks for a specific job |

Plus:
- `queue_position` field added to all `TaskResponse` objects
- `workers: list[str]` replaces `worker_count: int` in `JobResponse` and `JobSummary`
