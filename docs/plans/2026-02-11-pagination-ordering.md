# Pagination Ordering (Newest First) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** All paginated list endpoints return items newest-first by default, using `created_at DESC` ordering.

**Architecture:** Add `created_at` column to `Job` and `Worker` models. Change all 5 paginated list endpoints to use `order_by(*.created_at.desc())`. Tasks already have `created_at` so just flip `.asc()` → `.desc()`.

**Tech Stack:** SQLAlchemy 2.0, FastAPI, pytest

---

### Task 1: Write failing tests for ordering

**Files:**
- Create: `tests/test_ordering.py`

**Step 1: Write the failing tests**

We need tests for all 5 paginated endpoints proving newest-first ordering. Each test creates items with a known time gap (using `time.sleep(0.01)`) and asserts the first page returns the most recently created item first.

```python
# tests/test_ordering.py
"""Tests that all paginated list endpoints return items newest-first (created_at DESC)."""

import time

import pytest

from zndraw_joblib.schemas import (
    JobSummary,
    PaginatedResponse,
    TaskResponse,
    WorkerSummary,
)


# ── Jobs ────────────────────────────────────────────────────────────────


@pytest.fixture
def ordered_job_client(client):
    """Client with 3 global jobs registered in known order."""
    for name in ["First", "Second", "Third"]:
        resp = client.put(
            "/v1/joblib/rooms/@global/jobs",
            json={"category": "modifiers", "name": name, "schema": {}},
        )
        assert resp.status_code == 201
        time.sleep(0.01)
    return client


def test_list_jobs_newest_first(ordered_job_client):
    """GET /rooms/{room_id}/jobs returns newest job first."""
    response = ordered_job_client.get("/v1/joblib/rooms/@global/jobs")
    page = PaginatedResponse[JobSummary].model_validate(response.json())
    names = [j.name for j in page.items]
    assert names == ["Third", "Second", "First"]


def test_list_jobs_pagination_preserves_order(ordered_job_client):
    """Paginating through jobs maintains newest-first order."""
    # Page 1: limit=2 → Third, Second
    resp1 = ordered_job_client.get("/v1/joblib/rooms/@global/jobs?limit=2&offset=0")
    page1 = PaginatedResponse[JobSummary].model_validate(resp1.json())
    # Page 2: limit=2, offset=2 → First
    resp2 = ordered_job_client.get("/v1/joblib/rooms/@global/jobs?limit=2&offset=2")
    page2 = PaginatedResponse[JobSummary].model_validate(resp2.json())

    all_names = [j.name for j in page1.items] + [j.name for j in page2.items]
    assert all_names == ["Third", "Second", "First"]


# ── Tasks for room ─────────────────────────────────────────────────────


@pytest.fixture
def ordered_task_client(seeded_client):
    """Seeded client with 3 tasks submitted in known order."""
    for i in range(3):
        resp = seeded_client.post(
            "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
            json={"payload": {"index": i}},
        )
        assert resp.status_code == 201
        time.sleep(0.01)
    return seeded_client


def test_list_tasks_for_room_newest_first(ordered_task_client):
    """GET /rooms/{room_id}/tasks returns newest task first."""
    response = ordered_task_client.get("/v1/joblib/rooms/room_1/tasks")
    page = PaginatedResponse[TaskResponse].model_validate(response.json())
    indices = [t.payload["index"] for t in page.items]
    assert indices == [2, 1, 0]


def test_list_tasks_for_room_pagination_preserves_order(ordered_task_client):
    """Paginating through tasks maintains newest-first order."""
    resp1 = ordered_task_client.get("/v1/joblib/rooms/room_1/tasks?limit=2&offset=0")
    page1 = PaginatedResponse[TaskResponse].model_validate(resp1.json())
    resp2 = ordered_task_client.get("/v1/joblib/rooms/room_1/tasks?limit=2&offset=2")
    page2 = PaginatedResponse[TaskResponse].model_validate(resp2.json())

    indices = [t.payload["index"] for t in page1.items] + [
        t.payload["index"] for t in page2.items
    ]
    assert indices == [2, 1, 0]


# ── Tasks for job ──────────────────────────────────────────────────────


def test_list_tasks_for_job_newest_first(ordered_task_client):
    """GET /rooms/{room_id}/jobs/{job}/tasks returns newest task first."""
    response = ordered_task_client.get(
        "/v1/joblib/rooms/room_1/jobs/@global:modifiers:Rotate/tasks"
    )
    page = PaginatedResponse[TaskResponse].model_validate(response.json())
    indices = [t.payload["index"] for t in page.items]
    assert indices == [2, 1, 0]


# ── Workers (global list) ──────────────────────────────────────────────


def test_list_workers_newest_first(client_factory):
    """GET /workers returns newest worker first."""
    worker_ids = []
    for name in ["worker-a", "worker-b", "worker-c"]:
        c = client_factory(name)
        resp = c.put(
            "/v1/joblib/rooms/@global/jobs",
            json={"category": "modifiers", "name": f"job-{name}", "schema": {}},
        )
        assert resp.status_code == 201
        worker_ids.append(resp.json()["worker_id"])
        time.sleep(0.01)

    response = c.get("/v1/joblib/workers")
    page = PaginatedResponse[WorkerSummary].model_validate(response.json())
    returned_ids = [str(w.id) for w in page.items]
    # Newest (worker-c) should be first
    assert returned_ids[0] == worker_ids[2]
    assert returned_ids[-1] == worker_ids[0]


# ── Workers for room ───────────────────────────────────────────────────


def test_list_workers_for_room_newest_first(client_factory):
    """GET /rooms/{room_id}/workers returns newest worker first."""
    worker_ids = []
    for name in ["worker-x", "worker-y", "worker-z"]:
        c = client_factory(name)
        resp = c.put(
            "/v1/joblib/rooms/room_1/jobs",
            json={"category": "modifiers", "name": f"job-{name}", "schema": {}},
        )
        assert resp.status_code == 201
        worker_ids.append(resp.json()["worker_id"])
        time.sleep(0.01)

    response = c.get("/v1/joblib/rooms/room_1/workers")
    page = PaginatedResponse[WorkerSummary].model_validate(response.json())
    returned_ids = [str(w.id) for w in page.items]
    # Newest (worker-z) should be first
    assert returned_ids[0] == worker_ids[2]
    assert returned_ids[-1] == worker_ids[0]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ordering.py -v`
Expected: FAIL — Job model has no `created_at`, task order is currently ASC not DESC

**Step 3: Commit the failing tests**

```bash
git add tests/test_ordering.py
git commit -m "test: add ordering tests for all paginated list endpoints"
```

---

### Task 2: Add `created_at` to Job and Worker models

**Files:**
- Modify: `src/zndraw_joblib/models.py:46-68` (Job class)
- Modify: `src/zndraw_joblib/models.py:70-88` (Worker class)

**Step 1: Add `created_at` column to both models**

In `Job` class, after the `deleted` field:
```python
created_at: Mapped[datetime] = mapped_column(
    DateTime, default=lambda: datetime.now(timezone.utc)
)
```

In `Worker` class, after the `last_heartbeat` field:
```python
created_at: Mapped[datetime] = mapped_column(
    DateTime, default=lambda: datetime.now(timezone.utc)
)
```

**Step 2: Run model tests to verify nothing breaks**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS

---

### Task 3: Add `order_by(created_at.desc())` to all 5 endpoints

**Files:**
- Modify: `src/zndraw_joblib/router.py:394-395` (list_jobs)
- Modify: `src/zndraw_joblib/router.py:443-448` (list_workers_for_room)
- Modify: `src/zndraw_joblib/router.py:485-489` (list_tasks_for_room)
- Modify: `src/zndraw_joblib/router.py:525-529` (list_tasks_for_job)
- Modify: `src/zndraw_joblib/router.py:861-862` (list_workers)

Changes:

1. **list_jobs** — add `.order_by(Job.created_at.desc())` before `.offset()`:
```python
base_query.options(selectinload(Job.workers))
    .order_by(Job.created_at.desc())
    .offset(offset).limit(limit)
```

2. **list_workers_for_room** — add `.order_by(Worker.created_at.desc())`:
```python
select(Worker)
    .where(Worker.id.in_(worker_id_query))
    .options(selectinload(Worker.jobs))
    .order_by(Worker.created_at.desc())
    .offset(offset).limit(limit)
```

3. **list_tasks_for_room** — flip `.asc()` → `.desc()`:
```python
.order_by(Task.created_at.desc())
```

4. **list_tasks_for_job** — flip `.asc()` → `.desc()`:
```python
.order_by(Task.created_at.desc())
```

5. **list_workers** — add `.order_by(Worker.created_at.desc())`:
```python
select(Worker).options(selectinload(Worker.jobs))
    .order_by(Worker.created_at.desc())
    .offset(offset).limit(limit)
```

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: ALL PASS (including the new ordering tests from Task 1)

**Step 3: Lint and format**

Run: `uv run ruff check --fix && uv run ruff format`

**Step 4: Commit**

```bash
git add src/zndraw_joblib/models.py src/zndraw_joblib/router.py
git commit -m "feat: order all paginated endpoints by created_at DESC (newest first)"
```
