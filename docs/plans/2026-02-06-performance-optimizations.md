# Performance Optimizations

Addresses deferred performance issues from code review: long-poll session holding, N+1 queries, and missing pagination.

## 1. Long-Poll Session Factory

**Problem:** `get_task_status` holds an open session for up to 120s during long-poll.

**Fix:** Add `SessionFactoryDep` dependency that returns a context manager factory. The endpoint creates short-lived sessions on demand, one per poll iteration.

### New dependency (`dependencies.py`)

```python
from contextlib import asynccontextmanager

async def get_session_factory(
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> Callable[[], AsyncContextManager[AsyncSession]]:
    @asynccontextmanager
    async def create_session():
        async for session in get_async_session(auth_settings):
            yield session
    return create_session

SessionFactoryDep = Annotated[..., Depends(get_session_factory)]
```

### Endpoint change (`router.py`)

`get_task_status` switches from `session: SessionDep` to `session_factory: SessionFactoryDep`. All session usage wrapped in `async with session_factory() as session:`. Only one session open at a time.

### Test fixture (`conftest.py`)

Add `session_factory` fixture that wraps `async_session_factory` in the same pattern. Override `get_session_factory` in app dependency overrides.

## 2. N+1 Query Fixes with selectinload

**Problem:** List endpoints issue O(N) extra queries to fetch related data.

### `list_jobs` — selectinload workers

```python
result = await session.execute(
    select(Job)
    .where(...)
    .options(selectinload(Job.workers))
    .offset(offset).limit(limit)
)
for job in jobs:
    worker_ids = [w.id for w in job.workers]  # pre-loaded
```

### `list_workers_for_room` / `list_workers` — selectinload jobs

```python
result = await session.execute(
    select(Worker)
    .where(Worker.id.in_(worker_ids))
    .options(selectinload(Worker.jobs))
)
for worker in workers:
    job_count = len(worker.jobs)  # pre-loaded
```

### `list_tasks_*` — bulk `_task_responses`

Create `_bulk_task_responses(session, tasks)` that:
1. Uses `selectinload(Task.job)` when loading tasks (job already on object)
2. Computes queue positions in a single batch query for all pending tasks

```python
async def _bulk_task_responses(
    session: AsyncSession, tasks: list[Task]
) -> list[TaskResponse]:
    pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
    queue_positions = {}
    if pending_tasks:
        # Single query: count pending tasks created before each task, grouped
        for task in pending_tasks:
            # Use a window function or batch approach
            result = await session.execute(
                select(func.count())
                .select_from(Task)
                .where(
                    Task.job_id == task.job_id,
                    Task.status == TaskStatus.PENDING,
                    Task.created_at < task.created_at,
                )
            )
            queue_positions[task.id] = result.scalar() + 1

    return [
        TaskResponse(
            id=t.id,
            job_name=t.job.full_name if t.job else "",
            room_id=t.room_id,
            status=t.status,
            created_at=t.created_at,
            started_at=t.started_at,
            completed_at=t.completed_at,
            worker_id=t.worker_id,
            error=t.error,
            payload=t.payload,
            queue_position=queue_positions.get(t.id),
        )
        for t in tasks
    ]
```

Note: The queue_position computation is still O(N) queries for pending tasks. A true single-query batch using window functions (ROW_NUMBER) would be more optimal but significantly more complex. Since pending tasks in a listing are typically few, this is acceptable.

The `_task_response` (singular) helper remains for single-task endpoints.

## 3. Pagination with Envelope

### Generic schema (`schemas.py`)

```python
from typing import Generic, TypeVar
T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
```

### Endpoints affected

All 5 list endpoints gain `limit` and `offset` query params:

```python
@router.get("/rooms/{room_id}/jobs", response_model=PaginatedResponse[JobSummary])
async def list_jobs(
    room_id: str,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    # ... build base query ...
    total_result = await session.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()

    query = query.offset(offset).limit(limit)
    # ... execute and build response ...

    return PaginatedResponse(items=results, total=total, limit=limit, offset=offset)
```

### Breaking change

Response shape changes from `[...]` to `{"items": [...], "total": N, ...}`. No consumers exist yet (verified in zndraw-fastapi), so this is safe.

## Test Plan

- Update all existing list endpoint tests to expect envelope format (`.json()["items"]`)
- Add pagination-specific tests: default limit, custom limit/offset, offset beyond total
- Add test for session factory dependency
- Verify long-poll tests still pass with factory approach
- Verify N+1 fix by checking that selectinload queries are issued (test behavior, not query count)
