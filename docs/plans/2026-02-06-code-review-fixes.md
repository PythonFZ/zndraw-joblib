# Code Review Fixes

Addresses issues from code review of `src/zndraw_joblib` and `tests/`.

## Scope

Correctness bugs and low-effort fixes only. Deferred: N+1 queries, pagination, long-poll session optimization.

## Fixes

### 1. Add auth to `update_task_status` (Critical)

**File:** `src/zndraw_joblib/router.py` — `update_task_status`

Add `user: CurrentUserDep` parameter. Authorization logic:
- Superuser: always allowed
- Otherwise: look up `task.worker_id` → `Worker.user_id`, verify matches `user.id`
- If task has no worker_id (not yet claimed), reject non-superusers

```python
async def update_task_status(
    task_id: UUID,
    request: TaskUpdateRequest,
    session: LockedSessionDep,
    user: CurrentUserDep,  # ADD THIS
):
    # ... fetch task ...

    # Authorization: worker owner or superuser
    if not user.is_superuser:
        if task.worker_id is None:
            raise Forbidden.exception(detail="Task not claimed by any worker")
        result = await session.execute(
            select(Worker).where(Worker.id == task.worker_id)
        )
        worker = result.scalar_one_or_none()
        if not worker or worker.user_id != user.id:
            raise Forbidden.exception(detail="Not authorized to update this task")

    # ... rest of endpoint ...
```

**Tests to update:** `tests/test_router_task_status.py` — all calls to `PATCH /tasks/{id}` must include auth headers (use the `authenticated_client` fixture).

### 2. Fix soft-deleted job re-registration (Important)

**File:** `src/zndraw_joblib/router.py` — `register_job` lines 190-218

Change the if/else logic to handle three cases:

```python
if existing_job and existing_job.deleted:
    # Re-activate soft-deleted job with new schema
    existing_job.deleted = False
    existing_job.schema_ = request.schema_
    job = existing_job
    # 201 since it's effectively a new registration
elif existing_job:
    # Active job exists - validate schema match (existing behavior)
    if existing_job.schema_ != request.schema_:
        raise SchemaConflict.exception(...)
    job = existing_job
    response.status_code = status.HTTP_200_OK
else:
    # Create new job (existing behavior)
    job = Job(...)
    session.add(job)
    await session.flush()
```

**New test:** Register → delete worker (triggers soft-delete) → re-register with different schema → should succeed with 201.

### 3. Add ForeignKey to `Task.created_by_id`

**File:** `src/zndraw_joblib/models.py` line 98

```python
# Before:
created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
    default=None, index=True, nullable=True
)

# After:
created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
    ForeignKey("user.id"), default=None, index=True, nullable=True
)
```

No `ondelete="CASCADE"` — we want tasks to retain history even if user is deleted. SQLite doesn't enforce FK by default anyway, but this documents intent.

### 4. Fix f-string logging in sweeper

**File:** `src/zndraw_joblib/sweeper.py` — 4 lines

```python
# Before:
logger.info(f"Cleaning up stale worker: {worker.id}")
logger.info(f"Starting sweeper with interval={interval}s, ...")
logger.info(f"Cleaned up {count} stale worker(s)")
logger.exception(f"Error in sweeper: {e}")

# After:
logger.info("Cleaning up stale worker: %s", worker.id)
logger.info("Starting sweeper with interval=%ss, worker_timeout=%ss", interval, settings.worker_timeout_seconds)
logger.info("Cleaned up %s stale worker(s)", count)
logger.exception("Error in sweeper: %s", e)
```

### 5. Fix pyproject.toml

**File:** `pyproject.toml`

- Set `description = "FastAPI job queue with SQLite-backed persistence for ZnDraw"`
- Remove `pydantic-socketio` from `dependencies`
- Remove `pydantic-socketio` entry from `[tool.uv.sources]`

### 6. Add `raise_for_status()` to client methods

**File:** `src/zndraw_joblib/client.py`

- `claim()` line 193: Add `response.raise_for_status()` before `TaskClaimResponse.model_validate(...)`
- `heartbeat()` line 234: Capture response and call `raise_for_status()`

### 7. Fix deprecated `datetime.utcnow()` in tests

**File:** `tests/test_schemas.py` lines 51, 69

```python
# Before:
created_at=datetime.utcnow(),

# After:
created_at=datetime.now(timezone.utc),
```

Add `from datetime import timezone` to imports.

## Test Plan

- Run full test suite after each fix
- Add new test for soft-deleted job re-registration
- Verify existing `test_router_task_status.py` tests pass with auth requirement
- Verify `test_client.py` still passes with `raise_for_status()` additions
