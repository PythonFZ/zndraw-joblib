# Fix: Shared jobs return tasks from all rooms + limit=0 validation

## Context

When a user opens Room A and Room B in separate browser tabs and runs an extension in Room A, the TaskHistoryPanel in Room B shows the **same tasks** as Room A. This happens because `@internal` and `@global` jobs are shared across rooms — they have a single `job_id` in the database. The `list_tasks_for_job` endpoint queries by `job_id` without filtering by `room_id`, so it returns tasks from all rooms.

Additionally, the frontend sends `limit=0` to fetch only the total count of pending tasks (count-only query), but the backend validates `limit >= 1`, causing repeated 422 errors and React Query retries.

## Changes

### 1. Add room filter to `list_tasks_for_job`

**File:** `src/zndraw_joblib/router.py` (line 518)

```python
# Before:
base_query = select(Task).where(Task.job_id == job.id)

# After:
base_query = select(Task).where(Task.job_id == job.id, Task.room_id == room_id)
```

This ensures shared jobs (`@internal:modifiers:Empty`, `@global:*:*`) only return tasks for the requesting room.

### 2. Allow `limit=0` for count-only queries

**File:** `src/zndraw_joblib/router.py`

Change `ge=1` to `ge=0` on all 5 paginated endpoints:
- Line 379: `list_jobs`
- Line 418: `list_workers_for_room`
- Line 472: `list_tasks_for_room`
- Line 510: `list_tasks_for_job`
- Line 856: `list_workers`

### 3. Capture orphan job emissions in `update_task_status`

**File:** `src/zndraw_joblib/router.py` (line 841)

```python
# Before:
await _soft_delete_orphan_job(session, task.job_id)
...
emissions: set[Emission] = {await _task_status_emission(session, task)}

# After:
orphan_emissions = await _soft_delete_orphan_job(session, task.job_id)
...
emissions: set[Emission] = {await _task_status_emission(session, task)}
emissions |= orphan_emissions
```

This follows the same pattern used in `cleanup_worker` (sweeper.py:137).

## Verification

1. `uv run pytest tests/` — all existing tests pass
2. `uv run pyright .` — no new type errors
3. `uv run ruff format . && uv run ruff check --select I --fix .` — formatting clean
4. Manual test: open two rooms in browser tabs, run extension in one, verify TaskHistoryPanel only shows that room's tasks
5. Manual test: verify pending task count chip works without 422 errors
