# Socket.IO Real-Time Events for zndraw-joblib

## Problem

zndraw-joblib is pure REST. The current zndraw-fastapi worker system uses Redis + Socket.IO
for real-time notifications (task available, status changes, job list invalidation). Migrating
to zndraw-joblib requires these events to live in the joblib package itself.

## Decision: Dependency-injected tsio

The `zndraw-socketio` `AsyncServerWrapper` is provided via a FastAPI dependency (`get_tsio`)
that host apps override — same pattern as `get_locked_async_session`. This keeps joblib
self-contained while the host only wires the dependency.

## Event Models (`events.py`)

All models are `frozen=True` for hashability/set dedup.

### JobsInvalidate
- **Payload**: none
- **Room**: `room:{room_id}` or `room:@global`
- **Trigger**: job registered/deleted, worker connected/disconnected
- **Consumer**: frontend (refetches job list)

### TaskAvailable
- **Payload**: `job_name`, `room_id`, `task_id`
- **Room**: `jobs:{full_name}` (e.g., `jobs:@global:modifiers:Rotate`)
- **Trigger**: task submitted (non-`@internal` jobs only)
- **Consumer**: ZnDraw client workers (trigger claim via REST)

### TaskStatus
- **Payload**: `id`, `name`, `room_id`, `status` (Literal), timestamps, `worker_id`, `error`
- **Room**: `room:{room_id}`
- **Trigger**: any task status transition
- **Consumer**: frontend (live status display)

## Emission Infrastructure

```python
class Emission(NamedTuple):
    event: BaseModel
    room: str
```

Functions that modify state return `set[Emission]`. Callers emit **after commit**.
Set dedup eliminates duplicate `JobsInvalidate` for the same room.

## Emit Points

### Router (emit inline after DB commit)

| Endpoint | Event |
|----------|-------|
| `PUT /rooms/{room_id}/jobs` | `JobsInvalidate` → `room:{room_id}` |
| `POST /rooms/{room_id}/tasks/{job_name}` | `TaskAvailable` → `jobs:{full_name}` + `TaskStatus` → `room:{room_id}` |
| `POST /tasks/claim` | `TaskStatus` → `room:{room_id}` |
| `PATCH /tasks/{task_id}` | `TaskStatus` → `room:{room_id}` |
| `DELETE /workers/{worker_id}` | `JobsInvalidate` + `TaskStatus` (from `_cleanup_worker`) |
| `POST /workers` | `JobsInvalidate` → `room:{room_id}` (per linked job) |

### Sweeper (return events, emit at top level)

| Function | Returns |
|----------|---------|
| `_soft_delete_orphan_job()` | `set[Emission]` (JobsInvalidate if deleted) |
| `_cleanup_worker()` | `set[Emission]` (TaskStatus per failed task + JobsInvalidate per orphan) |
| `cleanup_stale_workers()` | `tuple[int, set[Emission]]` |
| `cleanup_stuck_internal_tasks()` | `tuple[int, set[Emission]]` |
| `run_sweeper()` | emits accumulated events after each commit |

## Dependency

```python
# dependencies.py
async def get_tsio() -> AsyncServerWrapper | None:
    return None

TsioDep = Annotated[AsyncServerWrapper | None, Depends(get_tsio)]
```

## Room Conventions

| Room | Purpose |
|------|---------|
| `room:{room_id}` | Frontend clients in a room |
| `room:@global` | Frontend clients watching global jobs |
| `jobs:{full_name}` | Workers registered for a specific job |
