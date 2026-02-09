# Worker Disconnect Design

## Problem

When a worker's Socket.IO connection drops (crash, network loss, graceful shutdown), the server has no way to immediately clean up. It relies on the sweeper's heartbeat timeout, leaving tasks stuck in CLAIMED/RUNNING and SIDs lingering in Socket.IO rooms.

## Design

### Event Model Changes

Add `worker_id: str` to `JoinJobRoom` and `LeaveJobRoom`:

```python
class JoinJobRoom(FrozenEvent):
    job_name: str
    worker_id: str

class LeaveJobRoom(FrozenEvent):
    job_name: str
    worker_id: str
```

The client already has a server-assigned `worker_id` (UUID) by the time it emits these events.

### Public `cleanup_worker`

Rename `_cleanup_worker` to `cleanup_worker` in `sweeper.py` and export from `__init__.py`. No logic change. This allows the host app to call it from its disconnect handler.

### Client `disconnect()` and Context Manager

Add `disconnect()` to `JobManager`:

1. Emit `LeaveJobRoom` for each registered job (socket room cleanup)
2. Call `DELETE /workers/{worker_id}` (DB cleanup)
3. Clear local registry state

Add `__enter__`/`__exit__` for `with` block support.

### Host App Integration

The host app (zndraw-fastapi) wires up three things:

**`JoinJobRoom` handler** stores `worker_id` in the SIO session:

```python
@tsio.on(JoinJobRoom)
async def on_join_job_room(sid: str, data: JoinJobRoom):
    await tsio.enter_room(sid, f"jobs:{data.job_name}")
    session = await tsio.get_session(sid)
    session["worker_id"] = data.worker_id
    await tsio.save_session(sid, session)
```

**`LeaveJobRoom` handler** leaves the room (graceful path):

```python
@tsio.on(LeaveJobRoom)
async def on_leave_job_room(sid: str, data: LeaveJobRoom):
    await tsio.leave_room(sid, f"jobs:{data.job_name}")
```

**Existing `on_disconnect` handler** adds worker cleanup:

```python
worker_id = session.get("worker_id")
if worker_id:
    async with get_session() as db:
        worker = await db.get(Worker, UUID(worker_id))
        if worker:
            emissions = await cleanup_worker(db, worker)
            await db.commit()
            await emit(tsio, emissions)
```

The SID-to-worker mapping uses the existing `tsio.save_session`/`get_session` pattern (already used for `user_id` and `current_room_id`). No new infrastructure needed; sticky sessions are already required for Socket.IO.

### Disconnect Scenarios

| Scenario | Handler |
|----------|---------|
| Network drop / kill | Server-side SIO disconnect (immediate) |
| Graceful shutdown (`with manager:`) | Client `disconnect()` (LeaveJobRoom + DELETE) |
| REST-only workers (no SIO) | Sweeper heartbeat timeout |

## Files Changed

| File | Change |
|------|--------|
| `events.py` | Add `worker_id: str` to `JoinJobRoom` and `LeaveJobRoom` |
| `client.py` | Add `disconnect()`, `__enter__`, `__exit__`; pass `worker_id` in `JoinJobRoom` emit |
| `sweeper.py` | Rename `_cleanup_worker` to `cleanup_worker` |
| `__init__.py` | Export `cleanup_worker` |
| `router.py` | Update import `_cleanup_worker` to `cleanup_worker` |
| `README.md` | Add Host App Integration section with SIO handler examples |
| Tests | Update event tests, add disconnect/context manager tests |
