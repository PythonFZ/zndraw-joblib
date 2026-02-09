# Worker Disconnect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable automatic server-side worker cleanup on Socket.IO disconnect, plus a client-side `disconnect()` helper with context manager support.

**Architecture:** Add `worker_id` to `JoinJobRoom`/`LeaveJobRoom` events so the host app can map SID to worker. Make `cleanup_worker` public for host app disconnect handlers. Add `JobManager.disconnect()` and `__enter__`/`__exit__` on the client side.

**Tech Stack:** Pydantic event models, httpx client, SQLAlchemy async, pytest

---

### Task 1: Add `worker_id` to Event Models

**Files:**
- Modify: `src/zndraw_joblib/events.py:55-72`
- Modify: `tests/test_events.py:59-72`

**Step 1: Update failing tests for new `worker_id` field**

In `tests/test_events.py`, update the two existing tests to include `worker_id`:

```python
def test_join_job_room_frozen():
    ev = JoinJobRoom(job_name="@global:modifiers:Rotate", worker_id="abc-123")
    assert ev.job_name == "@global:modifiers:Rotate"
    assert ev.worker_id == "abc-123"
    with pytest.raises((ValidationError, AttributeError)):
        ev.job_name = "x"


def test_leave_job_room_frozen():
    ev = LeaveJobRoom(job_name="@global:modifiers:Rotate", worker_id="abc-123")
    assert ev.job_name == "@global:modifiers:Rotate"
    assert ev.worker_id == "abc-123"
    a = LeaveJobRoom(job_name="@global:modifiers:Rotate", worker_id="abc-123")
    b = LeaveJobRoom(job_name="@global:modifiers:Rotate", worker_id="abc-123")
    assert a == b
    assert hash(a) == hash(b)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_events.py::test_join_job_room_frozen tests/test_events.py::test_leave_job_room_frozen -v`
Expected: FAIL — `worker_id` is an unexpected keyword argument.

**Step 3: Add `worker_id` field to both event models**

In `src/zndraw_joblib/events.py`, update:

```python
class JoinJobRoom(FrozenEvent):
    """Worker requests to join a job's notification room.

    Sent by the client after REST job registration. The host app's
    socketio handler should call ``tsio.enter_room(sid, f"jobs:{job_name}")``
    and store the ``worker_id`` in the SIO session for disconnect cleanup.
    """

    job_name: str
    worker_id: str


class LeaveJobRoom(FrozenEvent):
    """Worker requests to leave a job's notification room.

    Sent by the client on graceful disconnect or job unregistration.
    The host app's handler should call ``tsio.leave_room(sid, f"jobs:{job_name}")``.
    """

    job_name: str
    worker_id: str
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/events.py tests/test_events.py
git commit -m "feat: add worker_id to JoinJobRoom and LeaveJobRoom events"
```

---

### Task 2: Make `cleanup_worker` Public

**Files:**
- Modify: `src/zndraw_joblib/sweeper.py:72` (rename function)
- Modify: `src/zndraw_joblib/router.py:53` (update import)
- Modify: `src/zndraw_joblib/__init__.py` (add export)

**Step 1: Rename `_cleanup_worker` to `cleanup_worker` in sweeper.py**

In `src/zndraw_joblib/sweeper.py`, change line 72:

```python
async def cleanup_worker(session: AsyncSession, worker: Worker) -> set[Emission]:
```

Also update the internal call on line 144:

```python
        emissions = await cleanup_worker(session, worker)
```

**Step 2: Update the import in router.py**

In `src/zndraw_joblib/router.py`, change line 53:

```python
from zndraw_joblib.sweeper import cleanup_worker, _soft_delete_orphan_job
```

**Step 3: Add export in `__init__.py`**

In `src/zndraw_joblib/__init__.py`, add to the sweeper imports:

```python
from zndraw_joblib.sweeper import (
    run_sweeper,
    cleanup_stale_workers,
    cleanup_stuck_internal_tasks,
    cleanup_worker,
)
```

And add `"cleanup_worker"` to the `__all__` list in the `# Sweeper` section:

```python
    # Sweeper
    "run_sweeper",
    "cleanup_stale_workers",
    "cleanup_stuck_internal_tasks",
    "cleanup_worker",
```

**Step 4: Run full test suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: All tests PASS (this is a pure rename)

**Step 5: Commit**

```bash
git add src/zndraw_joblib/sweeper.py src/zndraw_joblib/router.py src/zndraw_joblib/__init__.py
git commit -m "refactor: make cleanup_worker public for host app disconnect handlers"
```

---

### Task 3: Update Client to Pass `worker_id` in `JoinJobRoom`

**Files:**
- Modify: `src/zndraw_joblib/client.py:182-183`
- Modify: `tests/test_client.py:401-435`

**Step 1: Update existing tests to expect `worker_id` in emitted events**

In `tests/test_client.py`, update `test_job_manager_register_emits_join_job_room`:

```python
def test_job_manager_register_emits_join_job_room(client):
    """register() should emit JoinJobRoom with worker_id when tsio is provided."""
    mock_tsio = MagicMock()
    api = MockClientApi(client)
    manager = JobManager(api, tsio=mock_tsio)

    @manager.register
    class Rotate(ConcreteExtension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    mock_tsio.emit.assert_called_once()
    event = mock_tsio.emit.call_args[0][0]
    assert isinstance(event, JoinJobRoom)
    assert event.job_name == "@global:modifiers:Rotate"
    assert event.worker_id == str(manager.worker_id)
```

Update `test_job_manager_register_emits_join_for_each_job`:

```python
def test_job_manager_register_emits_join_for_each_job(client):
    """register() should emit JoinJobRoom with worker_id for each registered job."""
    mock_tsio = MagicMock()
    api = MockClientApi(client)
    manager = JobManager(api, tsio=mock_tsio)

    @manager.register
    class Job1(ConcreteExtension):
        category: ClassVar[Category] = Category.MODIFIER

    @manager.register
    class Job2(ConcreteExtension):
        category: ClassVar[Category] = Category.SELECTION

    assert mock_tsio.emit.call_count == 2
    events = [call[0][0] for call in mock_tsio.emit.call_args_list]
    worker_id = str(manager.worker_id)
    assert events[0] == JoinJobRoom(job_name="@global:modifiers:Job1", worker_id=worker_id)
    assert events[1] == JoinJobRoom(job_name="@global:selections:Job2", worker_id=worker_id)
```

Update `test_job_manager_register_room_emits_correct_job_name`:

```python
def test_job_manager_register_room_emits_correct_job_name(client):
    """register(room=...) should emit JoinJobRoom with room-scoped job name and worker_id."""
    mock_tsio = MagicMock()
    api = MockClientApi(client)
    manager = JobManager(api, tsio=mock_tsio)

    @manager.register(room="my_room")
    class PrivateJob(ConcreteExtension):
        category: ClassVar[Category] = Category.MODIFIER

    event = mock_tsio.emit.call_args[0][0]
    assert isinstance(event, JoinJobRoom)
    assert event.job_name == "my_room:modifiers:PrivateJob"
    assert event.worker_id == str(manager.worker_id)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_client.py::test_job_manager_register_emits_join_job_room tests/test_client.py::test_job_manager_register_emits_join_for_each_job tests/test_client.py::test_job_manager_register_room_emits_correct_job_name -v`
Expected: FAIL — `worker_id` missing from emitted events.

**Step 3: Update `_register_impl` to pass `worker_id`**

In `src/zndraw_joblib/client.py`, change line 183:

```python
        if self.tsio is not None:
            self.tsio.emit(
                JoinJobRoom(
                    job_name=full_name, worker_id=str(self._worker_id)
                )
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/client.py tests/test_client.py
git commit -m "feat: pass worker_id in JoinJobRoom emission from client"
```

---

### Task 4: Add `disconnect()` and Context Manager to `JobManager`

**Files:**
- Modify: `src/zndraw_joblib/client.py:16,79-98` (add import, add methods)
- Modify: `tests/test_client.py` (add new tests)

**Step 1: Write tests for `disconnect()` and context manager**

Add these imports at the top of `tests/test_client.py`:

```python
from zndraw_joblib.events import JoinJobRoom, LeaveJobRoom
```

(Replace the existing `from zndraw_joblib.events import JoinJobRoom` line.)

Add these tests at the end of `tests/test_client.py`:

```python
def test_job_manager_disconnect_emits_leave_and_deletes_worker(client):
    """disconnect() should emit LeaveJobRoom for each job and DELETE the worker."""
    mock_tsio = MagicMock()
    api = MockClientApi(client)
    manager = JobManager(api, tsio=mock_tsio)

    @manager.register
    class Job1(ConcreteExtension):
        category: ClassVar[Category] = Category.MODIFIER

    @manager.register
    class Job2(ConcreteExtension):
        category: ClassVar[Category] = Category.SELECTION

    worker_id = manager.worker_id
    assert worker_id is not None
    mock_tsio.reset_mock()

    manager.disconnect()

    # Should have emitted LeaveJobRoom for each job
    assert mock_tsio.emit.call_count == 2
    events = [call[0][0] for call in mock_tsio.emit.call_args_list]
    assert all(isinstance(e, LeaveJobRoom) for e in events)
    job_names = {e.job_name for e in events}
    assert job_names == {"@global:modifiers:Job1", "@global:selections:Job2"}

    # Local state should be cleared
    assert len(manager) == 0
    assert manager.worker_id is None

    # Worker should be deleted from server
    resp = client.patch(f"/v1/joblib/workers/{worker_id}")
    assert resp.status_code == 404


def test_job_manager_disconnect_no_tsio(client):
    """disconnect() should work without tsio (REST-only cleanup)."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class RestJob(ConcreteExtension):
        category: ClassVar[Category] = Category.MODIFIER

    worker_id = manager.worker_id
    assert worker_id is not None

    manager.disconnect()

    assert len(manager) == 0
    assert manager.worker_id is None

    # Worker should be deleted
    resp = client.patch(f"/v1/joblib/workers/{worker_id}")
    assert resp.status_code == 404


def test_job_manager_disconnect_no_worker_id():
    """disconnect() should be safe to call when no worker_id is set."""
    api = MagicMock()
    manager = JobManager(api)
    manager.disconnect()  # Should not raise
    assert manager.worker_id is None


def test_job_manager_context_manager(client):
    """JobManager should support with-statement for automatic disconnect."""
    mock_tsio = MagicMock()
    api = MockClientApi(client)

    with JobManager(api, tsio=mock_tsio) as manager:
        @manager.register
        class CtxJob(ConcreteExtension):
            category: ClassVar[Category] = Category.MODIFIER

        worker_id = manager.worker_id
        assert worker_id is not None

    # After exiting context, state should be cleared
    assert manager.worker_id is None
    assert len(manager) == 0

    # Worker should be deleted from server
    resp = client.patch(f"/v1/joblib/workers/{worker_id}")
    assert resp.status_code == 404
```

**Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_client.py::test_job_manager_disconnect_emits_leave_and_deletes_worker tests/test_client.py::test_job_manager_disconnect_no_tsio tests/test_client.py::test_job_manager_disconnect_no_worker_id tests/test_client.py::test_job_manager_context_manager -v`
Expected: FAIL — `disconnect` and `__enter__` not defined.

**Step 3: Implement `disconnect()`, `__enter__`, `__exit__`**

In `src/zndraw_joblib/client.py`, add `LeaveJobRoom` to the events import (line 16):

```python
from zndraw_joblib.events import JoinJobRoom, LeaveJobRoom
```

Add these methods to `JobManager` after the existing `__iter__` method (after line 98):

```python
    def __enter__(self) -> "JobManager":
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    def disconnect(self) -> None:
        """Gracefully disconnect the worker.

        1. Emits LeaveJobRoom for each registered job (socket room cleanup)
        2. Calls DELETE /workers/{worker_id} (DB cleanup: fail tasks, remove links, soft-delete orphan jobs)
        3. Clears local registry state
        """
        if self.tsio is not None and self._worker_id is not None:
            for job_name in self._registry:
                self.tsio.emit(
                    LeaveJobRoom(
                        job_name=job_name, worker_id=str(self._worker_id)
                    )
                )

        if self._worker_id is not None:
            resp = self.api.http.delete(
                f"{self.api.base_url}/v1/joblib/workers/{self._worker_id}",
                headers=self.api.get_headers(),
            )
            resp.raise_for_status()

        self._registry.clear()
        self._worker_id = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/client.py tests/test_client.py
git commit -m "feat: add JobManager.disconnect() and context manager support"
```

---

### Task 5: Update README with Host App Integration Section

**Files:**
- Modify: `README.md`

**Step 1: Update the Worker Notification Pattern section in README.md**

Replace the existing "Worker Notification Pattern" section (lines ~342-367) and add a new "Server-Side Disconnect Cleanup" section after it. The JoinJobRoom/LeaveJobRoom handler examples need to include `worker_id` and the disconnect cleanup pattern.

Replace the existing `JoinJobRoom`/`LeaveJobRoom` handler example block:

```python
@tsio.on(JoinJobRoom)
async def handle_join(sid: str, data: JoinJobRoom):
    await tsio.enter_room(sid, f"jobs:{data.job_name}")
```

with:

```python
@tsio.on(JoinJobRoom)
async def handle_join(sid: str, data: JoinJobRoom):
    await tsio.enter_room(sid, f"jobs:{data.job_name}")
    session = await tsio.get_session(sid)
    session["worker_id"] = data.worker_id
    await tsio.save_session(sid, session)

@tsio.on(LeaveJobRoom)
async def handle_leave(sid: str, data: LeaveJobRoom):
    await tsio.leave_room(sid, f"jobs:{data.job_name}")
```

Also update the JoinJobRoom/LeaveJobRoom event table row to show the new `worker_id` payload:

| `JoinJobRoom` | `job_name`, `worker_id` | *(client -> server)* | Worker joins a job's notification room after REST registration |
| `LeaveJobRoom` | `job_name`, `worker_id` | *(client -> server)* | Worker leaves a job's notification room |

Add a new section after the handler examples:

```markdown
### Server-Side Disconnect Cleanup

When a worker's Socket.IO connection drops (crash, network loss), the host app can
immediately clean up by calling `cleanup_worker` from its existing disconnect handler.
The `worker_id` stored in the SIO session during `JoinJobRoom` enables the mapping:

\```python
from zndraw_joblib import cleanup_worker, emit

@tsio.on("disconnect")
async def on_disconnect(sid: str, reason: str):
    session = await tsio.get_session(sid)

    # ... existing cleanup (presence, locks, etc.) ...

    # Worker cleanup
    worker_id = session.get("worker_id")
    if worker_id:
        async with get_session() as db:
            worker = await db.get(Worker, UUID(worker_id))
            if worker:
                emissions = await cleanup_worker(db, worker)
                await db.commit()
                await emit(tsio, emissions)
\```

This provides immediate cleanup (fail stuck tasks, remove job links, soft-delete
orphan jobs) without waiting for the background sweeper's heartbeat timeout.

| Disconnect Scenario | Handler |
|---------------------|---------|
| Network drop / process kill | Server-side SIO disconnect (immediate) |
| Graceful shutdown (`with manager:`) | Client `disconnect()` emits `LeaveJobRoom` + calls `DELETE /workers` |
| REST-only workers (no SIO) | Background sweeper heartbeat timeout |
```

Also update the Client usage example at the bottom of the README to show the context manager pattern:

```python
# Context manager for automatic cleanup
with JobManager(api, tsio=tsio) as manager:
    @manager.register
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    for task in manager.listen(stop_event=shutdown):
        manager.heartbeat()
        task.extension.run()
# disconnect() called automatically: LeaveJobRoom emitted, DELETE /workers called
```

**Step 2: Verify lint passes**

Run: `uv run ruff check`
Expected: No errors

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add server-side disconnect cleanup guide to README"
```

---

### Task 6: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Run linter and formatter**

Run: `uv run ruff check && uv run ruff format --check`
Expected: No errors

**Step 3: Verify exports work**

Run: `uv run python -c "from zndraw_joblib import cleanup_worker; print('OK')"`
Expected: prints `OK`
