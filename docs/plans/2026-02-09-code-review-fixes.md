# Code Review Fixes — Implementation Plan

**Date:** 2026-02-09
**Scope:** All 53 findings from comprehensive code review (3+1 Critical, 14+9 Important, 13+13 Minor)
**Branch:** `fix/code-review-2026-02-09`

## Decisions

- Event emission: router style (pass Pydantic models to zndraw-socketio API)
- DB lock: move to `app.state`, initialize via lifespan
- Settings: replace `@lru_cache` with proper FastAPI dependency
- Extension: make ABC with `@abstractmethod`
- Performance: fix now (window function, exponential backoff, composite index)
- SeededClient: dataclass instead of monkey-patched TestClient

---

## WS1: Event Emission Unification

**Files:** `events.py`, `router.py`, `sweeper.py`
**Findings:** Critical #2, #3, Minor #20

### Task 1.1: Add `FrozenEvent` base class to `events.py`

- Create `FrozenEvent(BaseModel)` with `model_config = ConfigDict(frozen=True)`
- All 5 event models (`JobsInvalidate`, `TaskAvailable`, `TaskStatusEvent`, `JoinJobRoom`, `LeaveJobRoom`) inherit from `FrozenEvent` instead of `BaseModel`
- Update `Emission.event` type from `BaseModel` to `FrozenEvent`

### Task 1.2: Move `_task_status_emission` to `events.py`

- Move the router's version (async, queries job, computes queue_position) to `events.py` as `async def task_status_emission(session, task) -> Emission`
- Delete the sweeper's version (sync, no queue_position)
- Delete the router's local version
- Both `router.py` and `sweeper.py` import from `events.py`

### Task 1.3: Move `_emit` to `events.py`

- Move `async def emit(tsio, emissions: set[Emission]) -> None` to `events.py`
- Router-style: `await tsio.emit(emission.event, room=emission.room)`
- Delete router's local `_emit`
- Sweeper's `run_sweeper` replaces manual serialization with `await emit(tsio, emissions)`

### Task 1.4: Move `_soft_delete_orphan_job` to shared location

- Currently duplicated between `router.py` and `sweeper.py`
- Move to `events.py` (or a new `helpers.py` if `events.py` gets too large)
- Both files import the shared version

---

## WS2: Internal Dispatch Race Condition

**Files:** `router.py`
**Findings:** Critical #1

### Task 2.1: Move registry check before task creation in `submit_task`

- Before `Task(...)` creation, check:
  ```python
  if job.room_id == "@internal":
      if internal_registry is None or job.full_name not in internal_registry.tasks:
          raise InternalJobNotConfigured.exception(...)
  ```
- Remove the post-commit check

### Task 2.2: Wrap `kiq()` in try/except

- After commit, if `@internal`:
  ```python
  try:
      await internal_registry.tasks[job.full_name].kiq(...)
  except Exception:
      task.status = TaskStatus.FAILED
      task.completed_at = func.now()
      await session.commit()
      raise
  ```
- Move event emission to after successful dispatch for `@internal` jobs

---

## WS3: Router Refactoring

**Files:** `router.py`, `schemas.py`, `models.py`
**Findings:** Important #4,5,6,7,10,11,13,14,16,17, Minor #18,19,21,22,23,29

### Task 3.1: Extract `_room_job_filter(room_id)` helper

- Replace duplicated ternary in `list_jobs` and `list_workers_for_room`
- Returns a SQLAlchemy filter expression

### Task 3.2: Extract `async def paginate(session, query, limit, offset)` utility

- Returns `tuple[list[T], int]`
- Replaces 5 repetitions of count + offset + limit boilerplate
- Used by `list_jobs`, `list_workers_for_room`, `list_tasks_for_room`, `list_tasks_for_job`, `list_workers`

### Task 3.3: Extract `_find_or_create_job()` and `_ensure_worker_link()` from `register_job`

- `register_job` becomes a thin orchestrator calling these helpers
- Each helper is ~30 lines with a single responsibility

### Task 3.4: Extract `_list_tasks()` shared helper

- `list_tasks_for_room` and `list_tasks_for_job` become thin wrappers that build `base_query` and delegate to `_list_tasks(session, base_query, limit, offset)`

### Task 3.5: Window function for queue positions

- Replace N+1 `_queue_position()` calls in `_bulk_task_responses` with single query:
  ```sql
  SELECT id, ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY created_at)
  FROM task WHERE status = 'pending' AND id IN (...)
  ```
- Delete `_queue_position()` function

### Task 3.6: Use eager-loaded `task.job` in `_task_response`

- When relationship is loaded, use `task.job` directly
- Remove the separate `select(Job).where(Job.id == task.job_id)` query
- Ensure all callers use `selectinload(Task.job)` when loading tasks

### Task 3.7: Add composite index on Task model

- Add `Index("ix_task_job_status_created", "job_id", "status", "created_at")` to `Task.__table_args__`

### Task 3.8: Single transaction for task update + orphan cleanup

- In `update_task_status`, remove the second `await session.commit()`
- Do task status update and orphan job soft-delete in one commit

### Task 3.9: Log + raise after max claim retries

- After exhausting `claim_max_attempts`, log the error and raise an appropriate HTTP error instead of silently returning `task=None`
- Only catch `OperationalError` subtypes related to locking, not all `OperationalError`

### Task 3.10: Validate empty room_id

- In `validate_room_id`, add: `if not room_id: raise InvalidRoomId.exception(...)`

### Task 3.11: Type safety cleanup

- `SessionFactoryDep`: type as `Callable[..., AsyncContextManager[AsyncSession]]` or a Protocol
- Rename `status` parameter to `task_status` in `list_tasks_for_room` and `list_tasks_for_job`
- Replace `Optional[X]` with `X | None` throughout
- `TaskSubmitRequest.payload`: use `Field(default_factory=dict)`
- Clean up `uuid` imports in `models.py`: use `from uuid import UUID, uuid4`

---

## WS4: Sweeper, Dependencies & Settings

**Files:** `sweeper.py`, `dependencies.py`, `registry.py`, `settings.py`
**Findings:** Important #8,9,12, Minor #26,27,28,30

### Task 4.1: Type `tsio` properly in sweeper

- Replace `tsio: Any = None` with `tsio: AsyncServerWrapper | None = None`
- Import `AsyncServerWrapper` from events or appropriate module

### Task 4.2: Move `_db_lock` to `app.state` via lifespan

- Add lifespan context manager that creates `app.state.db_lock = asyncio.Lock()`
- Add `get_db_lock` dependency that reads from `request.app.state`
- Remove module-level `_db_lock` singleton from `dependencies.py`
- Update `get_locked_async_session` to use the dependency

### Task 4.3: Replace `get_settings` `@lru_cache` with FastAPI dependency

- Remove `@lru_cache` from `get_settings`
- Make it a regular FastAPI dependency (overridable in tests)
- Update all `Depends(get_settings)` call sites (should work as-is)

### Task 4.4: Type `registry.py` parameters properly

- `app: FastAPI` instead of `Any`
- `session_factory: Callable[..., AsyncContextManager[AsyncSession]]` instead of `Any`
- `payload: dict[str, Any]` in `_execute` closure

### Task 4.5: Long-poll exponential backoff

- In `get_task_status`, replace fixed 1s interval with exponential backoff:
  - Start at 1s, multiply by 1.5 each iteration, cap at 5s
- Reduce `long_poll_max_wait_seconds` default from 120 to 60

---

## WS5: Client SDK Fixes

**Files:** `client.py`
**Findings:** Minor #24, #25

### Task 5.1: Make `Extension` an ABC

- `Extension` inherits from `BaseModel, ABC`
- `run()` decorated with `@abstractmethod`, remove `NotImplementedError` body

### Task 5.2: Add stop mechanism to `listen()`

- Accept optional `stop_event: threading.Event | None = None`
- Check `stop_event.is_set()` each iteration
- `while not (stop_event and stop_event.is_set()):`

---

## WS6: Test DRY & Fixtures

**Files:** `conftest.py`, `test_router_task_status.py`, `test_router_events.py`, `test_client.py`, `test_events.py`, `test_settings.py`
**Findings:** Test Important #2,7,8,9, Test Minor #1,3,4,5,11,12,13,14,15

### Task 6.1: Extract `_build_app()` in conftest.py

- Shared function that creates FastAPI app with all dependency overrides
- Both `app` fixture and `client_factory` call it
- Parameters: session deps, current_user dep, optional extras

### Task 6.2: Replace `seeded_client` with `SeededClient` dataclass

- Define `SeededClient` with `client: TestClient` and `worker_id: str`
- Update all test files: `seeded_client.post(...)` → `seeded_client.client.post(...)`, `seeded_client.seeded_worker_id` → `seeded_client.worker_id`
- **Separate commit** due to widespread mechanical change

### Task 6.3: Build `app_with_tsio` on top of `app` fixture

- In `test_router_events.py`, override only `get_tsio` on existing `app`
- Delete duplicated dependency wiring

### Task 6.4: Extract `claimed_task_id` fixture + parametrize transitions

- New fixture in `test_router_task_status.py`: submit + claim + return task ID
- Single parametrized test for all invalid transitions
- Replaces 7 copy-pasted test functions

### Task 6.5: Parametrize category tests in test_client.py

- Combine `test_job_manager_register_modifier/selection/analysis` into one `@pytest.mark.parametrize`

### Task 6.6: Small test cleanups

- Replace bare try/except with `pytest.raises` in `test_events.py`
- Delete `test_internal_task_timeout_default` (duplicate)
- Remove unused variables (`task3`, etc.)
- Delete `test_public_api_exports`, `test_all_exports_in_dunder_all`, `test_get_internal_registry_import`

---

## WS7: Missing Test Coverage

**Files:** existing + possibly new test files
**Findings:** Test Critical #1, Test Important #3,4,5,6, Test Minor #6,7,8,9,10

### Task 7.1: `run_sweeper` behavioral test

- Start sweeper as `asyncio.create_task`
- Create stale worker with old heartbeat
- Wait for sweep cycle with `asyncio.wait_for(timeout=...)`
- Verify worker cleaned up
- Cancel sweeper task

### Task 7.2: CANCELLED happy path tests

- Test PENDING → CANCELLED returns 200, `completed_at` set
- Test CLAIMED → CANCELLED returns 200, `completed_at` set

### Task 7.3: Malformed job name test

- Submit task with `"noColons"` → 404 with detail about invalid format
- Submit task with `"one:part"` → 404

### Task 7.4: Cross-room access rejection test

- Register job in `room_A`
- Submit task from `room_B` → 404 with detail about not accessible

### Task 7.5: OperationalError retry test

- Mock session to raise `OperationalError` on first claim attempt
- Verify claim succeeds on retry

### Task 7.6: Additional edge case tests

- `validate_room_id` on submit endpoint with `"bad@room"` → 422
- Submit with `null` payload and missing payload key
- `Prefer: wait=abc` → default behavior (no wait)
- `Extension()` instantiation fails (ABC)
- Enhance existing 4xx tests with `detail` field assertions

---

## Execution Notes

- **Branch:** create `fix/code-review-2026-02-09` from `main`
- **Commit strategy:** one commit per work stream, except WS6 Task 6.2 (SeededClient) gets its own commit due to widespread mechanical changes
- **Parallelization:** WS1-WS5 are sequential (shared files). WS6 and WS7 can run after src is stable. Within WS3, tasks 3.1-3.4 (extraction) before 3.5-3.7 (performance) before 3.8-3.11 (fixes).
- **Verification:** `uv run pytest` after each work stream. `uv run ruff check && uv run ruff format` before each commit.
