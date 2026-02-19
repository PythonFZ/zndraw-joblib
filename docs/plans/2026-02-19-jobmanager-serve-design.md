# JobManager.serve() — Unified Event Loop Design

## Problem

SIO handler registration is split across two codebases:
- `JobManager.register_provider()` registers `ProviderRequest` handler
- `ZnDraw.serve()` registers `TaskAvailable` handler

The `serve()` loop in `ZnDraw` duplicates concerns that belong in `JobManager`:
heartbeat, claim loop, signal handling, SIO event wiring.

## Design

### API

```python
manager = JobManager(
    api=api,
    tsio=tsio,
    execute=my_callback,           # optional: task execution callback
    heartbeat_interval=30.0,       # optional
    polling_interval=2.0,          # optional
)
```

- `execute: Callable[[ClaimedTask], None] | None` — host-provided callback.
  `JobManager` wraps it with lifecycle management (`start`, `complete`, `fail`).
  If `None`, no claim loop runs (provider-only worker).

### Lifecycle

1. **`__init__`** — stores config, registers SIO handlers on `tsio` (if provided):
   - `ProviderRequest` → `self._on_provider_request`
   - `TaskAvailable` → wakes claim loop via `threading.Event`

2. **First `register()` or `register_provider()`** — creates worker in DB,
   starts background daemon threads (heartbeat, claim loop). A client that
   never registers anything is not a worker and has no DB presence.

3. **`disconnect()`** — stops background threads, emits `LeaveProviderRoom` /
   `LeaveJobRoom`, deletes worker from DB, clears local state. Called
   automatically by `__exit__`.

4. **`wait()`** (optional) — blocks main thread until `disconnect()` is called.
   Installs SIGINT/SIGTERM handlers that call `disconnect()`, restores
   originals on exit. Only needed for dedicated CLI worker processes.

### Background Threads

Started lazily by `_ensure_background_threads()`, called from `_register_impl()`
and `register_provider()` after successful REST registration.

**Heartbeat thread:**
```python
def _heartbeat_loop(self) -> None:
    while not self._stop.wait(self._heartbeat_interval):
        try:
            self.heartbeat()
        except Exception as e:
            logger.warning("Heartbeat failed: %s", e)
```

**Claim loop** (only when `execute` is provided):
```python
def _claim_loop(self) -> None:
    while not self._stop.is_set():
        self._task_ready.clear()
        try:
            claimed = self.claim()
        except Exception as e:
            logger.warning("Claim failed: %s", e)
            self._task_ready.wait(timeout=self._polling_interval)
            continue
        if claimed is not None:
            self.start(claimed)
            try:
                self._execute(claimed)
            except Exception as e:
                self.fail(claimed, str(e))
            else:
                self.complete(claimed)
        else:
            self._task_ready.wait(timeout=self._polling_interval)
```

Wakes instantly on `TaskAvailable` SIO events, polls as fallback.

**ProviderRequest handler** — runs on SIO's own thread, no dedicated thread:
```python
def _on_provider_request(self, event: ProviderRequest) -> None:
    reg = self._providers.get(event.provider_name)
    if reg is None:
        return
    params = json.loads(event.params)
    instance = reg.cls(**params)
    result = instance.read(reg.handler)
    self.api.http.post(
        f"{self.api.base_url}/v1/joblib/providers/{reg.id}/results",
        headers=self.api.get_headers(),
        json={"request_hash": event.request_id, "data": result},
    )
```

### Host App Integration

`ZnDraw.serve()` simplifies to:

```python
def serve(self) -> None:
    for ext in self._extensions:
        self.jobs.register(ext)
    self.jobs.wait()  # blocks until SIGINT
```

The `_execute_task` callback:

```python
def _execute_task(self, task: ClaimedTask) -> None:
    with ZnDraw(url=self.url, room=task.room_id, ...) as task_vis:
        task.extension.run(task_vis, handlers=self.jobs.handlers)
```

### Usage Patterns

**Full worker (tasks + providers):**
```python
with JobManager(api, tsio, execute=execute_fn) as manager:
    manager.register(Rotate)
    manager.register_provider(FsProvider, name="local", handler=fs)
    manager.wait()
```

**Provider-only worker:**
```python
with JobManager(api, tsio) as manager:
    manager.register_provider(FsProvider, name="local", handler=fs)
    manager.wait()
```

**Embedded (non-blocking):**
```python
manager = JobManager(api, tsio, execute=execute_fn)
manager.register(Rotate)
# Worker is active in background. Main thread does other things.
```

### What Changes

| Component | Before | After |
|-----------|--------|-------|
| SIO handler registration | Split: `register_provider()` + `ZnDraw.serve()` | `__init__` (one place) |
| Claim loop | `ZnDraw.serve()` | `JobManager._claim_loop()` |
| Heartbeat thread | `ZnDraw.serve()` | `JobManager._heartbeat_loop()` |
| Signal handling | `ZnDraw.serve()` | `JobManager.wait()` |
| Task lifecycle (`start/complete/fail`) | `ZnDraw._execute_task()` | `JobManager._claim_loop()` |
| `listen()` generator | Public API | Kept for backward compat |

### Existing Methods Kept

- `register()`, `register_provider()`, `unregister_provider()` — unchanged
- `claim()`, `start()`, `complete()`, `fail()`, `cancel()` — unchanged (still public)
- `heartbeat()`, `submit()` — unchanged
- `listen()` — kept for backward compatibility (manual polling)
- `disconnect()` — extended to stop threads
- `handlers` property — unchanged
