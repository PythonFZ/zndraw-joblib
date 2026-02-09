# Task Update Methods on JobManager

## Decision

Add `start`, `complete`, `fail`, and `cancel` methods to `JobManager` so workers
don't need raw PATCH calls to update task status.

## Design

Methods live on `JobManager` (not `ClaimedTask`) per Single Responsibility:
`ClaimedTask` stays a pure data container, `JobManager` owns all HTTP communication.

### New methods

```python
def start(self, task: ClaimedTask) -> None:
    """Transition task from CLAIMED to RUNNING."""

def complete(self, task: ClaimedTask) -> None:
    """Transition task from RUNNING to COMPLETED."""

def fail(self, task: ClaimedTask, error: str) -> None:
    """Transition task from RUNNING to FAILED with error message."""

def cancel(self, task: ClaimedTask) -> None:
    """Transition task from CLAIMED or RUNNING to CANCELLED."""
```

### Internal helper

All four delegate to a private `_update_task(task_id, status, error=None)` that
issues `PATCH /tasks/{task_id}` with `TaskUpdateRequest`-shaped JSON.

### Typical usage

```python
for task in manager.listen():
    manager.start(task)
    try:
        task.extension.run()
        manager.complete(task)
    except Exception as e:
        manager.fail(task, str(e))
```

### Return value

Methods return `None`. The server response is validated (raise_for_status) but
not exposed — callers don't need the updated task state, they just need to know
the call succeeded.
