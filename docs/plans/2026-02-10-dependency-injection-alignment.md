# zndraw-joblib: Dependency Injection Alignment

**Date**: 2026-02-10
**Status**: Design
**Scope**: Align with zndraw-auth pure DI, remove redundant wrappers

---

## Goal

Maximum DRY: Use zndraw-auth's session dependencies directly, no wrappers.

**Remove:**
- `get_async_session_maker()` wrapper (redundant)
- `get_locked_async_session()` (locking is host app concern)
- `get_session_factory()` (just use session_maker directly)

**Keep:**
- Models (inherit from `Base`)
- Routes (use `SessionDep` from auth)
- Business logic

---

## Architectural Role

**zndraw-joblib is a feature library:**
- Provides job/task/worker management
- **Does NOT** provide session dependencies (uses auth's)
- **Does NOT** provide database initialization (host app does)
- **Does NOT** handle locking (host app does based on DB type)

**Why no session wrappers?**
- DRY violation: Duplicate session logic
- Locking is DB concern, not joblib concern
- Auth already provides what we need

---

## Dependency Usage

### Before (Wrapper Pattern)

```python
# zndraw_joblib/dependencies.py
def get_async_session_maker(...) -> async_sessionmaker:
    return get_session_maker(...)  # Wrapper

async def get_locked_async_session(...) -> AsyncSession:
    # Duplicate session logic + locking
    ...

def get_session_factory(...) -> Callable:
    # Wrapper around session_maker
    ...

# routes.py
@router.post("/jobs")
async def create_job(session: Annotated[AsyncSession, Depends(get_locked_async_session)]):
    ...
```

**Problems:**
- ❌ DRY violation (duplicate session yielding)
- ❌ Locking in wrong place (should be host app concern)
- ❌ Extra boilerplate

---

### After (Direct Usage - Maximum DRY)

```python
# zndraw_joblib/dependencies.py
# No session dependencies at all!

# routes.py
from zndraw_auth.db import SessionDep

@router.post("/jobs")
async def create_job(session: SessionDep):
    # Uses auth's dependency directly
    job = Job(...)
    session.add(job)
    await session.commit()
```

**Benefits:**
- ✅ DRY: Session logic written once (in auth)
- ✅ Clear: Direct dependency on auth
- ✅ Simple: Less code, less maintenance

---

## Locking Strategy

**Old approach (Wrong):**
```python
# Locking in joblib
async def get_locked_async_session(...):
    if enable_db_lock:
        async with db_lock:
            yield session
```

**New approach (Correct):**
```python
# Locking in host app (zndraw-fastapi)
# In lifespan - wrap session_maker if SQLite
if settings.database_url.startswith("sqlite"):
    base_maker = async_sessionmaker(engine)

    @asynccontextmanager
    async def locked_maker():
        async with app.state.db_lock:
            async with base_maker() as session:
                yield session

    app.state.session_maker = locked_maker
```

**Why this is better:**
- Host app knows database type
- Host app decides locking strategy
- Joblib stays database-agnostic
- No settings.enable_db_lock needed in joblib

---

## Long-Polling Pattern

**Old approach:**
```python
def get_session_factory(session_maker):
    @asynccontextmanager
    async def create_session():
        async with session_maker() as session:
            yield session
    return create_session
```

**New approach:**
```python
# Just use session_maker directly!
from zndraw_auth.db import get_session_maker

@router.get("/tasks/poll")
async def poll_tasks(
    session_maker: Annotated[async_sessionmaker, Depends(get_session_maker)]
):
    # Create multiple sessions in loop
    while True:
        async with session_maker() as session:
            tasks = await session.execute(...)
            if tasks:
                return tasks
        await asyncio.sleep(1)
```

**Why simpler:**
- No wrapper function needed
- session_maker IS the factory
- Direct, clear intent

---

## Models (No Changes)

```python
# zndraw_joblib/models.py
from zndraw_auth import Base

class Job(Base):
    __tablename__ = "job"
    # All Job fields...

class Worker(Base):
    __tablename__ = "worker"
    # All Worker fields...

class Task(Base):
    __tablename__ = "task"
    # All Task fields...
```

**Status:** Already correct! Models inherit from `Base`, share metadata.

---

## Routes (Simplified)

### Before
```python
from zndraw_joblib.dependencies import get_locked_async_session

LockedSessionDep = Annotated[AsyncSession, Depends(get_locked_async_session)]

@router.post("/jobs")
async def create_job(session: LockedSessionDep):
    ...
```

### After
```python
from zndraw_auth.db import SessionDep

@router.post("/jobs")
async def create_job(session: SessionDep):
    # Locking handled by host app (if needed)
    ...
```

**Key:** Routes don't need to know about locking. Host app handles it.

---

## What This Package Provides

### Public API

```python
# Router (with all endpoints)
from zndraw_joblib import router

# Models (for type hints)
from zndraw_joblib.models import Job, Task, Worker, WorkerJobLink

# Exceptions
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler

# Settings (for configuration)
from zndraw_joblib.settings import JobLibSettings, get_settings

# Internal registry (for TaskIQ integration)
from zndraw_joblib.registry import InternalRegistry, register_internal_jobs

# Sweeper (for worker cleanup)
from zndraw_joblib.sweeper import run_sweeper

# Socket.IO events
from zndraw_joblib.events import JoinJobRoom, LeaveJobRoom
```

**No session dependencies!** Uses auth's directly.

---

## Usage

### Host Application Integration

```python
from fastapi import FastAPI
from zndraw_joblib import router
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler

app = FastAPI()

# Register exception handler and router
app.add_exception_handler(ProblemException, problem_exception_handler)
app.include_router(router)

# Initialize db_lock (for SQLite if needed)
app.state.db_lock = asyncio.Lock()
```

---

## Testing

### Before (Override joblib dependency)
```python
app.dependency_overrides[get_async_session_maker] = lambda: test_maker
```

### After (Override auth dependency)
```python
from zndraw_auth.db import get_session_maker

app.dependency_overrides[get_session_maker] = lambda: test_maker
```

**Simpler:** One override point at auth level, everything flows from there.

**For SQLite stress tests:**
```python
# Create locked session maker for testing
base_maker = async_sessionmaker(test_engine)
db_lock = asyncio.Lock()

@asynccontextmanager
async def locked_test_maker():
    async with db_lock:
        async with base_maker() as session:
            yield session

# Override with locked version
app.dependency_overrides[get_session_maker] = lambda: locked_test_maker
```

---

## Changes Required

### Remove
- `get_async_session_maker()` function
- `get_locked_async_session()` function
- `get_session_factory()` function
- `enable_db_lock` setting
- `db_lock_timeout_seconds` setting
- `get_db_lock()` dependency

### Modify
- All routes: Import `SessionDep` from `zndraw_auth.db`
- Long-polling endpoints: Use `session_maker` directly via `get_session_maker`
- Tests: Override at auth level, not joblib level

### Keep Unchanged
- All models
- All business logic
- Router structure
- Exception handling
- Settings (except locking settings)
- Internal registry
- Sweeper

---

## Package Coordination

### With zndraw-auth
- Imports `SessionDep` from `zndraw_auth.db`
- Imports `get_session_maker` for long-polling
- Models inherit from `Base` (shared metadata)
- No wrappers, maximum DRY
- See: `/Users/fzills/tools/zndraw-auth/docs/plans/2026-02-10-pure-di-architecture.md`

### With zndraw-fastapi
- Fastapi includes joblib router
- Fastapi initializes joblib tables (via shared metadata)
- Fastapi handles locking (if needed for SQLite)
- Fastapi provides `db_lock` in app.state (for internal use)
- See: `/Users/fzills/tools/zndraw-fastapi/docs/plans/2026-02-10-database-initialization.md`

---

## Migration Notes

**Breaking Changes:**
- `get_async_session_maker` removed - use `get_session_maker` from auth
- `get_locked_async_session` removed - locking now in host app
- `get_session_factory` removed - use `get_session_maker` directly
- Settings for locking removed - host app decides locking

**Benefits:**
- Cleaner code (less boilerplate)
- Better separation of concerns
- Maximum DRY
- Easier testing

---

## Success Criteria

- [ ] No session dependency functions in joblib
- [ ] All routes use `SessionDep` from auth
- [ ] Long-polling uses `get_session_maker` from auth directly
- [ ] No locking logic in joblib
- [ ] Tests override at auth level
- [ ] All tests pass with new pattern
- [ ] Documentation updated
- [ ] README reflects direct auth dependency usage
