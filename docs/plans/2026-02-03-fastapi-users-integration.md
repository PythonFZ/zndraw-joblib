# FastAPI-Users Integration for zndraw-joblib

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace zndraw-joblib's stub dependencies with direct fastapi-users dependencies, enabling shared auth between zndraw-fastapi and zndraw-joblib.

**Architecture:** Remove the `get_current_identity` and `get_is_admin` stubs. Instead, define typed dependencies that expect fastapi-users' `User` model pattern. The host app provides configured `current_active_user` and `current_superuser` via `dependency_overrides`. For `get_db_session`, adopt fastapi-users' `get_async_session` pattern.

**Tech Stack:** fastapi-users, SQLModel, FastAPI dependency injection, pytest

---

## Summary of Changes

| Current (stub pattern) | New (fastapi-users pattern) |
|------------------------|----------------------------|
| `get_db_session() -> AsyncSession` | `get_async_session() -> AsyncSession` (same, rename for consistency) |
| `get_current_identity() -> str` | `current_active_user() -> User` then use `user.id` |
| `get_is_admin() -> bool` | `current_superuser() -> User` (raises 403 if not admin) |

---

## Task 1: Add fastapi-users Dependency

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/pyproject.toml`

**Step 1: Add fastapi-users to dependencies**

Edit `pyproject.toml` dependencies section:

```toml
dependencies = [
    "fastapi>=0.128.0",
    "fastapi-users[sqlalchemy]>=14.0.0",
    "httpx>=0.28.1",
    "pydantic>=2.12.5",
    "pydantic-settings>=2.12.0",
    "pydantic-socketio @ git+https://github.com/zincware/pydantic-socketio.git@main",
    "sqlmodel>=0.0.31",
]
```

**Step 2: Run uv sync to install**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv sync`
Expected: fastapi-users installed successfully

**Step 3: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi-users dependency"
```

---

## Task 2: Create User Protocol for Type Safety

**Files:**
- Create: `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/protocols.py`
- Test: `/Users/fzills/tools/zndraw-joblib/tests/test_protocols.py`

**Step 1: Write the test for UserProtocol**

Create `/Users/fzills/tools/zndraw-joblib/tests/test_protocols.py`:

```python
"""Tests for protocol definitions."""

import uuid
from typing import runtime_checkable

import pytest
from pydantic import BaseModel

from zndraw_joblib.protocols import UserProtocol


class MockUser(BaseModel):
    """Mock user matching UserProtocol."""

    id: uuid.UUID
    email: str
    is_active: bool
    is_superuser: bool


def test_mock_user_satisfies_protocol():
    """Verify our mock user satisfies UserProtocol."""
    user = MockUser(
        id=uuid.uuid4(),
        email="test@example.com",
        is_active=True,
        is_superuser=False,
    )
    # Protocol check at runtime
    assert isinstance(user, UserProtocol)


def test_protocol_provides_id():
    """Verify protocol exposes id attribute."""
    user = MockUser(
        id=uuid.uuid4(),
        email="test@example.com",
        is_active=True,
        is_superuser=False,
    )
    assert hasattr(user, "id")
    assert isinstance(user.id, uuid.UUID)


def test_protocol_provides_is_superuser():
    """Verify protocol exposes is_superuser attribute."""
    user = MockUser(
        id=uuid.uuid4(),
        email="admin@example.com",
        is_active=True,
        is_superuser=True,
    )
    assert hasattr(user, "is_superuser")
    assert user.is_superuser is True
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_protocols.py -v`
Expected: FAIL with "No module named 'zndraw_joblib.protocols'"

**Step 3: Create the protocols module**

Create `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/protocols.py`:

```python
"""Protocol definitions for dependency injection."""

import uuid
from typing import Protocol, runtime_checkable


@runtime_checkable
class UserProtocol(Protocol):
    """Protocol for user objects from fastapi-users.

    This protocol defines the minimal interface that zndraw-joblib
    expects from a user object. Any fastapi-users User model will
    satisfy this protocol.
    """

    id: uuid.UUID
    is_superuser: bool
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_protocols.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add src/zndraw_joblib/protocols.py tests/test_protocols.py
git commit -m "feat: add UserProtocol for fastapi-users compatibility"
```

---

## Task 3: Rewrite Dependencies Module

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/dependencies.py`
- Modify: `/Users/fzills/tools/zndraw-joblib/tests/test_dependencies.py`

**Step 1: Update test_dependencies.py with new expectations**

Replace `/Users/fzills/tools/zndraw-joblib/tests/test_dependencies.py`:

```python
"""Tests for dependency stubs."""

import uuid

import pytest
from pydantic import BaseModel

from zndraw_joblib.dependencies import (
    get_async_session,
    get_current_user,
    get_current_superuser,
    get_settings,
)
from zndraw_joblib.protocols import UserProtocol
from zndraw_joblib.settings import JobLibSettings


@pytest.mark.anyio
async def test_get_async_session_raises_not_implemented():
    """Stub should raise NotImplementedError until overridden."""
    with pytest.raises(NotImplementedError, match="not configured"):
        async for _ in get_async_session():
            pass


@pytest.mark.anyio
async def test_get_current_user_raises_not_implemented():
    """Stub should raise NotImplementedError until overridden."""
    with pytest.raises(NotImplementedError, match="not configured"):
        await get_current_user()


@pytest.mark.anyio
async def test_get_current_superuser_raises_not_implemented():
    """Stub should raise NotImplementedError until overridden."""
    with pytest.raises(NotImplementedError, match="not configured"):
        await get_current_superuser()


def test_get_settings_returns_settings():
    """Settings dependency should return JobLibSettings instance."""
    settings = get_settings()
    assert isinstance(settings, JobLibSettings)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py -v`
Expected: FAIL with import errors (old names don't exist)

**Step 3: Rewrite dependencies.py**

Replace `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/dependencies.py`:

```python
"""Dependency stubs for FastAPI dependency injection.

These stubs are designed to be overridden by the host application
using `app.dependency_overrides`. The host app should provide
fastapi-users configured dependencies.

Example host app setup:
    from fastapi_users import FastAPIUsers
    from zndraw_joblib import get_async_session, get_current_user, get_current_superuser

    # Configure fastapi-users
    fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

    # Override zndraw-joblib dependencies
    app.dependency_overrides[get_async_session] = get_async_session_impl
    app.dependency_overrides[get_current_user] = fastapi_users.current_user(active=True)
    app.dependency_overrides[get_current_superuser] = fastapi_users.current_user(active=True, superuser=True)
"""

from functools import lru_cache
from typing import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession

from zndraw_joblib.protocols import UserProtocol
from zndraw_joblib.settings import JobLibSettings


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Stub: Override with your async session generator.

    Expected to yield an AsyncSession from SQLAlchemy/SQLModel.
    This matches the fastapi-users pattern for database access.
    """
    raise NotImplementedError("Dependency 'get_async_session' not configured")
    yield  # type: ignore[misc]


async def get_current_user() -> UserProtocol:
    """Stub: Override with fastapi_users.current_user(active=True).

    Expected to return a User object satisfying UserProtocol.
    The user must have `id: uuid.UUID` and `is_superuser: bool` attributes.
    """
    raise NotImplementedError("Dependency 'get_current_user' not configured")


async def get_current_superuser() -> UserProtocol:
    """Stub: Override with fastapi_users.current_user(active=True, superuser=True).

    Expected to return a User object satisfying UserProtocol.
    FastAPI-users will raise 403 Forbidden if user is not a superuser.
    """
    raise NotImplementedError("Dependency 'get_current_superuser' not configured")


@lru_cache
def get_settings() -> JobLibSettings:
    """Returns cached JobLibSettings instance.

    This is NOT a stub - it always returns real settings.
    Settings are loaded from environment variables with ZNDRAW_JOBLIB_ prefix.
    """
    return JobLibSettings()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add src/zndraw_joblib/dependencies.py tests/test_dependencies.py
git commit -m "refactor: rename dependencies to match fastapi-users patterns

- get_db_session -> get_async_session
- get_current_identity -> get_current_user (returns UserProtocol)
- get_is_admin -> get_current_superuser (returns UserProtocol)"
```

---

## Task 4: Update Router to Use New Dependencies

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/router.py`

**Step 1: Update imports in router.py**

Change lines 11-16 from:
```python
from zndraw_joblib.dependencies import (
    get_current_identity,
    get_db_session,
    get_is_admin,
    get_settings,
)
```

To:
```python
from zndraw_joblib.dependencies import (
    get_async_session,
    get_current_superuser,
    get_current_user,
    get_settings,
)
from zndraw_joblib.protocols import UserProtocol
```

**Step 2: Create type aliases for cleaner signatures**

Add after imports (around line 30):

```python
# Type aliases for dependency injection
from typing import Annotated

SessionDep = Annotated[AsyncSession, Depends(get_async_session)]
CurrentUserDep = Annotated[UserProtocol, Depends(get_current_user)]
SuperuserDep = Annotated[UserProtocol, Depends(get_current_superuser)]
SettingsDep = Annotated[JobLibSettings, Depends(get_settings)]
```

**Step 3: Update register_job endpoint (lines 82-174)**

Change the function signature from:
```python
async def register_job(
    room_id: str,
    body: JobRegisterRequest,
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
    is_admin: bool = Depends(get_is_admin),
    settings: JobLibSettings = Depends(get_settings),
) -> JobResponse:
```

To:
```python
async def register_job(
    room_id: str,
    body: JobRegisterRequest,
    db: SessionDep,
    user: CurrentUserDep,
    settings: SettingsDep,
) -> JobResponse:
```

Then update the admin check (around line 101) from:
```python
if room_id == "@global" and not is_admin:
    raise Forbidden.exception("Admin privileges required for global jobs")
```

To:
```python
if room_id == "@global" and not user.is_superuser:
    raise Forbidden.exception("Admin privileges required for global jobs")
```

And update identity usage (worker creation) - change `identity` to `str(user.id)`:
```python
# Where identity was used, now use str(user.id)
worker = Worker(id=str(user.id))
```

**Step 4: Update list_jobs endpoint (lines 177-211)**

Change signature from:
```python
async def list_jobs(
    room_id: str,
    db: Session = Depends(get_db_session),
) -> JobListResponse:
```

To:
```python
async def list_jobs(
    room_id: str,
    db: SessionDep,
) -> JobListResponse:
```

**Step 5: Update all remaining endpoints**

Apply the same pattern to all endpoints:

| Endpoint | Old Signature | New Signature |
|----------|---------------|---------------|
| `list_workers_for_room` | `db: Session = Depends(get_db_session)` | `db: SessionDep` |
| `list_tasks_for_room` | `db: Session = Depends(get_db_session)` | `db: SessionDep` |
| `list_tasks_for_job` | `db: Session = Depends(get_db_session)` | `db: SessionDep` |
| `get_job` | `db: Session = Depends(get_db_session)` | `db: SessionDep` |
| `submit_task` | `db, identity` | `db: SessionDep, user: CurrentUserDep` then use `str(user.id)` |
| `claim_task` | `db, identity` | `db: SessionDep, user: CurrentUserDep` then use `str(user.id)` |
| `get_task_status` | `db, settings` | `db: SessionDep, settings: SettingsDep` |
| `update_task_status` | `db` | `db: SessionDep` |
| `list_workers` | `db` | `db: SessionDep` |
| `worker_heartbeat` | `db` | `db: SessionDep` |
| `delete_worker` | `db` | `db: SessionDep` |

**Step 6: Verify no old imports remain**

Run: `cd /Users/fzills/tools/zndraw-joblib && grep -n "get_db_session\|get_current_identity\|get_is_admin" src/zndraw_joblib/router.py`
Expected: No matches

**Step 7: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add src/zndraw_joblib/router.py
git commit -m "refactor: update router to use fastapi-users dependency pattern

- Use Annotated type aliases for cleaner signatures
- Replace get_db_session with get_async_session
- Replace get_current_identity with get_current_user (use user.id)
- Replace get_is_admin with user.is_superuser check"
```

---

## Task 5: Update Test Fixtures

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/tests/conftest.py`

**Step 1: Update imports**

Change imports from:
```python
from zndraw_joblib.dependencies import (
    get_current_identity,
    get_db_session,
    get_is_admin,
)
```

To:
```python
from zndraw_joblib.dependencies import (
    get_async_session,
    get_current_superuser,
    get_current_user,
)
from zndraw_joblib.protocols import UserProtocol
```

**Step 2: Create MockUser class**

Add after imports:

```python
import uuid
from pydantic import BaseModel


class MockUser(BaseModel):
    """Mock user for testing, satisfies UserProtocol."""

    id: uuid.UUID
    email: str
    is_active: bool = True
    is_superuser: bool = False

    class Config:
        frozen = True
```

**Step 3: Update mock_identity fixture to mock_user**

Replace `mock_identity` fixture:
```python
@pytest.fixture
def mock_user() -> MockUser:
    """Default test user."""
    return MockUser(
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        email="test@example.com",
        is_active=True,
        is_superuser=True,  # Default to superuser for most tests
    )
```

**Step 4: Update mock_identity_factory to mock_user_factory**

Replace:
```python
@pytest.fixture
def mock_user_factory(mock_user: MockUser):
    """Factory for get_current_user override."""
    async def get_test_user() -> UserProtocol:
        return mock_user
    return get_test_user
```

**Step 5: Update mock_is_admin to mock_superuser_factory**

Replace:
```python
@pytest.fixture
def mock_superuser_factory(mock_user: MockUser):
    """Factory for get_current_superuser override."""
    async def get_test_superuser() -> UserProtocol:
        if not mock_user.is_superuser:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Not a superuser")
        return mock_user
    return get_test_superuser
```

**Step 6: Update app fixture**

Change from:
```python
@pytest.fixture
def app(db_session, mock_identity_factory, mock_is_admin):
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_db_session] = db_session
    app.dependency_overrides[get_current_identity] = mock_identity_factory
    app.dependency_overrides[get_is_admin] = mock_is_admin
    return app
```

To:
```python
@pytest.fixture
def app(db_session, mock_user_factory, mock_superuser_factory):
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_async_session] = db_session
    app.dependency_overrides[get_current_user] = mock_user_factory
    app.dependency_overrides[get_current_superuser] = mock_superuser_factory
    return app
```

**Step 7: Update client_factory fixture**

Change from creating identity string to creating MockUser:
```python
@pytest.fixture
def client_factory(db_session, mock_superuser_factory):
    """Factory to create test clients with different user identities."""
    def create_client(
        user_id: uuid.UUID | str | None = None,
        is_superuser: bool = True,
    ) -> TestClient:
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id) if user_id else uuid.uuid4()
        elif user_id is None:
            user_id = uuid.uuid4()

        user = MockUser(
            id=user_id,
            email=f"{user_id}@test.com",
            is_active=True,
            is_superuser=is_superuser,
        )

        async def get_user() -> UserProtocol:
            return user

        async def get_superuser() -> UserProtocol:
            if not user.is_superuser:
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail="Not a superuser")
            return user

        app = FastAPI()
        app.include_router(router)
        app.add_exception_handler(ProblemException, problem_exception_handler)
        app.dependency_overrides[get_async_session] = db_session
        app.dependency_overrides[get_current_user] = get_user
        app.dependency_overrides[get_current_superuser] = get_superuser
        return TestClient(app)

    return create_client
```

**Step 8: Run tests to check for failures**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/ -v --tb=short`
Expected: Some tests may fail due to identity string vs UUID changes

**Step 9: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add tests/conftest.py
git commit -m "refactor: update test fixtures for fastapi-users pattern

- Replace mock_identity with mock_user (MockUser class)
- Replace mock_is_admin with mock_superuser_factory
- Update client_factory to create users with UUID ids
- Update dependency overrides to new names"
```

---

## Task 6: Fix Failing Tests - Identity Changes

**Files:**
- Modify: Various test files that expect string identity

**Step 1: Identify failing tests**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/ -v 2>&1 | grep -E "FAILED|ERROR"`

**Step 2: Update tests that compare identity strings**

Tests that previously used `"test_worker_id"` now need to use the UUID from mock_user.

In test files, change assertions like:
```python
assert response.json()["created_by_id"] == "test_worker_id"
```

To:
```python
assert response.json()["created_by_id"] == "12345678-1234-5678-1234-567812345678"
```

Or better, use a fixture:
```python
def test_submit_task_stores_creator(seeded_client, mock_user):
    response = seeded_client.post(...)
    assert response.json()["created_by_id"] == str(mock_user.id)
```

**Step 3: Update client_factory usage in tests**

Tests using `client_factory("worker_1")` need to change to:
```python
client_factory(user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"))
```

Or for simpler cases:
```python
client_factory()  # Gets random UUID
```

**Step 4: Run all tests**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add tests/
git commit -m "fix: update tests for UUID-based user identity"
```

---

## Task 7: Update Public Exports

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/src/zndraw_joblib/__init__.py`

**Step 1: Update __all__ exports**

Change the dependency exports from:
```python
"get_current_identity",
"get_db_session",
"get_is_admin",
```

To:
```python
"get_async_session",
"get_current_superuser",
"get_current_user",
"UserProtocol",
```

**Step 2: Update imports**

Change:
```python
from zndraw_joblib.dependencies import (
    get_current_identity,
    get_db_session,
    get_is_admin,
    get_settings,
)
```

To:
```python
from zndraw_joblib.dependencies import (
    get_async_session,
    get_current_superuser,
    get_current_user,
    get_settings,
)
from zndraw_joblib.protocols import UserProtocol
```

**Step 3: Update test_init.py if it exists**

Check for and update any tests that verify exports.

**Step 4: Run tests**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_init.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add src/zndraw_joblib/__init__.py tests/test_init.py
git commit -m "refactor: update public API exports for fastapi-users pattern"
```

---

## Task 8: Update README Documentation

**Files:**
- Modify: `/Users/fzills/tools/zndraw-joblib/README.md`

**Step 1: Update the integration example**

Replace the old example showing stub overrides with the new fastapi-users pattern:

```markdown
## Integration with FastAPI-Users

```python
from fastapi import FastAPI
from fastapi_users import FastAPIUsers

from zndraw_joblib import (
    router,
    get_async_session,
    get_current_user,
    get_current_superuser,
    problem_exception_handler,
    ProblemException,
)

# Your fastapi-users setup
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

app = FastAPI()
app.include_router(router)
app.add_exception_handler(ProblemException, problem_exception_handler)

# Wire up dependencies
app.dependency_overrides[get_async_session] = get_async_session_impl
app.dependency_overrides[get_current_user] = fastapi_users.current_user(active=True)
app.dependency_overrides[get_current_superuser] = fastapi_users.current_user(active=True, superuser=True)
```
```

**Step 2: Commit**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add README.md
git commit -m "docs: update README with fastapi-users integration example"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/ -v --tb=long`
Expected: All tests PASS

**Step 2: Run type checking (if configured)**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run mypy src/zndraw_joblib/`
Expected: No errors (or only pre-existing ones)

**Step 3: Run linting (if configured)**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run ruff check src/zndraw_joblib/`
Expected: No errors

**Step 4: Create final commit if any fixes needed**

```bash
cd /Users/fzills/tools/zndraw-joblib
git add -A
git commit -m "chore: final cleanup for fastapi-users integration"
```

---

## Summary of Breaking Changes

For consumers of zndraw-joblib, the following changes are required:

| Old | New | Notes |
|-----|-----|-------|
| `get_db_session` | `get_async_session` | Same signature, just renamed |
| `get_current_identity` | `get_current_user` | Returns `UserProtocol` instead of `str`. Use `str(user.id)` for identity string. |
| `get_is_admin` | `get_current_superuser` | Returns `UserProtocol`. FastAPI-users raises 403 if not superuser. |

The host application (zndraw-fastapi) should provide:
```python
app.dependency_overrides[get_async_session] = your_session_generator
app.dependency_overrides[get_current_user] = fastapi_users.current_user(active=True)
app.dependency_overrides[get_current_superuser] = fastapi_users.current_user(active=True, superuser=True)
```
