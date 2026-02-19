# tests/conftest.py
"""Shared test fixtures for DRY tests."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from zndraw_auth import Base, User

from zndraw_joblib.dependencies import get_result_backend
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.router import router
from zndraw_joblib.settings import JobLibSettings

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
def test_user_id():
    """Fixed UUID for test user."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def test_user(test_user_id):
    """Create a mock User for testing."""
    user = MagicMock(spec=User)
    user.id = test_user_id
    user.email = "test@example.com"
    user.is_active = True
    user.is_superuser = True
    user.is_verified = True
    return user


@pytest.fixture
async def async_engine():
    """Create an async in-memory SQLite database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_factory(async_engine):
    """Create tables and return an async session factory."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def db_session(async_session_factory):
    """Return an async session generator for the sweeper and other non-DI uses."""

    async def get_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    return get_test_session


@pytest.fixture
def mock_current_user(test_user):
    """Factory that returns the current user dependency."""

    async def get_current_user():
        return test_user

    return get_current_user


class InMemoryResultBackend:
    """In-memory result backend for testing."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._inflight: set[str] = set()

    async def store(self, key: str, data: bytes, ttl: int) -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def acquire_inflight(self, key: str, ttl: int) -> bool:
        if key in self._inflight:
            return False
        self._inflight.add(key)
        return True

    async def release_inflight(self, key: str) -> None:
        self._inflight.discard(key)


def _build_app(
    *,
    session_maker,
    current_user,
) -> FastAPI:
    """Build a configured FastAPI app with standard dependency overrides."""
    from zndraw_auth import current_active_user, current_superuser
    from zndraw_auth.db import get_session_maker

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_session_maker] = lambda: session_maker
    app.dependency_overrides[current_active_user] = current_user
    app.dependency_overrides[current_superuser] = current_user
    app.state.joblib_settings = JobLibSettings()
    result_backend = InMemoryResultBackend()
    app.dependency_overrides[get_result_backend] = lambda: result_backend
    return app


@pytest.fixture
def app(async_session_factory, mock_current_user):
    """Create a FastAPI app with dependency overrides."""
    return _build_app(
        session_maker=async_session_factory,
        current_user=mock_current_user,
    )


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    """Client with a pre-registered job for testing."""
    resp = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    # Store the worker_id for tests that need it
    worker_id = resp.json().get("worker_id")
    client.seeded_worker_id = worker_id
    return client


@pytest.fixture
def client_factory(async_session_factory):
    """Factory to create clients with different user identities."""

    def create_client(identity: str, is_superuser: bool = True) -> TestClient:
        user_id = uuid.uuid5(uuid.NAMESPACE_DNS, identity)

        user = MagicMock(spec=User)
        user.id = user_id
        user.email = f"{identity}@example.com"
        user.is_active = True
        user.is_superuser = is_superuser
        user.is_verified = True

        async def get_current_user():
            return user

        app = _build_app(
            session_maker=async_session_factory,
            current_user=get_current_user,
        )

        test_client = TestClient(app)
        test_client.user_id = user_id
        return test_client

    return create_client


@pytest.fixture
async def async_client(async_session_factory, mock_current_user):
    """Async HTTP client with SQLite locking for concurrent stress testing."""
    db_lock = asyncio.Lock()

    @asynccontextmanager
    async def locked_session_maker():
        async with db_lock:
            async with async_session_factory() as session:
                yield session

    app = _build_app(
        session_maker=locked_session_maker,
        current_user=mock_current_user,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
