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
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from zndraw_auth import Base, User
from zndraw_joblib.router import router
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler


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
def async_engine():
    """Create an async in-memory SQLite database engine."""
    return create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
async def async_session_factory(async_engine):
    """Create tables and return an async session factory."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def test_db_lock():
    """Create a fresh lock for each test (bound to test's event loop)."""
    return asyncio.Lock()


@pytest.fixture
def db_session(async_session_factory):
    """Return an async session generator for dependency injection."""

    async def get_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    return get_test_session


@pytest.fixture
def locked_db_session(async_session_factory, test_db_lock):
    """Return a locked async session generator for dependency injection.

    This serializes all database access to prevent SQLite concurrency issues.
    """

    async def get_locked_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_db_lock:
            async with async_session_factory() as session:
                yield session

    return get_locked_test_session


@pytest.fixture
def test_session_factory(async_session_factory):
    """Return a session factory for dependency injection (used by long-poll)."""

    async def get_test_session_factory():
        @asynccontextmanager
        async def create_session():
            async with async_session_factory() as session:
                yield session

        return create_session

    return get_test_session_factory


@pytest.fixture
def mock_current_user(test_user):
    """Factory that returns the current user dependency."""

    async def get_current_user():
        return test_user

    return get_current_user


@pytest.fixture
def app(db_session, locked_db_session, test_session_factory, mock_current_user):
    """Create a FastAPI app with dependency overrides."""
    from zndraw_auth import current_active_user, current_superuser, get_async_session
    from zndraw_joblib.dependencies import get_locked_async_session, get_session_factory

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_async_session] = db_session
    app.dependency_overrides[get_locked_async_session] = locked_db_session
    app.dependency_overrides[get_session_factory] = test_session_factory
    app.dependency_overrides[current_active_user] = mock_current_user
    app.dependency_overrides[current_superuser] = mock_current_user
    return app


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
def client_factory(async_session_factory, test_db_lock):
    """Factory to create clients with different user identities."""
    from zndraw_auth import current_active_user, current_superuser, get_async_session
    from zndraw_joblib.dependencies import get_locked_async_session, get_session_factory

    def create_client(identity: str, is_superuser: bool = True) -> TestClient:
        # Create a unique user ID for each identity
        user_id = uuid.uuid5(uuid.NAMESPACE_DNS, identity)

        user = MagicMock(spec=User)
        user.id = user_id
        user.email = f"{identity}@example.com"
        user.is_active = True
        user.is_superuser = is_superuser
        user.is_verified = True

        async def get_current_user():
            return user

        async def get_test_session() -> AsyncGenerator[AsyncSession, None]:
            async with async_session_factory() as session:
                yield session

        async def get_locked_test_session() -> AsyncGenerator[AsyncSession, None]:
            async with test_db_lock:
                async with async_session_factory() as session:
                    yield session

        async def get_test_session_factory():
            @asynccontextmanager
            async def create_session():
                async with async_session_factory() as session:
                    yield session

            return create_session

        app = FastAPI()
        app.include_router(router)
        app.add_exception_handler(ProblemException, problem_exception_handler)
        app.dependency_overrides[get_async_session] = get_test_session
        app.dependency_overrides[get_locked_async_session] = get_locked_test_session
        app.dependency_overrides[get_session_factory] = get_test_session_factory
        app.dependency_overrides[current_active_user] = get_current_user
        app.dependency_overrides[current_superuser] = get_current_user

        test_client = TestClient(app)
        # Attach user info for tests that need to track workers
        test_client.user_id = user_id
        return test_client

    return create_client


@pytest.fixture
async def async_client(app):
    """Async HTTP client for concurrent stress testing."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
