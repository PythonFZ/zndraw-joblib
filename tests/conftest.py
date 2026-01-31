# tests/conftest.py
"""Shared test fixtures for DRY tests."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import (
    get_db_session,
    get_current_identity,
    get_is_admin,
    get_redis_client,
)
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


class MockRedis:
    """Mock Redis client for tests."""

    async def publish(self, channel: str, message: str) -> None:
        pass

    async def subscribe(self, channel: str) -> None:
        pass


@pytest.fixture
def engine():
    """Create an in-memory SQLite database engine."""
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def db_session(engine):
    """Create tables and return a session generator."""
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    return get_test_session


@pytest.fixture
def mock_identity():
    """Default identity for tests."""
    return "test_worker_id"


@pytest.fixture
def mock_identity_factory(mock_identity):
    """Factory that returns the identity dependency."""

    async def get_test_identity():
        return mock_identity

    return get_test_identity


@pytest.fixture
def mock_is_admin():
    """Default admin status for tests."""

    async def get_test_is_admin():
        return True

    return get_test_is_admin


@pytest.fixture
def mock_redis():
    """Mock Redis client."""

    async def get_mock_redis():
        return MockRedis()

    return get_mock_redis


@pytest.fixture
def app(db_session, mock_identity_factory, mock_is_admin, mock_redis):
    """Create a FastAPI app with dependency overrides."""
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_db_session] = db_session
    app.dependency_overrides[get_current_identity] = mock_identity_factory
    app.dependency_overrides[get_is_admin] = mock_is_admin
    app.dependency_overrides[get_redis_client] = mock_redis
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    """Client with a pre-registered job for testing."""
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    return client


@pytest.fixture
def client_factory(db_session, mock_is_admin, mock_redis):
    """Factory to create clients with different identities."""

    def create_client(identity: str) -> TestClient:
        async def get_identity():
            return identity

        app = FastAPI()
        app.include_router(router)
        app.add_exception_handler(ProblemException, problem_exception_handler)
        app.dependency_overrides[get_db_session] = db_session
        app.dependency_overrides[get_current_identity] = get_identity
        app.dependency_overrides[get_is_admin] = mock_is_admin
        app.dependency_overrides[get_redis_client] = mock_redis
        return TestClient(app)

    return create_client
