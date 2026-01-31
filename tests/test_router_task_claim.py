# tests/test_router_task_claim.py
"""Tests for task claim endpoint using shared fixtures."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin, get_redis_client
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.schemas import TaskClaimResponse


def test_claim_task_returns_null_when_empty(seeded_client):
    response = seeded_client.post("/v1/joblib/tasks/claim")
    assert response.status_code == 200
    data = TaskClaimResponse.model_validate(response.json())
    assert data.task is None


def test_claim_task_returns_oldest_first(seeded_client):
    # Submit two tasks
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"order": 1}},
    )
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"order": 2}},
    )

    # Claim should return first (oldest)
    response = seeded_client.post("/v1/joblib/tasks/claim")
    assert response.status_code == 200
    data = TaskClaimResponse.model_validate(response.json())
    assert data.task is not None
    assert data.task.payload["order"] == 1
    assert data.task.status.value == "claimed"


def test_claim_task_marks_as_claimed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    response = seeded_client.post("/v1/joblib/tasks/claim")
    data = TaskClaimResponse.model_validate(response.json())
    assert data.task is not None
    assert data.task.status.value == "claimed"


def test_claim_task_only_registered_jobs(engine):
    """Worker can only claim tasks for jobs they are registered for."""
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    # Worker 1 registers job
    async def get_worker_1():
        return "worker_1"

    # Worker 2 tries to claim
    async def get_worker_2():
        return "worker_2"

    async def get_admin():
        return True

    async def get_mock_redis():
        class MockRedis:
            async def publish(self, channel, message):
                pass
        return MockRedis()

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProblemException, problem_exception_handler)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_is_admin] = get_admin
    app.dependency_overrides[get_redis_client] = get_mock_redis

    # Worker 1 registers
    app.dependency_overrides[get_current_identity] = get_worker_1
    client1 = TestClient(app)
    client1.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )

    # Submit task
    client1.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    # Worker 2 tries to claim (not registered)
    app.dependency_overrides[get_current_identity] = get_worker_2
    client2 = TestClient(app)
    response = client2.post("/v1/joblib/tasks/claim")
    data = TaskClaimResponse.model_validate(response.json())
    assert data.task is None  # Can't claim - not registered
