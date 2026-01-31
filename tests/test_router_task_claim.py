# tests/test_router_task_claim.py
"""Tests for task claim endpoint using shared fixtures."""
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


def test_claim_task_only_registered_jobs(client_factory):
    """Worker can only claim tasks for jobs they are registered for."""
    client1 = client_factory("worker_1")
    client2 = client_factory("worker_2")

    # Worker 1 registers job
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
    response = client2.post("/v1/joblib/tasks/claim")
    data = TaskClaimResponse.model_validate(response.json())
    assert data.task is None  # Can't claim - not registered
