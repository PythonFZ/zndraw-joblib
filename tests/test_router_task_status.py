# tests/test_router_task_status.py
"""Tests for task status and update endpoints using shared fixtures."""
from uuid import uuid4

from zndraw_joblib.schemas import TaskResponse, TaskClaimResponse
from zndraw_joblib.exceptions import ProblemDetail


def test_get_task_status(seeded_client):
    # Submit a task
    submit_resp = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = submit_resp.json()["id"]

    response = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert str(data.id) == task_id
    assert data.status.value == "pending"


def test_get_task_not_found(seeded_client):
    fake_id = str(uuid4())
    response = seeded_client.get(f"/v1/joblib/tasks/{fake_id}")
    assert response.status_code == 404
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 404


def test_update_task_claimed_to_running(seeded_client):
    # Submit task
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    # Claim task
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    # Update to running
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "running"


def test_update_task_running_to_completed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "completed"


def test_update_task_running_to_failed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "failed", "error": "Something went wrong"},
    )
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "failed"
    assert data.error == "Something went wrong"


def test_update_task_invalid_transition(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    # Try to go from claimed directly to completed (invalid)
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409
