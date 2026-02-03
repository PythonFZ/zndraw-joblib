# tests/test_router_task_submit.py
"""Tests for task submission endpoint using shared fixtures."""

from zndraw_joblib.schemas import TaskResponse
from zndraw_joblib.exceptions import ProblemDetail


def test_submit_task(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/room_123/tasks/@global:modifiers:Rotate",
        json={"payload": {"angle": 90}},
    )
    assert response.status_code == 202
    assert "Location" in response.headers
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "pending"
    assert data.payload == {"angle": 90}


def test_submit_task_job_not_found(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/room_123/tasks/@global:modifiers:NonExistent",
        json={"payload": {}},
    )
    assert response.status_code == 404
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 404


def test_submit_task_sets_room_id(seeded_client):
    # TODO: How is this different from test_submit_task?
    response = seeded_client.post(
        "/v1/joblib/rooms/my_room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    assert response.status_code == 202
    data = TaskResponse.model_validate(response.json())
    assert data.room_id == "my_room"
