# tests/test_router_job_list.py
"""Tests for job listing and details endpoints using shared fixtures."""
import pytest


@pytest.fixture
def multi_job_client(client):
    """Client with multiple jobs registered."""
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "selections", "name": "All", "schema": {}},
    )
    client.put(
        "/v1/joblib/rooms/room_123/jobs",
        json={"category": "modifiers", "name": "Translate", "schema": {}},
    )
    return client


def test_list_jobs_global_only(multi_job_client):
    response = multi_job_client.get("/v1/joblib/rooms/@global/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = [j["full_name"] for j in data]
    assert "@global:modifiers:Rotate" in names
    assert "@global:selections:All" in names


def test_list_jobs_room_includes_global(multi_job_client):
    response = multi_job_client.get("/v1/joblib/rooms/room_123/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3  # 2 global + 1 room
    names = [j["full_name"] for j in data]
    assert "@global:modifiers:Rotate" in names
    assert "room_123:modifiers:Translate" in names


def test_get_job_details(multi_job_client):
    response = multi_job_client.get(
        "/v1/joblib/rooms/room_123/jobs/@global:modifiers:Rotate"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["full_name"] == "@global:modifiers:Rotate"
    assert data["category"] == "modifiers"


def test_get_job_not_found(multi_job_client):
    response = multi_job_client.get(
        "/v1/joblib/rooms/room_123/jobs/@global:modifiers:NonExistent"
    )
    assert response.status_code == 404
