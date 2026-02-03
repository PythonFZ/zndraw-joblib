# tests/test_router_jobs.py
"""Tests for job registration endpoint using shared fixtures from conftest.py."""
from zndraw_joblib.schemas import JobResponse
from zndraw_joblib.exceptions import ProblemDetail


def test_register_job_global(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0}},
    )
    assert response.status_code == 201
    data = JobResponse.model_validate(response.json())
    assert data.full_name == "@global:modifiers:Rotate"
    assert len(data.workers) == 1
    assert "test_worker_id" in data.workers


def test_register_job_private(client):
    response = client.put(
        "/v1/joblib/rooms/room_123/jobs",
        json={"category": "selections", "name": "All", "schema": {}},
    )
    assert response.status_code == 201
    data = JobResponse.model_validate(response.json())
    assert data.full_name == "room_123:selections:All"


def test_register_job_invalid_category(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "invalid_cat", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 400


def test_register_job_schema_conflict(client):
    # First registration
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0}},
    )
    # Second registration with different schema
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0, "axis": "z"}},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_register_job_same_schema_idempotent(client):
    schema = {"angle": 0}
    # First registration
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": schema},
    )
    # Second registration with same schema
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": schema},
    )
    assert response.status_code == 200  # OK, not 201
    data = JobResponse.model_validate(response.json())
    assert data.full_name == "@global:modifiers:Rotate"


def test_register_job_invalid_room_id_with_at(client):
    response = client.put(
        "/v1/joblib/rooms/room@123/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 400


def test_register_job_invalid_room_id_with_colon(client):
    response = client.put(
        "/v1/joblib/rooms/room:123/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 400


def test_register_job_returns_worker_ids(seeded_client):
    """Register job returns list of worker IDs instead of count."""
    # seeded_client already registered @global:modifiers:Rotate
    response = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    data = response.json()

    assert "workers" in data
    assert isinstance(data["workers"], list)
    assert "test_worker_id" in data["workers"]
    assert "worker_count" not in data


def test_list_jobs_returns_worker_ids(client_factory):
    """List jobs returns worker IDs for each job."""
    client1 = client_factory("worker-a")
    client2 = client_factory("worker-b")

    client1.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "TestJob", "schema": {}})
    client2.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "TestJob", "schema": {}})

    response = client1.get("/v1/joblib/rooms/@global/jobs")
    jobs = response.json()

    job = next(j for j in jobs if j["name"] == "TestJob")
    assert "workers" in job
    assert set(job["workers"]) == {"worker-a", "worker-b"}
    assert "worker_count" not in job
