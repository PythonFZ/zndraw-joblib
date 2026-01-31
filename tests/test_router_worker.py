# tests/test_router_worker.py
"""Tests for worker heartbeat and deletion endpoints using shared fixtures."""
from datetime import datetime

from pydantic import BaseModel


class WorkerResponse(BaseModel):
    id: str
    last_heartbeat: datetime


def test_worker_heartbeat(seeded_client):
    response = seeded_client.patch("/v1/joblib/workers/test_worker_id")
    assert response.status_code == 200
    data = WorkerResponse.model_validate(response.json())
    assert data.id == "test_worker_id"
    assert data.last_heartbeat is not None


def test_worker_heartbeat_not_found(client):
    response = client.patch("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404


def test_worker_delete(seeded_client):
    response = seeded_client.delete("/v1/joblib/workers/test_worker_id")
    assert response.status_code == 204


def test_worker_delete_not_found(client):
    response = client.delete("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404


def test_worker_delete_removes_links(seeded_client):
    # Verify worker-job link exists by deleting worker
    seeded_client.delete("/v1/joblib/workers/test_worker_id")

    # Register again - should create new worker and link
    response = seeded_client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 200  # Idempotent, job exists
    assert response.json()["worker_count"] == 1  # New link created
