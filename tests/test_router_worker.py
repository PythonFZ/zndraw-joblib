# tests/test_router_worker.py
"""Tests for worker heartbeat and deletion endpoints using shared fixtures."""
import time
from datetime import datetime
from pydantic import BaseModel

from zndraw_joblib.schemas import JobResponse, TaskResponse
from zndraw_joblib.exceptions import ProblemDetail


class WorkerResponse(BaseModel):
    id: str
    last_heartbeat: datetime


def test_worker_heartbeat_updates_timestamp(seeded_client):
    """Heartbeat should update timestamp, second call should have later timestamp."""
    response1 = seeded_client.patch("/v1/joblib/workers/test_worker_id")
    assert response1.status_code == 200
    data1 = WorkerResponse.model_validate(response1.json())
    assert data1.id == "test_worker_id"
    assert data1.last_heartbeat is not None

    time.sleep(0.01)  # Small delay to ensure timestamp differs

    response2 = seeded_client.patch("/v1/joblib/workers/test_worker_id")
    assert response2.status_code == 200
    data2 = WorkerResponse.model_validate(response2.json())

    assert data2.last_heartbeat > data1.last_heartbeat


def test_worker_heartbeat_not_found(client):
    response = client.patch("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 404


def test_worker_delete(seeded_client):
    response = seeded_client.delete("/v1/joblib/workers/test_worker_id")
    assert response.status_code == 204


def test_worker_delete_not_found(client):
    response = client.delete("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 404


def test_worker_delete_removes_orphan_job(seeded_client):
    """Deleting sole worker of a job should remove the job too."""
    # seeded_client already registered @global:modifiers:Rotate with test_worker_id

    seeded_client.delete("/v1/joblib/workers/test_worker_id")
    response = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert response.status_code == 404


def test_worker_delete_keeps_job_with_pending_task(seeded_client):
    """Job should remain if there are non-terminal tasks, even without workers."""
    # Submit a task (creates a pending task)
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    seeded_client.delete("/v1/joblib/workers/test_worker_id")

    # Job should still exist because there's a pending task
    response = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert response.status_code == 200
    data = JobResponse.model_validate(response.json())
    assert data.full_name == "@global:modifiers:Rotate"
    assert data.worker_count == 0


def test_worker_delete_removes_job_after_task_completes(seeded_client):
    """Job should be removed when sole worker deleted and all tasks are terminal."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "completed"})

    seeded_client.delete("/v1/joblib/workers/test_worker_id")

    # Job should be removed (no workers, no pending tasks)
    response = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert response.status_code == 404


def test_worker_count_changes_with_workers(client_factory):
    """worker_count should increase when workers register and decrease when removed."""
    client1 = client_factory("worker_1")
    client2 = client_factory("worker_2")
    client3 = client_factory("worker_3")

    # Worker 1 registers
    resp = client1.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code == 201
    data = JobResponse.model_validate(resp.json())
    assert data.worker_count == 1

    # Worker 2 registers same job
    resp = client2.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert resp.status_code == 200  # Idempotent
    data = JobResponse.model_validate(resp.json())
    assert data.worker_count == 2

    # Worker 3 registers same job
    resp = client3.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    data = JobResponse.model_validate(resp.json())
    assert data.worker_count == 3

    # Remove worker 2
    client2.delete("/v1/joblib/workers/worker_2")

    # Check worker count is now 2
    resp = client1.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    data = JobResponse.model_validate(resp.json())
    assert data.worker_count == 2

    # Remove worker 1
    client1.delete("/v1/joblib/workers/worker_1")

    # Check worker count is now 1
    resp = client3.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    data = JobResponse.model_validate(resp.json())
    assert data.worker_count == 1


def test_worker_delete_fails_running_tasks(seeded_client):
    """Deleting worker should mark their running/claimed tasks as failed."""
    # Submit a task
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    # Claim and start running
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})

    # Delete worker - task should be marked as failed
    seeded_client.delete("/v1/joblib/workers/test_worker_id")

    # Task should still exist and be marked as failed
    task_resp = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert task_resp.status_code == 200
    task_data = TaskResponse.model_validate(task_resp.json())
    assert task_data.status.value == "failed"
    assert "worker disconnected" in task_data.error.lower()

    # Job should be soft-deleted (returns 404 on GET)
    resp = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert resp.status_code == 404


def test_worker_delete_soft_deletes_job_but_keeps_task(seeded_client):
    """Job should be soft-deleted but task remains accessible."""
    # Submit and complete a task
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"test": "data"}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "completed"})

    # Delete sole worker - job becomes orphan
    seeded_client.delete("/v1/joblib/workers/test_worker_id")

    # Job should return 404 (soft-deleted)
    resp = seeded_client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert resp.status_code == 404

    # But task should still be accessible with all its data
    task_resp = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert task_resp.status_code == 200
    task_data = TaskResponse.model_validate(task_resp.json())
    assert task_data.status.value == "completed"
    assert task_data.payload == {"test": "data"}
    # Job name should still be available from the soft-deleted job
    assert task_data.job_name == "@global:modifiers:Rotate"


def test_list_workers_empty(client):
    """List workers returns empty list when no workers exist."""
    response = client.get("/v1/joblib/workers")
    assert response.status_code == 200
    assert response.json() == []


def test_list_workers_returns_all(client_factory):
    """List workers returns all workers with job counts."""
    client1 = client_factory("worker-a")
    client2 = client_factory("worker-b")

    # Worker A registers two jobs
    client1.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "job1", "schema": {}})
    client1.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "job2", "schema": {}})

    # Worker B registers one job
    client2.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "job3", "schema": {}})

    response = client1.get("/v1/joblib/workers")
    assert response.status_code == 200
    workers = response.json()
    assert len(workers) == 2

    worker_map = {w["id"]: w for w in workers}
    assert worker_map["worker-a"]["job_count"] == 2
    assert worker_map["worker-b"]["job_count"] == 1


def test_list_workers_for_room_empty(client):
    """List workers for room returns empty list when no workers."""
    response = client.get("/v1/joblib/rooms/my-room/workers")
    assert response.status_code == 200
    assert response.json() == []


def test_list_workers_for_room_filters_by_room(client_factory):
    """List workers for room only returns workers serving that room."""
    client_a = client_factory("worker-a")
    client_b = client_factory("worker-b")
    client_c = client_factory("worker-c")

    # Worker A serves room1 and @global
    client_a.put("/v1/joblib/rooms/room1/jobs", json={"category": "modifiers", "name": "job1", "schema": {}})
    client_a.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "global-job", "schema": {}})

    # Worker B serves room2
    client_b.put("/v1/joblib/rooms/room2/jobs", json={"category": "modifiers", "name": "job2", "schema": {}})

    # Worker C serves room1
    client_c.put("/v1/joblib/rooms/room1/jobs", json={"category": "selections", "name": "job3", "schema": {}})

    # List workers for room1 - should include A (has room1 job + @global) and C
    response = client_a.get("/v1/joblib/rooms/room1/workers")
    assert response.status_code == 200
    workers = response.json()
    worker_ids = {w["id"] for w in workers}

    assert "worker-a" in worker_ids
    assert "worker-c" in worker_ids
    assert "worker-b" not in worker_ids  # Only serves room2


def test_list_workers_for_global_room(client_factory):
    """List workers for @global room only returns workers serving @global jobs."""
    client_a = client_factory("worker-a")
    client_b = client_factory("worker-b")

    client_a.put("/v1/joblib/rooms/@global/jobs", json={"category": "modifiers", "name": "global-job", "schema": {}})
    client_b.put("/v1/joblib/rooms/room1/jobs", json={"category": "modifiers", "name": "job1", "schema": {}})

    response = client_a.get("/v1/joblib/rooms/@global/workers")
    assert response.status_code == 200
    workers = response.json()
    worker_ids = {w["id"] for w in workers}

    assert "worker-a" in worker_ids
    assert "worker-b" not in worker_ids
