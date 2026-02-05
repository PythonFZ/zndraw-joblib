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
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    assert claim_data.task is not None
    task_id = str(claim_data.task.id)

    # Update to running
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "running"

    # Verify GET returns updated status
    get_response = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert get_response.status_code == 200
    get_data = TaskResponse.model_validate(get_response.json())
    assert get_data.status.value == "running"
    assert get_data.started_at is not None


def test_update_task_running_to_completed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    assert claim_data.task is not None
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 200
    data = TaskResponse.model_validate(response.json())
    assert data.status.value == "completed"
    assert data.completed_at is not None


def test_update_task_running_to_failed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
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
    assert data.completed_at is not None


def test_update_task_invalid_transition_claimed_to_completed(seeded_client):
    """Cannot skip running and go directly to completed."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_update_task_invalid_transition_running_to_pending(seeded_client):
    """Cannot go backwards from running to pending."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "pending"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_update_task_invalid_transition_completed_to_failed(seeded_client):
    """Cannot transition from completed to failed."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "completed"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "failed"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_update_task_invalid_transition_completed_to_running(seeded_client):
    """Cannot go backwards from completed to running."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "completed"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_update_task_invalid_transition_failed_to_running(seeded_client):
    """Cannot go backwards from failed to running."""
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim", json={"worker_id": seeded_client.seeded_worker_id})
    claim_data = TaskClaimResponse.model_validate(claim_resp.json())
    task_id = str(claim_data.task.id)

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "failed"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert response.status_code == 409
    error = ProblemDetail.model_validate(response.json())
    assert error.status == 409


def test_list_tasks_for_room_empty(client):
    """List tasks for room returns empty list when no tasks exist."""
    response = client.get("/v1/joblib/rooms/my-room/tasks")
    assert response.status_code == 200
    assert response.json() == []


def test_list_tasks_for_room_returns_tasks(client_factory):
    """List tasks for room returns all tasks submitted to that room."""
    client1 = client_factory("worker-a")
    client2 = client_factory("worker-b")

    # Register jobs
    client1.put(
        "/v1/joblib/rooms/room1/jobs",
        json={"category": "modifiers", "name": "job1", "schema": {}},
    )
    client2.put(
        "/v1/joblib/rooms/room2/jobs",
        json={"category": "modifiers", "name": "job2", "schema": {}},
    )

    # Submit tasks to different rooms
    task1 = client1.post(
        "/v1/joblib/rooms/room1/tasks/room1:modifiers:job1",
        json={"payload": {"data": 1}},
    )
    task2 = client1.post(
        "/v1/joblib/rooms/room1/tasks/room1:modifiers:job1",
        json={"payload": {"data": 2}},
    )
    task3 = client2.post(
        "/v1/joblib/rooms/room2/tasks/room2:modifiers:job2",
        json={"payload": {"data": 3}},
    )

    response = client1.get("/v1/joblib/rooms/room1/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 2

    task_ids = {t["id"] for t in tasks}
    assert task1.json()["id"] in task_ids
    assert task2.json()["id"] in task_ids
    assert task3.json()["id"] not in task_ids


def test_list_tasks_for_room_with_status_filter(seeded_client):
    """List tasks for room can filter by status."""
    # Submit tasks
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 1}},
    )
    task2 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 2}},
    )

    # Cancel second task
    seeded_client.patch(
        f"/v1/joblib/tasks/{task2.json()['id']}", json={"status": "cancelled"}
    )

    # Filter for pending only
    response = seeded_client.get("/v1/joblib/rooms/room_1/tasks?status=pending")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["status"] == "pending"


def test_list_tasks_for_room_includes_queue_position(seeded_client):
    """List tasks includes queue_position for pending tasks."""
    # Submit 3 tasks
    task1 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 1}},
    )
    task2 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 2}},
    )
    task3 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 3}},
    )

    response = seeded_client.get("/v1/joblib/rooms/room_1/tasks")
    assert response.status_code == 200
    tasks = response.json()

    task_map = {t["id"]: t for t in tasks}
    assert task_map[task1.json()["id"]]["queue_position"] == 1
    assert task_map[task2.json()["id"]]["queue_position"] == 2
    assert task_map[task3.json()["id"]]["queue_position"] == 3


def test_list_tasks_for_job_empty(seeded_client):
    """List tasks for job returns empty list when no tasks exist."""
    response = seeded_client.get(
        "/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate/tasks"
    )
    assert response.status_code == 200
    assert response.json() == []


def test_list_tasks_for_job_filters_by_job(client_factory):
    """List tasks for job only returns tasks for that specific job."""
    client = client_factory("worker-a")

    # Register two jobs
    client.put(
        "/v1/joblib/rooms/room1/jobs",
        json={"category": "modifiers", "name": "job1", "schema": {}},
    )
    client.put(
        "/v1/joblib/rooms/room1/jobs",
        json={"category": "modifiers", "name": "job2", "schema": {}},
    )

    # Submit tasks to both jobs
    task1 = client.post(
        "/v1/joblib/rooms/room1/tasks/room1:modifiers:job1",
        json={"payload": {"data": 1}},
    )
    client.post(
        "/v1/joblib/rooms/room1/tasks/room1:modifiers:job2",
        json={"payload": {"data": 2}},
    )

    response = client.get("/v1/joblib/rooms/room1/jobs/room1:modifiers:job1/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task1.json()["id"]


def test_list_tasks_for_job_not_found(client):
    """List tasks for non-existent job returns 404."""
    response = client.get(
        "/v1/joblib/rooms/room1/jobs/room1:modifiers:nonexistent/tasks"
    )
    assert response.status_code == 404


def test_list_tasks_for_global_job_from_room(seeded_client):
    """Can list tasks for @global job from any room."""
    # seeded_client has @global:modifiers:Rotate registered

    # Submit task from room_1 to global job
    task1 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 1}},
    )

    response = seeded_client.get(
        "/v1/joblib/rooms/room_1/jobs/@global:modifiers:Rotate/tasks"
    )
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task1.json()["id"]


def test_get_task_includes_queue_position(seeded_client):
    """GET /tasks/{task_id} includes queue_position for pending tasks."""
    # Submit 3 tasks
    task1 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 1}},
    )
    task2 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 2}},
    )
    task3 = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 3}},
    )

    # Check queue positions
    resp1 = seeded_client.get(f"/v1/joblib/tasks/{task1.json()['id']}")
    resp2 = seeded_client.get(f"/v1/joblib/tasks/{task2.json()['id']}")
    resp3 = seeded_client.get(f"/v1/joblib/tasks/{task3.json()['id']}")

    assert resp1.json()["queue_position"] == 1
    assert resp2.json()["queue_position"] == 2
    assert resp3.json()["queue_position"] == 3


def test_get_task_queue_position_null_for_non_pending(seeded_client):
    """GET /tasks/{task_id} returns null queue_position for non-pending tasks."""
    task = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"data": 1}},
    )

    # Cancel the task
    seeded_client.patch(
        f"/v1/joblib/tasks/{task.json()['id']}", json={"status": "cancelled"}
    )

    response = seeded_client.get(f"/v1/joblib/tasks/{task.json()['id']}")
    assert response.json()["queue_position"] is None


def test_get_task_no_prefer_header_returns_immediately(seeded_client):
    """Without Prefer header, returns immediately without Preference-Applied."""
    submit_resp = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = submit_resp.json()["id"]

    response = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert response.status_code == 200
    assert "Preference-Applied" not in response.headers


def test_get_task_terminal_state_returns_immediately_despite_prefer(seeded_client):
    """Terminal state returns immediately even with Prefer header."""
    submit_resp = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = submit_resp.json()["id"]

    # Cancel the task (terminal state)
    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "cancelled"})

    response = seeded_client.get(
        f"/v1/joblib/tasks/{task_id}", headers={"Prefer": "wait=30"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert "Preference-Applied" not in response.headers


def test_get_task_with_prefer_wait_sets_preference_applied(seeded_client):
    """Long-polling sets Preference-Applied header."""
    submit_resp = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = submit_resp.json()["id"]

    # Use very short wait to avoid test timeout
    response = seeded_client.get(
        f"/v1/joblib/tasks/{task_id}", headers={"Prefer": "wait=1"}
    )
    assert response.status_code == 200
    assert response.headers.get("Preference-Applied") == "wait=1"


def test_orphan_job_soft_deleted_on_task_completion(client_factory):
    """When last task completes and no workers remain, job is soft-deleted (not hard-deleted).

    The task should still exist and the response should include the real job_name.
    """
    client = client_factory("worker-orphan")

    # 1. Register a job + worker
    reg_resp = client.put(
        "/v1/joblib/rooms/room_x/jobs",
        json={"category": "modifiers", "name": "OrphanTest", "schema": {}},
    )
    assert reg_resp.status_code == 201
    worker_id = reg_resp.json()["worker_id"]

    # 2. Submit a task
    submit_resp = client.post(
        "/v1/joblib/rooms/room_x/tasks/room_x:modifiers:OrphanTest",
        json={"payload": {"key": "value"}},
    )
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["id"]

    # 3. Delete the worker (orphan check skipped because pending task exists)
    del_resp = client.delete(f"/v1/joblib/workers/{worker_id}")
    assert del_resp.status_code == 204

    # Job should still be visible (pending task prevents soft-delete)
    job_resp = client.get("/v1/joblib/rooms/room_x/jobs/room_x:modifiers:OrphanTest")
    assert job_resp.status_code == 200

    # 4. Cancel the task (terminal state triggers orphan check in update_task_status)
    patch_resp = client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "cancelled"},
    )
    assert patch_resp.status_code == 200
    task_data = patch_resp.json()

    # Assert: task response has real job_name (not "")
    assert task_data["job_name"] == "room_x:modifiers:OrphanTest"

    # Assert: job returns 404 (soft-deleted, filtered by query)
    job_resp2 = client.get("/v1/joblib/rooms/room_x/jobs/room_x:modifiers:OrphanTest")
    assert job_resp2.status_code == 404

    # Assert: task is still retrievable (not hard-deleted)
    get_resp = client.get(f"/v1/joblib/tasks/{task_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "cancelled"


def test_parse_prefer_wait_various_formats():
    """Test Prefer header parsing."""
    from zndraw_joblib.router import parse_prefer_wait

    assert parse_prefer_wait(None) is None
    assert parse_prefer_wait("") is None
    assert parse_prefer_wait("wait=30") == 30
    assert parse_prefer_wait("wait=0") == 0
    assert parse_prefer_wait("wait=30, respond-async") == 30
    assert parse_prefer_wait("respond-async, wait=60") == 60
    assert parse_prefer_wait("respond-async") is None
