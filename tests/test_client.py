# tests/test_client.py
"""Integration tests for the client SDK against the actual server."""

from typing import ClassVar

from zndraw_joblib.client import (
    JobManager,
    Extension,
    Category,
)
from zndraw_joblib.schemas import JobSummary, JobResponse, TaskResponse, PaginatedResponse


class MockClientApi:
    """Adapter to make TestClient work with JobManager's ApiManager protocol."""

    def __init__(self, test_client, identity: str = "test_worker"):
        self._client = test_client
        self._identity = identity

    @property
    def http(self):
        return self._client

    @property
    def base_url(self) -> str:
        return ""  # TestClient doesn't need base_url

    def get_headers(self) -> dict[str, str]:
        return {}


def test_category_enum():
    """Category enum should have correct values."""
    assert Category.MODIFIER.value == "modifiers"
    assert Category.SELECTION.value == "selections"
    assert Category.ANALYSIS.value == "analysis"


def test_extension_requires_category():
    """Extension subclasses must define category."""

    class ValidExtension(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        value: int = 0

    assert ValidExtension.category == Category.MODIFIER


def test_job_manager_register_modifier(client):
    """JobManager.register should create job with MODIFIER category."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    assert "@global:modifiers:Rotate" in manager

    response = client.get("/v1/joblib/rooms/@global/jobs")
    assert response.status_code == 200
    page = PaginatedResponse[JobSummary].model_validate(response.json())
    job_names = [j.full_name for j in page.items]
    assert "@global:modifiers:Rotate" in job_names


def test_job_manager_register_selection(client):
    """JobManager.register should create job with SELECTION category."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class SelectAll(Extension):
        category: ClassVar[Category] = Category.SELECTION

    assert "@global:selections:SelectAll" in manager

    response = client.get("/v1/joblib/rooms/@global/jobs")
    assert response.status_code == 200
    page = PaginatedResponse[JobSummary].model_validate(response.json())
    job_names = [j.full_name for j in page.items]
    assert "@global:selections:SelectAll" in job_names


def test_job_manager_register_analysis(client):
    """JobManager.register should create job with ANALYSIS category."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class Measure(Extension):
        category: ClassVar[Category] = Category.ANALYSIS
        property_name: str = "distance"

    assert "@global:analysis:Measure" in manager

    response = client.get("/v1/joblib/rooms/@global/jobs")
    assert response.status_code == 200
    page = PaginatedResponse[JobSummary].model_validate(response.json())
    job_names = [j.full_name for j in page.items]
    assert "@global:analysis:Measure" in job_names


def test_job_manager_register_with_room(client):
    """JobManager.register(room=...) should create room-specific job."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register(room="my_room")
    class PrivateJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        data: str = ""

    assert "my_room:modifiers:PrivateJob" in manager

    response = client.get("/v1/joblib/rooms/my_room/jobs")
    assert response.status_code == 200
    page = PaginatedResponse[JobSummary].model_validate(response.json())
    job_names = [j.full_name for j in page.items]
    assert "my_room:modifiers:PrivateJob" in job_names


def test_job_manager_getitem_returns_class(client):
    """JobManager[full_name] should return the registered class."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class MyJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        x: int = 0

    assert manager["@global:modifiers:MyJob"] is MyJob


def test_job_manager_len(client):
    """len(JobManager) should return number of registered jobs."""
    api = MockClientApi(client)
    manager = JobManager(api)

    assert len(manager) == 0

    @manager.register
    class Job1(Extension):
        category: ClassVar[Category] = Category.MODIFIER

    assert len(manager) == 1

    @manager.register
    class Job2(Extension):
        category: ClassVar[Category] = Category.SELECTION

    assert len(manager) == 2


def test_job_manager_iter(client):
    """iter(JobManager) should iterate over registered job names."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class JobA(Extension):
        category: ClassVar[Category] = Category.MODIFIER

    @manager.register
    class JobB(Extension):
        category: ClassVar[Category] = Category.MODIFIER

    names = list(manager)
    assert "@global:modifiers:JobA" in names
    assert "@global:modifiers:JobB" in names


def test_job_manager_schema_sent_to_server(client):
    """JobManager should send the Pydantic schema to the server."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0
        axis: str = "z"

    response = client.get("/v1/joblib/rooms/@global/jobs/@global:modifiers:Rotate")
    assert response.status_code == 200
    job = JobResponse.model_validate(response.json())

    assert "properties" in job.schema_
    assert "angle" in job.schema_["properties"]
    assert "axis" in job.schema_["properties"]


def test_job_manager_listen_yields_extension_instance(client):
    """JobManager.listen() should yield Extension instances with payload data."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class Rotate(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        angle: float = 0.0

    # Submit a task
    submit_resp = client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"angle": 45.0}},
    )
    assert submit_resp.status_code == 202
    _ = TaskResponse.model_validate(submit_resp.json())

    # Listen should yield the Extension instance
    for claimed in manager.listen():
        assert isinstance(claimed.extension, Rotate)
        assert claimed.extension.angle == 45.0
        assert claimed.task_id is not None
        assert claimed.room_id == "room_1"
        break  # Only get one task


def test_job_manager_listen_returns_none_when_empty(client):
    """JobManager.listen() with timeout should return when no tasks available."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class EmptyJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER

    # No tasks submitted - claim should return None
    claimed = manager.claim()
    assert claimed is None


def test_job_manager_claim_until_empty(client):
    """Calling claim repeatedly should return None when no more tasks."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class BatchJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        index: int = 0

    # Submit 3 tasks
    for i in range(3):
        resp = client.post(
            "/v1/joblib/rooms/room_1/tasks/@global:modifiers:BatchJob",
            json={"payload": {"index": i}},
        )
        assert resp.status_code == 202

    # Claim all 3
    claimed_indices = []
    for _ in range(3):
        claimed = manager.claim()
        assert claimed is not None
        assert isinstance(claimed.extension, BatchJob)
        claimed_indices.append(claimed.extension.index)

    # Verify we got all 3 (order may vary due to FIFO)
    assert sorted(claimed_indices) == [0, 1, 2]

    # Fourth claim should return None
    assert manager.claim() is None


def test_job_manager_heartbeat(client):
    """JobManager.heartbeat() should update worker timestamp."""

    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class HeartbeatJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER

    # The worker_id was set during register - use that
    assert manager.worker_id is not None

    # First heartbeat
    manager.heartbeat()

    # Verify heartbeat was recorded by calling again
    response = client.patch(f"/v1/joblib/workers/{manager.worker_id}")
    assert response.status_code == 200


def test_job_manager_complete_workflow(client):
    """Test complete workflow: register, submit, claim, complete, verify empty."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class ProcessData(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        input_file: str = ""
        output_file: str = ""

    # 1. Submit two tasks
    for i in range(2):
        resp = client.post(
            "/v1/joblib/rooms/room_1/tasks/@global:modifiers:ProcessData",
            json={
                "payload": {"input_file": f"in{i}.txt", "output_file": f"out{i}.txt"}
            },
        )
        task = TaskResponse.model_validate(resp.json())
        assert task.status.value == "pending"

    # 2. Claim first task
    claimed1 = manager.claim()
    assert claimed1 is not None
    assert isinstance(claimed1.extension, ProcessData)

    # 3. Update to running and complete
    client.patch(f"/v1/joblib/tasks/{claimed1.task_id}", json={"status": "running"})
    complete_resp = client.patch(
        f"/v1/joblib/tasks/{claimed1.task_id}",
        json={"status": "completed"},
    )
    completed = TaskResponse.model_validate(complete_resp.json())
    assert completed.status.value == "completed"

    # 4. Claim second task
    claimed2 = manager.claim()
    assert claimed2 is not None
    assert isinstance(claimed2.extension, ProcessData)

    # 5. Complete second task
    client.patch(f"/v1/joblib/tasks/{claimed2.task_id}", json={"status": "running"})
    client.patch(f"/v1/joblib/tasks/{claimed2.task_id}", json={"status": "completed"})

    # 6. No more tasks - claim should return None
    assert manager.claim() is None

    # TODO 7. validate finished tasks have completed_at


def test_job_manager_claimed_task_has_metadata(client):
    """ClaimedTask should include task_id, room_id, job_name, and extension."""
    api = MockClientApi(client)
    manager = JobManager(api)

    @manager.register
    class MetadataJob(Extension):
        category: ClassVar[Category] = Category.MODIFIER
        value: int = 42

    client.post(
        "/v1/joblib/rooms/test_room/tasks/@global:modifiers:MetadataJob",
        json={"payload": {"value": 99}},
    )

    claimed = manager.claim()
    assert claimed is not None
    assert claimed.task_id is not None
    assert claimed.room_id == "test_room"
    assert claimed.job_name == "@global:modifiers:MetadataJob"
    assert isinstance(claimed.extension, MetadataJob)
    assert claimed.extension.value == 99
