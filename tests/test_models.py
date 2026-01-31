# tests/test_models.py
import pytest
from datetime import datetime, timedelta
from uuid import UUID

from zndraw_joblib.models import (
    TaskStatus,
    WorkerJobLink,
    Job,
    Worker,
    Task,
)


def test_task_status_enum():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.CLAIMED.value == "claimed"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.CANCELLED.value == "cancelled"


def test_job_full_name():
    job = Job(room_id="@global", category="modifiers", name="Rotate", schema_={})
    assert job.full_name == "@global:modifiers:Rotate"


def test_job_full_name_private():
    job = Job(room_id="room_123", category="selections", name="All", schema_={})
    assert job.full_name == "room_123:selections:All"


def test_worker_is_alive():
    worker = Worker(id="worker_1", last_heartbeat=datetime.utcnow())
    assert worker.is_alive(timedelta(seconds=60)) is True


def test_worker_is_dead():
    old_time = datetime.utcnow() - timedelta(seconds=120)
    worker = Worker(id="worker_1", last_heartbeat=old_time)
    assert worker.is_alive(timedelta(seconds=60)) is False


def test_task_has_uuid_id():
    task = Task(job_id=UUID("12345678-1234-5678-1234-567812345678"), room_id="room_1")
    assert isinstance(task.id, UUID)


def test_task_default_status_is_pending():
    task = Task(job_id=UUID("12345678-1234-5678-1234-567812345678"), room_id="room_1")
    assert task.status == TaskStatus.PENDING
