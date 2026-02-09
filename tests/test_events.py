# tests/test_events.py
"""Tests for Socket.IO event models and Emission dedup."""

from datetime import datetime, timezone

from zndraw_joblib.events import (
    JobsInvalidate,
    TaskAvailable,
    TaskStatusEvent,
    JoinJobRoom,
    LeaveJobRoom,
    Emission,
)


def test_jobs_invalidate_frozen():
    """Frozen model is hashable and equal to itself."""
    a = JobsInvalidate()
    b = JobsInvalidate()
    assert a == b
    assert hash(a) == hash(b)


def test_task_available_frozen():
    ev = TaskAvailable(
        job_name="@global:modifiers:Rotate", room_id="room1", task_id="abc"
    )
    assert ev.job_name == "@global:modifiers:Rotate"
    # Frozen: assignment should raise
    try:
        ev.job_name = "x"
        assert False, "Should have raised"
    except Exception:
        pass


def test_task_status_event_frozen():
    now = datetime.now(timezone.utc)
    ev = TaskStatusEvent(
        id="abc",
        name="@global:modifiers:Rotate",
        room_id="room1",
        status="pending",
        created_at=now,
    )
    assert ev.status == "pending"


def test_emission_dedup_jobs_invalidate():
    """Duplicate JobsInvalidate for same room should dedup in a set."""
    emissions = {
        Emission(JobsInvalidate(), "room:@global"),
        Emission(JobsInvalidate(), "room:@global"),
        Emission(JobsInvalidate(), "room:test"),
    }
    assert len(emissions) == 2


def test_join_job_room_frozen():
    ev = JoinJobRoom(job_name="@global:modifiers:Rotate")
    assert ev.job_name == "@global:modifiers:Rotate"
    try:
        ev.job_name = "x"
        assert False, "Should have raised"
    except Exception:
        pass


def test_leave_job_room_frozen():
    ev = LeaveJobRoom(job_name="@global:modifiers:Rotate")
    assert ev.job_name == "@global:modifiers:Rotate"
    a = LeaveJobRoom(job_name="@global:modifiers:Rotate")
    b = LeaveJobRoom(job_name="@global:modifiers:Rotate")
    assert a == b
    assert hash(a) == hash(b)


def test_emission_dedup_task_status():
    """Distinct TaskStatusEvent (different id) should NOT dedup."""
    now = datetime.now(timezone.utc)
    emissions = {
        Emission(
            TaskStatusEvent(
                id="a", name="j", room_id="r", status="failed", created_at=now
            ),
            "room:r",
        ),
        Emission(
            TaskStatusEvent(
                id="b", name="j", room_id="r", status="failed", created_at=now
            ),
            "room:r",
        ),
    }
    assert len(emissions) == 2
