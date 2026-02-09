# src/zndraw_joblib/events.py
"""Socket.IO event models for real-time notifications.

All models are frozen for hashability, enabling set-based deduplication
of emissions via the Emission NamedTuple.
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from zndraw_joblib.models import TaskStatus


class JobsInvalidate(BaseModel):
    """Frontend should refetch the job list."""

    model_config = ConfigDict(frozen=True)


class TaskAvailable(BaseModel):
    """A new task is available for claiming."""

    model_config = ConfigDict(frozen=True)

    job_name: str
    room_id: str
    task_id: str


class TaskStatusEvent(BaseModel):
    """A task's status changed."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    room_id: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_position: int | None = None
    worker_id: str | None = None
    error: str | None = None


class Emission(NamedTuple):
    """Hashable (event, room) pair for set-based deduplication."""

    event: BaseModel
    room: str
