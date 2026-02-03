# src/zndraw_joblib/schemas.py
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from zndraw_joblib.models import TaskStatus


class JobRegisterRequest(BaseModel):
    category: str
    name: str
    schema_: dict[str, Any] = Field(default={}, alias="schema")
    worker_id: UUID | None = None  # Optional: use existing worker

    model_config = {"populate_by_name": True}


class JobResponse(BaseModel):
    id: UUID
    room_id: str
    category: str
    name: str
    full_name: str
    schema_: dict[str, Any] = Field(alias="schema")
    workers: list[UUID]
    worker_id: UUID | None = (
        None  # The worker that was created/used for this registration
    )

    model_config = {"populate_by_name": True}


class JobSummary(BaseModel):
    full_name: str
    category: str
    name: str
    workers: list[UUID]


class WorkerSummary(BaseModel):
    id: UUID
    last_heartbeat: datetime
    job_count: int


class WorkerResponse(BaseModel):
    id: UUID
    last_heartbeat: datetime


class TaskSubmitRequest(BaseModel):
    payload: dict[str, Any] = {}


class TaskResponse(BaseModel):
    id: UUID
    job_name: str
    room_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[UUID] = None
    error: Optional[str] = None
    payload: dict[str, Any] = {}
    queue_position: Optional[int] = None


class TaskClaimResponse(BaseModel):
    task: Optional[TaskResponse] = None


class TaskUpdateRequest(BaseModel):
    status: TaskStatus
    error: Optional[str] = None
