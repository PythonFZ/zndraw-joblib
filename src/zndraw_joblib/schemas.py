# src/zndraw_joblib/schemas.py
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel

from zndraw_joblib.models import TaskStatus


class JobRegisterRequest(BaseModel):
    category: str
    name: str
    schema: dict[str, Any] = {}


class JobResponse(BaseModel):
    id: UUID
    room_id: str
    category: str
    name: str
    full_name: str
    schema: dict[str, Any]
    worker_count: int


class JobSummary(BaseModel):
    full_name: str
    category: str
    name: str
    worker_count: int


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
    error: Optional[str] = None
    payload: dict[str, Any] = {}


class TaskClaimResponse(BaseModel):
    task: Optional[TaskResponse] = None


class TaskUpdateRequest(BaseModel):
    status: TaskStatus
    error: Optional[str] = None
