# src/zndraw_joblib/router.py
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel as PydanticBaseModel
from sqlmodel import Session, select

from zndraw_joblib.dependencies import (
    get_db_session,
    get_current_identity,
    get_is_admin,
    get_redis_client,
    get_settings,
)
from zndraw_joblib.exceptions import (
    InvalidCategory,
    InvalidRoomId,
    SchemaConflict,
    Forbidden,
    JobNotFound,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
)
from zndraw_joblib.models import Job, Worker, WorkerJobLink, Task, TaskStatus
from zndraw_joblib.settings import JobLibSettings

router = APIRouter(prefix="/v1/joblib", tags=["joblib"])


def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global)."""
    if room_id == "@global":
        return
    if "@" in room_id or ":" in room_id:
        raise InvalidRoomId.exception(
            detail=f"Room ID '{room_id}' contains invalid characters (@ or :)"
        )
