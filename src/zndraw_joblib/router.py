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
from zndraw_joblib.schemas import JobRegisterRequest, JobResponse
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


@router.put(
    "/rooms/{room_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_job(
    room_id: str,
    request: JobRegisterRequest,
    response: Response,
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
    is_admin: bool = Depends(get_is_admin),
    settings: JobLibSettings = Depends(get_settings),
):
    """Register a job for a room. Creates worker and link if not exists."""
    # Validate room_id
    validate_room_id(room_id)

    # Check admin for @global
    if room_id == "@global" and not is_admin:
        raise Forbidden.exception(detail="Admin required for @global job registration")

    # Validate category
    if request.category not in settings.allowed_categories:
        raise InvalidCategory.exception(
            detail=f"Category '{request.category}' not in allowed list: {settings.allowed_categories}"
        )

    # Check if job exists
    existing_job = db.exec(
        select(Job).where(
            Job.room_id == room_id,
            Job.category == request.category,
            Job.name == request.name,
        )
    ).first()

    if existing_job:
        # Validate schema match
        if existing_job.schema_ != request.schema:
            raise SchemaConflict.exception(
                detail=f"Schema mismatch for job '{existing_job.full_name}'"
            )
        # Idempotent - ensure worker link exists
        job = existing_job
        response.status_code = status.HTTP_200_OK
    else:
        # Create new job
        job = Job(
            room_id=room_id,
            category=request.category,
            name=request.name,
            schema_=request.schema,
        )
        db.add(job)
        db.flush()

    # Ensure worker exists
    worker = db.exec(select(Worker).where(Worker.id == identity)).first()
    if not worker:
        worker = Worker(id=identity)
        db.add(worker)
        db.flush()

    # Ensure worker-job link exists
    link = db.exec(
        select(WorkerJobLink).where(
            WorkerJobLink.worker_id == identity,
            WorkerJobLink.job_id == job.id,
        )
    ).first()
    if not link:
        link = WorkerJobLink(worker_id=identity, job_id=job.id)
        db.add(link)

    db.commit()
    db.refresh(job)

    # Count workers for this job
    worker_count = len(
        db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
    )

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        worker_count=worker_count,
    )
