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
from zndraw_joblib.schemas import (
    JobRegisterRequest,
    JobResponse,
    JobSummary,
    TaskSubmitRequest,
    TaskResponse,
    TaskClaimResponse,
)
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


@router.get("/rooms/{room_id}/jobs", response_model=list[JobSummary])
async def list_jobs(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    """List jobs for a room. Includes @global jobs unless room_id is @global."""
    validate_room_id(room_id)

    if room_id == "@global":
        jobs = db.exec(select(Job).where(Job.room_id == "@global")).all()
    else:
        jobs = db.exec(
            select(Job).where((Job.room_id == "@global") | (Job.room_id == room_id))
        ).all()

    result = []
    for job in jobs:
        worker_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
        )
        result.append(
            JobSummary(
                full_name=job.full_name,
                category=job.category,
                name=job.name,
                worker_count=worker_count,
            )
        )
    return result


@router.get("/rooms/{room_id}/jobs/{job_name:path}", response_model=JobResponse)
async def get_job(
    room_id: str,
    job_name: str,
    db: Session = Depends(get_db_session),
):
    """Get job details by full name."""
    validate_room_id(room_id)

    # Parse job_name: room_id:category:name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # For room requests, allow access to both @global and room-specific jobs
    if room_id != "@global" and job_room_id not in ("@global", room_id):
        raise JobNotFound.exception(detail=f"Job '{job_name}' not accessible from room '{room_id}'")

    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

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


@router.post(
    "/rooms/{room_id}/tasks/{job_name:path}",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task(
    room_id: str,
    job_name: str,
    request: TaskSubmitRequest,
    response: Response,
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
):
    """Submit a task for processing."""
    validate_room_id(room_id)

    # Parse job_name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # Find the job
    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    # For private jobs, ensure room matches
    if job_room_id != "@global" and job_room_id != room_id:
        raise JobNotFound.exception(
            detail=f"Job '{job_name}' not accessible from room '{room_id}'"
        )

    # Create task
    task = Task(
        job_id=job.id,
        room_id=room_id,
        created_by_id=identity,
        payload=request.payload,
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Set Location header
    response.headers["Location"] = f"/v1/joblib/tasks/{task.id}"
    response.headers["Retry-After"] = "1"

    return TaskResponse(
        id=task.id,
        job_name=job.full_name,
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        payload=task.payload,
    )


@router.post("/tasks/claim", response_model=TaskClaimResponse)
async def claim_task(
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
):
    """Claim the oldest pending task for jobs the worker is registered for."""
    # Find oldest pending task for jobs this worker is registered for
    # Using subquery to get job IDs worker is registered for
    worker_job_ids = db.exec(
        select(WorkerJobLink.job_id).where(WorkerJobLink.worker_id == identity)
    ).all()

    if not worker_job_ids:
        return TaskClaimResponse(task=None)

    # Find oldest pending task
    task = db.exec(
        select(Task)
        .where(Task.job_id.in_(worker_job_ids), Task.status == TaskStatus.PENDING)
        .order_by(Task.created_at.asc())
        .limit(1)
    ).first()

    if not task:
        return TaskClaimResponse(task=None)

    # Claim the task
    task.status = TaskStatus.CLAIMED
    task.worker_id = identity
    db.add(task)
    db.commit()
    db.refresh(task)

    # Get job for full name
    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    return TaskClaimResponse(
        task=TaskResponse(
            id=task.id,
            job_name=job.full_name if job else "",
            room_id=task.room_id,
            status=task.status,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            error=task.error,
            payload=task.payload,
        )
    )
