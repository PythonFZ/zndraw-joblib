# src/zndraw_joblib/router.py
import asyncio
import re
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel as PydanticBaseModel
from sqlmodel import Session, select

from zndraw_joblib.dependencies import (
    get_db_session,
    get_current_identity,
    get_is_admin,
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
from typing import Optional

from zndraw_joblib.schemas import (
    JobRegisterRequest,
    JobResponse,
    JobSummary,
    TaskSubmitRequest,
    TaskResponse,
    TaskClaimResponse,
    TaskSummary,
    TaskUpdateRequest,
    WorkerSummary,
)
from zndraw_joblib.settings import JobLibSettings

# Valid status transitions
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.CLAIMED, TaskStatus.CANCELLED},
    TaskStatus.CLAIMED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}

TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


def parse_prefer_wait(prefer_header: str | None) -> int | None:
    """
    Parse RFC 7240 Prefer header for wait directive.
    Returns seconds to wait, or None if not specified.
    """
    if not prefer_header:
        return None
    match = re.search(r'\bwait=(\d+)\b', prefer_header)
    if match:
        return int(match.group(1))
    return None


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

    # Get worker IDs for this job
    worker_links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    ).all()
    worker_ids = [link.worker_id for link in worker_links]

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        workers=worker_ids,
    )


@router.get("/rooms/{room_id}/jobs", response_model=list[JobSummary])
async def list_jobs(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    """List jobs for a room. Includes @global jobs unless room_id is @global."""
    validate_room_id(room_id)

    if room_id == "@global":
        jobs = db.exec(
            select(Job).where(Job.room_id == "@global", Job.deleted == False)
        ).all()
    else:
        jobs = db.exec(
            select(Job).where(
                (Job.room_id == "@global") | (Job.room_id == room_id),
                Job.deleted == False,
            )
        ).all()

    result = []
    for job in jobs:
        worker_links = db.exec(
            select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
        ).all()
        worker_ids = [link.worker_id for link in worker_links]
        result.append(
            JobSummary(
                full_name=job.full_name,
                category=job.category,
                name=job.name,
                workers=worker_ids,
            )
        )
    return result


@router.get("/rooms/{room_id}/workers", response_model=list[WorkerSummary])
async def list_workers_for_room(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    """List workers serving jobs in a room. Includes @global workers unless room_id is @global."""
    validate_room_id(room_id)

    # Find jobs for this room (and @global if not requesting @global specifically)
    if room_id == "@global":
        jobs = db.exec(
            select(Job).where(Job.room_id == "@global", Job.deleted == False)
        ).all()
    else:
        jobs = db.exec(
            select(Job).where(
                (Job.room_id == "@global") | (Job.room_id == room_id),
                Job.deleted == False,
            )
        ).all()

    job_ids = [job.id for job in jobs]
    if not job_ids:
        return []

    # Find workers linked to these jobs
    worker_ids = db.exec(
        select(WorkerJobLink.worker_id).where(WorkerJobLink.job_id.in_(job_ids)).distinct()
    ).all()

    if not worker_ids:
        return []

    workers = db.exec(select(Worker).where(Worker.id.in_(worker_ids))).all()
    result = []
    for worker in workers:
        job_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)).all()
        )
        result.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return result


@router.get("/rooms/{room_id}/tasks", response_model=list[TaskSummary])
async def list_tasks_for_room(
    room_id: str,
    status: Optional[TaskStatus] = None,
    db: Session = Depends(get_db_session),
):
    """List tasks for a room, optionally filtered by status. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    query = select(Task).where(Task.room_id == room_id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    tasks = db.exec(query).all()

    result = []
    for task in tasks:
        job = db.exec(select(Job).where(Job.id == task.job_id)).first()

        queue_position = None
        if task.status == TaskStatus.PENDING:
            count = db.exec(
                select(Task).where(
                    Task.job_id == task.job_id,
                    Task.status == TaskStatus.PENDING,
                    Task.created_at < task.created_at,
                )
            ).all()
            queue_position = len(count) + 1

        result.append(
            TaskSummary(
                id=task.id,
                job_name=job.full_name if job else "",
                status=task.status,
                created_at=task.created_at,
                queue_position=queue_position,
            )
        )
    return result


@router.get("/rooms/{room_id}/jobs/{job_name:path}/tasks", response_model=list[TaskSummary])
async def list_tasks_for_job(
    room_id: str,
    job_name: str,
    status: Optional[TaskStatus] = None,
    db: Session = Depends(get_db_session),
):
    """List tasks for a specific job. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    # Parse job_name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # Validate access (same logic as get_job)
    if room_id != "@global" and job_room_id not in ("@global", room_id):
        raise JobNotFound.exception(detail=f"Job '{job_name}' not accessible from room '{room_id}'")

    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job or job.deleted:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    query = select(Task).where(Task.job_id == job.id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    tasks = db.exec(query).all()

    result = []
    for task in tasks:
        queue_position = None
        if task.status == TaskStatus.PENDING:
            count = db.exec(
                select(Task).where(
                    Task.job_id == task.job_id,
                    Task.status == TaskStatus.PENDING,
                    Task.created_at < task.created_at,
                )
            ).all()
            queue_position = len(count) + 1

        result.append(
            TaskSummary(
                id=task.id,
                job_name=job.full_name,
                status=task.status,
                created_at=task.created_at,
                queue_position=queue_position,
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

    if not job or job.deleted:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    worker_links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    ).all()
    worker_ids = [link.worker_id for link in worker_links]

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        workers=worker_ids,
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

    if not job or job.deleted:
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

    # Calculate queue position
    count = db.exec(
        select(Task).where(
            Task.job_id == job.id,
            Task.status == TaskStatus.PENDING,
            Task.created_at < task.created_at,
        )
    ).all()
    queue_position = len(count) + 1

    return TaskResponse(
        id=task.id,
        job_name=job.full_name,
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        payload=task.payload,
        queue_position=queue_position,
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
            queue_position=None,  # Claimed tasks are not in queue
        )
    )


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: UUID,
    db: Session = Depends(get_db_session),
):
    """Get task status."""
    task = db.exec(select(Task).where(Task.id == task_id)).first()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    # Calculate queue position for pending tasks
    queue_position = None
    if task.status == TaskStatus.PENDING:
        count = db.exec(
            select(Task).where(
                Task.job_id == task.job_id,
                Task.status == TaskStatus.PENDING,
                Task.created_at < task.created_at,
            )
        ).all()
        queue_position = len(count) + 1

    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        payload=task.payload,
        queue_position=queue_position,
    )


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    request: TaskUpdateRequest,
    db: Session = Depends(get_db_session),
):
    """Update task status."""
    task = db.exec(select(Task).where(Task.id == task_id)).first()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    # Validate transition
    if request.status not in VALID_TRANSITIONS.get(task.status, set()):
        raise InvalidTaskTransition.exception(
            detail=f"Cannot transition from '{task.status.value}' to '{request.status.value}'"
        )

    # Update status
    task.status = request.status
    now = datetime.now(timezone.utc)

    if request.status == TaskStatus.RUNNING:
        task.started_at = now
    elif request.status in TERMINAL_STATES:
        task.completed_at = now
        if request.error:
            task.error = request.error

    db.add(task)
    db.commit()
    db.refresh(task)

    # Check for orphan job cleanup (no workers and no non-terminal tasks)
    if request.status in TERMINAL_STATES:
        job_id = task.job_id
        remaining_workers = db.exec(
            select(WorkerJobLink).where(WorkerJobLink.job_id == job_id)
        ).first()

        if not remaining_workers:
            # No workers - check if any non-terminal tasks remain
            non_terminal_statuses = {TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.RUNNING}
            non_terminal_task = db.exec(
                select(Task).where(
                    Task.job_id == job_id,
                    Task.status.in_(non_terminal_statuses),
                )
            ).first()

            if not non_terminal_task:
                # Job is orphan - delete all tasks then the job
                all_tasks = db.exec(select(Task).where(Task.job_id == job_id)).all()
                for t in all_tasks:
                    db.delete(t)

                job_to_delete = db.exec(select(Job).where(Job.id == job_id)).first()
                if job_to_delete:
                    db.delete(job_to_delete)
                db.commit()

                # Return response for the deleted task
                return TaskResponse(
                    id=task.id,
                    job_name="",
                    room_id=task.room_id,
                    status=task.status,
                    created_at=task.created_at,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    error=task.error,
                    payload=task.payload,
                    queue_position=None,  # Terminal state, not in queue
                )

    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    # Calculate queue position if task is still pending
    queue_position = None
    if task.status == TaskStatus.PENDING:
        count = db.exec(
            select(Task).where(
                Task.job_id == task.job_id,
                Task.status == TaskStatus.PENDING,
                Task.created_at < task.created_at,
            )
        ).all()
        queue_position = len(count) + 1

    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        payload=task.payload,
        queue_position=queue_position,
    )


class WorkerResponse(PydanticBaseModel):
    id: str
    last_heartbeat: datetime


@router.get("/workers", response_model=list[WorkerSummary])
async def list_workers(
    db: Session = Depends(get_db_session),
):
    """List all workers with their job counts."""
    workers = db.exec(select(Worker)).all()
    result = []
    for worker in workers:
        job_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)).all()
        )
        result.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return result


@router.patch("/workers/{worker_id}", response_model=WorkerResponse)
async def worker_heartbeat(
    worker_id: str,
    db: Session = Depends(get_db_session),
):
    """Update worker heartbeat timestamp."""
    worker = db.exec(select(Worker).where(Worker.id == worker_id)).first()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    worker.last_heartbeat = datetime.utcnow()
    db.add(worker)
    db.commit()
    db.refresh(worker)

    return WorkerResponse(id=worker.id, last_heartbeat=worker.last_heartbeat)


@router.delete("/workers/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_worker(
    worker_id: str,
    db: Session = Depends(get_db_session),
):
    """Delete worker, fail their tasks, remove job links, and clean up orphan jobs."""
    worker = db.exec(select(Worker).where(Worker.id == worker_id)).first()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    # Fail any claimed/running tasks owned by this worker
    worker_tasks = db.exec(
        select(Task).where(
            Task.worker_id == worker_id,
            Task.status.in_({TaskStatus.CLAIMED, TaskStatus.RUNNING}),
        )
    ).all()
    now = datetime.now(timezone.utc)
    for task in worker_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Worker disconnected"
        db.add(task)

    # Get job IDs this worker is linked to before deleting links
    links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.worker_id == worker_id)
    ).all()
    job_ids = [link.job_id for link in links]

    # Delete all links
    for link in links:
        db.delete(link)

    # Delete worker
    db.delete(worker)
    db.flush()

    # Clean up orphan jobs (no workers and no non-terminal tasks)
    # Note: Tasks are kept as historical records even when job is deleted
    non_terminal_statuses = {TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.RUNNING}
    for job_id in job_ids:
        # Check if job has any remaining workers
        remaining_workers = db.exec(
            select(WorkerJobLink).where(WorkerJobLink.job_id == job_id)
        ).first()
        if remaining_workers:
            continue  # Job still has workers

        # Check if job has any non-terminal tasks
        non_terminal_task = db.exec(
            select(Task).where(
                Task.job_id == job_id,
                Task.status.in_(non_terminal_statuses),
            )
        ).first()
        if non_terminal_task:
            continue  # Job has pending/running tasks

        # Job is orphan - soft delete (tasks remain as historical records)
        job = db.exec(select(Job).where(Job.id == job_id)).first()
        if job:
            job.deleted = True
            db.add(job)

    db.commit()
