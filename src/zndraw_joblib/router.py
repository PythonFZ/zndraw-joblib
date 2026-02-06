# src/zndraw_joblib/router.py
import asyncio
import random
import re
from datetime import datetime, timezone
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Response, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from zndraw_auth import User, current_active_user, current_superuser, get_async_session

from zndraw_joblib.dependencies import get_settings, get_locked_async_session
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
from zndraw_joblib.models import Job, Worker, WorkerJobLink, Task, TaskStatus, TERMINAL_STATUSES
from zndraw_joblib.sweeper import _cleanup_worker, _soft_delete_orphan_job
from zndraw_joblib.schemas import (
    JobRegisterRequest,
    JobResponse,
    JobSummary,
    TaskSubmitRequest,
    TaskResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskUpdateRequest,
    WorkerSummary,
    WorkerResponse,
)
from zndraw_joblib.settings import JobLibSettings

# Type aliases for dependency injection
CurrentUserDep = Annotated[User, Depends(current_active_user)]
SuperUserDep = Annotated[User, Depends(current_superuser)]
SessionDep = Annotated[AsyncSession, Depends(get_async_session)]
LockedSessionDep = Annotated[AsyncSession, Depends(get_locked_async_session)]
SettingsDep = Annotated[JobLibSettings, Depends(get_settings)]

# Valid status transitions
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.CLAIMED, TaskStatus.CANCELLED},
    TaskStatus.CLAIMED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}



def parse_prefer_wait(prefer_header: str | None) -> int | None:
    """
    Parse RFC 7240 Prefer header for wait directive.
    Returns seconds to wait, or None if not specified.
    """
    if not prefer_header:
        return None
    match = re.search(r"\bwait=(\d+)\b", prefer_header)
    if match:
        return int(match.group(1))
    return None


async def _queue_position(session: AsyncSession, task: Task) -> int | None:
    if task.status != TaskStatus.PENDING:
        return None
    result = await session.execute(
        select(func.count())
        .select_from(Task)
        .where(
            Task.job_id == task.job_id,
            Task.status == TaskStatus.PENDING,
            Task.created_at < task.created_at,
        )
    )
    return result.scalar() + 1


async def _resolve_job(
    session: AsyncSession, job_name: str, room_id: str
) -> Job:
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")
    job_room_id, category, name = parts
    if room_id != "@global" and job_room_id not in ("@global", room_id):
        raise JobNotFound.exception(
            detail=f"Job '{job_name}' not accessible from room '{room_id}'"
        )
    result = await session.execute(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    )
    job = result.scalar_one_or_none()
    if not job or job.deleted:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")
    return job


async def _task_response(session: AsyncSession, task: Task) -> TaskResponse:
    result = await session.execute(select(Job).where(Job.id == task.job_id))
    job = result.scalar_one_or_none()
    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        worker_id=task.worker_id,
        error=task.error,
        payload=task.payload,
        queue_position=await _queue_position(session, task),
    )


router = APIRouter(prefix="/v1/joblib", tags=["joblib"])


def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global)."""
    if room_id == "@global":
        return
    if "@" in room_id or ":" in room_id:
        raise InvalidRoomId.exception(
            detail=f"Room ID '{room_id}' contains invalid characters (@ or :)"
        )


@router.post(
    "/workers", response_model=WorkerResponse, status_code=status.HTTP_201_CREATED
)
async def create_worker(
    session: LockedSessionDep,
    user: CurrentUserDep,
):
    """Create a new worker for the authenticated user.

    Returns the worker_id which the client should use for heartbeats
    and store locally for future requests.
    """
    worker = Worker(user_id=user.id)
    session.add(worker)
    await session.commit()
    await session.refresh(worker)
    return WorkerResponse(id=worker.id, last_heartbeat=worker.last_heartbeat)


@router.put(
    "/rooms/{room_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_job(
    room_id: str,
    request: JobRegisterRequest,
    response: Response,
    session: LockedSessionDep,
    user: CurrentUserDep,
    settings: SettingsDep,
):
    """Register a job for a room. Creates worker and link if not exists."""
    # Validate room_id
    validate_room_id(room_id)

    # Check admin for @global
    if room_id == "@global" and not user.is_superuser:
        raise Forbidden.exception(detail="Admin required for @global job registration")

    # Validate category
    if request.category not in settings.allowed_categories:
        raise InvalidCategory.exception(
            detail=f"Category '{request.category}' not in allowed list: {settings.allowed_categories}"
        )

    # Check if job exists
    result = await session.execute(
        select(Job).where(
            Job.room_id == room_id,
            Job.category == request.category,
            Job.name == request.name,
        )
    )
    existing_job = result.scalar_one_or_none()

    if existing_job and existing_job.deleted:
        # Re-activate soft-deleted job with new schema
        existing_job.deleted = False
        existing_job.schema_ = request.schema_
        job = existing_job
    elif existing_job:
        # Validate schema match
        if existing_job.schema_ != request.schema_:
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
            schema_=request.schema_,
        )
        session.add(job)
        await session.flush()

    # Handle worker: use provided worker_id or auto-create
    if request.worker_id:
        # Verify ownership
        result = await session.execute(
            select(Worker).where(
                Worker.id == request.worker_id,
                Worker.user_id == user.id,
            )
        )
        worker = result.scalar_one_or_none()
        if not worker:
            raise WorkerNotFound.exception(
                detail=f"Worker '{request.worker_id}' not found or not owned by user"
            )
    else:
        # Auto-create worker for this user
        worker = Worker(user_id=user.id)
        session.add(worker)
        await session.flush()

    # Ensure worker-job link exists
    result = await session.execute(
        select(WorkerJobLink).where(
            WorkerJobLink.worker_id == worker.id,
            WorkerJobLink.job_id == job.id,
        )
    )
    link = result.scalar_one_or_none()
    if not link:
        link = WorkerJobLink(worker_id=worker.id, job_id=job.id)
        session.add(link)

    await session.commit()
    await session.refresh(job)

    # Get worker IDs for this job
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    )
    worker_links = result.scalars().all()
    worker_ids = [link.worker_id for link in worker_links]

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        workers=worker_ids,
        worker_id=worker.id,
    )


@router.get("/rooms/{room_id}/jobs", response_model=list[JobSummary])
async def list_jobs(
    room_id: str,
    session: SessionDep,
):
    """List jobs for a room. Includes @global jobs unless room_id is @global."""
    validate_room_id(room_id)

    if room_id == "@global":
        result = await session.execute(
            select(Job).where(Job.room_id == "@global", Job.deleted.is_(False))
        )
    else:
        result = await session.execute(
            select(Job).where(
                (Job.room_id == "@global") | (Job.room_id == room_id),
                Job.deleted.is_(False),
            )
        )
    jobs = result.scalars().all()

    results = []
    for job in jobs:
        result = await session.execute(
            select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
        )
        worker_links = result.scalars().all()
        worker_ids = [link.worker_id for link in worker_links]
        results.append(
            JobSummary(
                full_name=job.full_name,
                category=job.category,
                name=job.name,
                workers=worker_ids,
            )
        )
    return results


@router.get("/rooms/{room_id}/workers", response_model=list[WorkerSummary])
async def list_workers_for_room(
    room_id: str,
    session: SessionDep,
):
    """List workers serving jobs in a room. Includes @global workers unless room_id is @global."""
    validate_room_id(room_id)

    # Find jobs for this room (and @global if not requesting @global specifically)
    if room_id == "@global":
        result = await session.execute(
            select(Job).where(Job.room_id == "@global", Job.deleted.is_(False))
        )
    else:
        result = await session.execute(
            select(Job).where(
                (Job.room_id == "@global") | (Job.room_id == room_id),
                Job.deleted.is_(False),
            )
        )
    jobs = result.scalars().all()

    job_ids = [job.id for job in jobs]
    if not job_ids:
        return []

    # Find workers linked to these jobs
    result = await session.execute(
        select(WorkerJobLink.worker_id)
        .where(WorkerJobLink.job_id.in_(job_ids))
        .distinct()
    )
    worker_ids = result.scalars().all()

    if not worker_ids:
        return []

    result = await session.execute(select(Worker).where(Worker.id.in_(worker_ids)))
    workers = result.scalars().all()

    results = []
    for worker in workers:
        result = await session.execute(
            select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)
        )
        job_count = len(result.scalars().all())
        results.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return results


@router.get("/rooms/{room_id}/tasks", response_model=list[TaskResponse])
async def list_tasks_for_room(
    room_id: str,
    session: SessionDep,
    status: Optional[TaskStatus] = None,
):
    """List tasks for a room, optionally filtered by status. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    query = select(Task).where(Task.room_id == room_id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    result = await session.execute(query)
    tasks = result.scalars().all()

    return [await _task_response(session, task) for task in tasks]


@router.get(
    "/rooms/{room_id}/jobs/{job_name:path}/tasks", response_model=list[TaskResponse]
)
async def list_tasks_for_job(
    room_id: str,
    job_name: str,
    session: SessionDep,
    status: Optional[TaskStatus] = None,
):
    """List tasks for a specific job. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    query = select(Task).where(Task.job_id == job.id)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.asc())

    result = await session.execute(query)
    tasks = result.scalars().all()

    return [await _task_response(session, task) for task in tasks]


@router.get("/rooms/{room_id}/jobs/{job_name:path}", response_model=JobResponse)
async def get_job(
    room_id: str,
    job_name: str,
    session: SessionDep,
):
    """Get job details by full name."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)
    )
    worker_links = result.scalars().all()
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
    session: LockedSessionDep,
    user: CurrentUserDep,
):
    """Submit a task for processing."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    # Create task
    task = Task(
        job_id=job.id,
        room_id=room_id,
        created_by_id=user.id,
        payload=request.payload,
        status=TaskStatus.PENDING,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Set Location header
    response.headers["Location"] = f"/v1/joblib/tasks/{task.id}"
    response.headers["Retry-After"] = "1"

    return await _task_response(session, task)


@router.post("/tasks/claim", response_model=TaskClaimResponse)
async def claim_task(
    request: TaskClaimRequest,
    session: LockedSessionDep,
    user: CurrentUserDep,
    settings: SettingsDep,
):
    """Claim the oldest pending task for jobs the specified worker is registered for."""
    # Validate that worker_id exists and belongs to the authenticated user
    result = await session.execute(
        select(Worker).where(Worker.id == request.worker_id)
    )
    worker = result.scalar_one_or_none()

    if not worker:
        raise WorkerNotFound.exception(f"Worker {request.worker_id} not found")

    if worker.user_id != user.id:
        raise Forbidden.exception("Worker belongs to a different user")

    # Find job IDs for this specific worker
    result = await session.execute(
        select(WorkerJobLink.job_id).where(WorkerJobLink.worker_id == request.worker_id)
    )
    worker_job_ids = result.scalars().all()

    if not worker_job_ids:
        return TaskClaimResponse(task=None)

    # Use atomic UPDATE with WHERE clause for optimistic locking
    # This handles the race condition where multiple workers try to claim the same task
    max_attempts = settings.claim_max_attempts
    base_delay = settings.claim_base_delay_seconds
    claimed_task_id = None

    for attempt in range(max_attempts):
        try:
            # Find oldest pending task for jobs this worker is registered for
            result = await session.execute(
                select(Task.id)
                .where(Task.job_id.in_(worker_job_ids), Task.status == TaskStatus.PENDING)
                .order_by(Task.created_at.asc())
                .limit(1)
            )
            task_id = result.scalar_one_or_none()

            if not task_id:
                return TaskClaimResponse(task=None)

            # Atomically update only if still PENDING (optimistic locking)
            stmt = (
                update(Task)
                .where(Task.id == task_id, Task.status == TaskStatus.PENDING)
                .values(status=TaskStatus.CLAIMED, worker_id=request.worker_id)
            )
            cursor_result = await session.execute(stmt)
            await session.commit()

            # Check if we actually claimed it (rowcount == 1 means success)
            if cursor_result.rowcount == 1:
                claimed_task_id = task_id
                break

            # Another worker claimed it first - exponential backoff with jitter
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            await asyncio.sleep(delay)

        except OperationalError:
            # Database locked/timeout - rollback and retry with backoff
            await session.rollback()
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            await asyncio.sleep(delay)

    if not claimed_task_id:
        return TaskClaimResponse(task=None)

    # Fetch the claimed task
    result = await session.execute(select(Task).where(Task.id == claimed_task_id))
    task = result.scalar_one()

    return TaskClaimResponse(task=await _task_response(session, task))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: UUID,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    prefer: str | None = Header(None),
):
    """Get task status."""
    result = await session.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    requested_wait = parse_prefer_wait(prefer)

    # Only long-poll if wait requested AND task not in terminal state
    if requested_wait and requested_wait > 0 and task.status not in TERMINAL_STATUSES:
        effective_wait = min(requested_wait, settings.long_poll_max_wait_seconds)
        elapsed = 0.0
        poll_interval = 1.0  # Check DB every second

        while elapsed < effective_wait and task.status not in TERMINAL_STATUSES:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Re-fetch task from DB
            session.expire(task)
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

        response.headers["Preference-Applied"] = f"wait={int(effective_wait)}"

    return await _task_response(session, task)


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    request: TaskUpdateRequest,
    session: LockedSessionDep,
    user: CurrentUserDep,
):
    """Update task status. Requires the task's worker owner or superuser."""
    result = await session.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    # Authorization: worker owner or superuser
    if not user.is_superuser:
        if task.worker_id is None:
            raise Forbidden.exception(detail="Task not claimed by any worker")
        result = await session.execute(
            select(Worker).where(Worker.id == task.worker_id)
        )
        worker = result.scalar_one_or_none()
        if not worker or worker.user_id != user.id:
            raise Forbidden.exception(detail="Not authorized to update this task")

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
    elif request.status in TERMINAL_STATUSES:
        task.completed_at = now
        if request.error:
            task.error = request.error

    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Soft-delete orphan job if task reached terminal state
    if request.status in TERMINAL_STATUSES:
        await _soft_delete_orphan_job(session, task.job_id)
        await session.commit()

    return await _task_response(session, task)


@router.get("/workers", response_model=list[WorkerSummary])
async def list_workers(
    session: SessionDep,
):
    """List all workers with their job counts."""
    result = await session.execute(select(Worker))
    workers = result.scalars().all()

    results = []
    for worker in workers:
        result = await session.execute(
            select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)
        )
        job_count = len(result.scalars().all())
        results.append(
            WorkerSummary(
                id=worker.id,
                last_heartbeat=worker.last_heartbeat,
                job_count=job_count,
            )
        )
    return results


@router.patch("/workers/{worker_id}", response_model=WorkerResponse)
async def worker_heartbeat(
    worker_id: UUID,
    session: LockedSessionDep,
    user: CurrentUserDep,
):
    """Update worker heartbeat. Worker must belong to authenticated user."""
    result = await session.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    if worker.user_id != user.id:
        raise Forbidden.exception(detail="Worker belongs to different user")

    worker.last_heartbeat = datetime.now(timezone.utc)
    session.add(worker)
    await session.commit()
    await session.refresh(worker)

    return WorkerResponse(id=worker.id, last_heartbeat=worker.last_heartbeat)


@router.delete("/workers/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_worker(
    worker_id: UUID,
    session: LockedSessionDep,
    user: CurrentUserDep,
):
    """Delete worker, fail their tasks, remove job links, and clean up orphan jobs.

    Worker must belong to authenticated user or user must be superuser.
    """
    result = await session.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    if worker.user_id != user.id and not user.is_superuser:
        raise Forbidden.exception(detail="Worker belongs to different user")

    await _cleanup_worker(session, worker)
    await session.commit()
