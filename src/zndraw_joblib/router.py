# src/zndraw_joblib/router.py
import asyncio
import random
import logging
import re
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from zndraw_auth import User, current_active_user, current_superuser

from zndraw_joblib.dependencies import (
    get_settings,
    get_locked_async_session,
    get_session_factory,
    get_internal_registry,
    get_tsio,
)
from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.events import (
    Emission,
    JobsInvalidate,
    TaskAvailable,
    build_task_status_emission,
    emit,
)
from zndraw_joblib.registry import InternalRegistry
from zndraw_joblib.exceptions import (
    InvalidCategory,
    InvalidRoomId,
    SchemaConflict,
    Forbidden,
    JobNotFound,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
    InternalJobNotConfigured,
)
from zndraw_joblib.models import (
    Job,
    Worker,
    WorkerJobLink,
    Task,
    TaskStatus,
    TERMINAL_STATUSES,
)
from zndraw_joblib.sweeper import cleanup_worker, _soft_delete_orphan_job
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
    PaginatedResponse,
)
from zndraw_joblib.settings import JobLibSettings

logger = logging.getLogger(__name__)

# Type aliases for dependency injection
CurrentUserDep = Annotated[User, Depends(current_active_user)]
SuperUserDep = Annotated[User, Depends(current_superuser)]
LockedSessionDep = Annotated[AsyncSession, Depends(get_locked_async_session)]
SettingsDep = Annotated[JobLibSettings, Depends(get_settings)]
SessionFactoryDep = Annotated[object, Depends(get_session_factory)]
InternalRegistryDep = Annotated[InternalRegistry | None, Depends(get_internal_registry)]
TsioDep = Annotated[AsyncServerWrapper | None, Depends(get_tsio)]

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
    """Get queue position for a single task via the bulk helper."""
    if task.status != TaskStatus.PENDING:
        return None
    positions = await _bulk_queue_positions(session, [task.id])
    return positions.get(task.id)


async def _resolve_job(session: AsyncSession, job_name: str, room_id: str) -> Job:
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")
    job_room_id, category, name = parts

    # For non-special rooms, the job must be in the room, @global, or @internal
    if room_id not in ("@global", "@internal") and job_room_id not in (
        "@global",
        "@internal",
        room_id,
    ):
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


async def _bulk_queue_positions(
    session: AsyncSession, task_ids: list[UUID]
) -> dict[UUID, int]:
    """Compute queue positions for pending tasks in a single query using a window function."""
    if not task_ids:
        return {}
    # Window function: rank each pending task within its job by created_at
    row_num = (
        func.row_number()
        .over(partition_by=Task.job_id, order_by=Task.created_at.asc())
        .label("pos")
    )
    subq = select(Task.id, row_num).where(Task.status == TaskStatus.PENDING).subquery()
    result = await session.execute(
        select(subq.c.id, subq.c.pos).where(subq.c.id.in_(task_ids))
    )
    return {row.id: row.pos for row in result}


async def _bulk_task_responses(
    session: AsyncSession, tasks: list[Task]
) -> list[TaskResponse]:
    """Build TaskResponse list efficiently by batching job lookups and queue positions.

    Expects tasks to have been loaded with selectinload(Task.job).
    """
    pending_ids = [t.id for t in tasks if t.status == TaskStatus.PENDING]
    queue_positions = await _bulk_queue_positions(session, pending_ids)

    return [
        TaskResponse(
            id=t.id,
            job_name=t.job.full_name if t.job else "",
            room_id=t.room_id,
            status=t.status,
            created_at=t.created_at,
            started_at=t.started_at,
            completed_at=t.completed_at,
            worker_id=t.worker_id,
            error=t.error,
            payload=t.payload,
            queue_position=queue_positions.get(t.id),
        )
        for t in tasks
    ]


async def _task_status_emission(session: AsyncSession, task: Task) -> Emission:
    """Build a TaskStatusEvent emission from a task, querying job name and queue position."""
    result = await session.execute(select(Job).where(Job.id == task.job_id))
    job = result.scalar_one_or_none()
    return build_task_status_emission(
        task,
        job_full_name=job.full_name if job else "",
        queue_position=await _queue_position(session, task),
    )


def _room_job_filter(room_id: str):
    """Build a SQLAlchemy filter for jobs visible from a given room."""
    if room_id == "@global":
        return Job.room_id == "@global"
    if room_id == "@internal":
        return Job.room_id == "@internal"
    return (Job.room_id.in_(["@global", "@internal"])) | (Job.room_id == room_id)


router = APIRouter(prefix="/v1/joblib", tags=["joblib"])


def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global and @internal)."""
    if room_id in ("@global", "@internal"):
        return
    if not room_id or "@" in room_id or ":" in room_id:
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
    tsio: TsioDep,
):
    """Register a job for a room. Creates worker and link if not exists."""
    # Validate room_id
    validate_room_id(room_id)

    # Check admin for @global and @internal
    if room_id in ("@global", "@internal") and not user.is_superuser:
        raise Forbidden.exception(
            detail="Admin required for @global/@internal job registration"
        )

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
    await emit(tsio, {Emission(JobsInvalidate(), f"room:{room_id}")})
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


@router.get("/rooms/{room_id}/jobs", response_model=PaginatedResponse[JobSummary])
async def list_jobs(
    room_id: str,
    session: LockedSessionDep,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List jobs for a room. Includes @global jobs unless room_id is @global."""
    validate_room_id(room_id)

    base_query = select(Job).where(_room_job_filter(room_id), Job.deleted.is_(False))

    # Total count
    total_result = await session.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = total_result.scalar()

    # Paginated + eager-load workers
    result = await session.execute(
        base_query.options(selectinload(Job.workers)).offset(offset).limit(limit)
    )
    jobs = result.scalars().all()

    items = [
        JobSummary(
            full_name=job.full_name,
            category=job.category,
            name=job.name,
            workers=[w.id for w in job.workers],
        )
        for job in jobs
    ]
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/rooms/{room_id}/workers", response_model=PaginatedResponse[WorkerSummary])
async def list_workers_for_room(
    room_id: str,
    session: LockedSessionDep,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List workers serving jobs in a room. Includes @global workers unless room_id is @global."""
    validate_room_id(room_id)

    result = await session.execute(
        select(Job.id).where(_room_job_filter(room_id), Job.deleted.is_(False))
    )
    job_ids = result.scalars().all()

    if not job_ids:
        return PaginatedResponse(items=[], total=0, limit=limit, offset=offset)

    # Find distinct worker IDs linked to these jobs
    worker_id_query = (
        select(WorkerJobLink.worker_id)
        .where(WorkerJobLink.job_id.in_(job_ids))
        .distinct()
    )

    # Total count
    total_result = await session.execute(
        select(func.count()).select_from(worker_id_query.subquery())
    )
    total = total_result.scalar()

    # Paginated workers with eager-loaded jobs
    result = await session.execute(
        select(Worker)
        .where(Worker.id.in_(worker_id_query))
        .options(selectinload(Worker.jobs))
        .offset(offset)
        .limit(limit)
    )
    workers = result.scalars().all()

    items = [
        WorkerSummary(
            id=worker.id,
            last_heartbeat=worker.last_heartbeat,
            job_count=len(worker.jobs),
        )
        for worker in workers
    ]
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/rooms/{room_id}/tasks", response_model=PaginatedResponse[TaskResponse])
async def list_tasks_for_room(
    room_id: str,
    session: LockedSessionDep,
    task_status: TaskStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List tasks for a room, optionally filtered by status. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    base_query = select(Task).where(Task.room_id == room_id)
    if task_status:
        base_query = base_query.where(Task.status == task_status)

    # Total count
    total_result = await session.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = total_result.scalar()

    # Paginated + eager-load job relationship
    result = await session.execute(
        base_query.options(selectinload(Task.job))
        .order_by(Task.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    tasks = result.scalars().all()

    items = await _bulk_task_responses(session, tasks)
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/rooms/{room_id}/jobs/{job_name:path}/tasks",
    response_model=PaginatedResponse[TaskResponse],
)
async def list_tasks_for_job(
    room_id: str,
    job_name: str,
    session: LockedSessionDep,
    task_status: TaskStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List tasks for a specific job. Includes queue position for pending tasks."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    base_query = select(Task).where(Task.job_id == job.id)
    if task_status:
        base_query = base_query.where(Task.status == task_status)

    # Total count
    total_result = await session.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = total_result.scalar()

    # Paginated + eager-load job relationship
    result = await session.execute(
        base_query.options(selectinload(Task.job))
        .order_by(Task.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    tasks = result.scalars().all()

    items = await _bulk_task_responses(session, tasks)
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/rooms/{room_id}/jobs/{job_name:path}", response_model=JobResponse)
async def get_job(
    room_id: str,
    job_name: str,
    session: LockedSessionDep,
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
    internal_registry: InternalRegistryDep,
    tsio: TsioDep,
):
    """Submit a task for processing."""
    validate_room_id(room_id)

    job = await _resolve_job(session, job_name, room_id)

    # Validate internal registry BEFORE creating the task to avoid orphans
    if job.room_id == "@internal":
        if internal_registry is None or job.full_name not in internal_registry.tasks:
            raise InternalJobNotConfigured.exception(
                detail=f"Internal job '{job.full_name}' is registered but no executor is available"
            )

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

    # Dispatch to taskiq for @internal jobs (fail task if dispatch fails)
    if job.room_id == "@internal":
        try:
            await internal_registry.tasks[job.full_name].kiq(
                task_id=str(task.id),
                room_id=room_id,
                payload=request.payload,
                base_url="",
            )
        except Exception:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error = "Failed to dispatch to internal executor"
            await session.commit()
            await session.refresh(task)
            return await _task_response(session, task)

    # Emit events (after successful dispatch for @internal)
    emissions: set[Emission] = {await _task_status_emission(session, task)}
    if job.room_id != "@internal":
        emissions.add(
            Emission(
                TaskAvailable(
                    job_name=job.full_name,
                    room_id=room_id,
                    task_id=str(task.id),
                ),
                f"jobs:{job.full_name}",
            )
        )
    await emit(tsio, emissions)

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
    tsio: TsioDep,
):
    """Claim the oldest pending task for jobs the specified worker is registered for."""
    # Validate that worker_id exists and belongs to the authenticated user
    result = await session.execute(select(Worker).where(Worker.id == request.worker_id))
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
                .where(
                    Task.job_id.in_(worker_job_ids), Task.status == TaskStatus.PENDING
                )
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
            delay = base_delay * (2**attempt) * (0.5 + random.random())
            await asyncio.sleep(delay)

        except OperationalError:
            # Database locked/timeout - rollback and retry with backoff
            logger.warning(
                "OperationalError during claim (attempt %d/%d)",
                attempt + 1,
                max_attempts,
            )
            await session.rollback()
            delay = base_delay * (2**attempt) * (0.5 + random.random())
            await asyncio.sleep(delay)

    if not claimed_task_id:
        logger.warning(
            "Failed to claim task after %d attempts for worker %s",
            max_attempts,
            request.worker_id,
        )
        return TaskClaimResponse(task=None)

    # Fetch the claimed task
    result = await session.execute(select(Task).where(Task.id == claimed_task_id))
    task = result.scalar_one()

    await emit(tsio, {await _task_status_emission(session, task)})

    return TaskClaimResponse(task=await _task_response(session, task))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: UUID,
    response: Response,
    session_factory: SessionFactoryDep,
    settings: SettingsDep,
    prefer: str | None = Header(None),
):
    """Get task status. Supports long-polling via Prefer: wait=N header."""
    # Initial lookup
    async with session_factory() as session:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if not task:
            raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    requested_wait = parse_prefer_wait(prefer)

    # Only long-poll if wait requested AND task not in terminal state
    if requested_wait and requested_wait > 0 and task.status not in TERMINAL_STATUSES:
        effective_wait = min(requested_wait, settings.long_poll_max_wait_seconds)
        elapsed = 0.0
        poll_interval = 1.0

        while elapsed < effective_wait and task.status not in TERMINAL_STATUSES:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            poll_interval = min(
                poll_interval * 1.5, 5.0
            )  # Exponential backoff, cap at 5s

            async with session_factory() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if not task:
                    raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

        response.headers["Preference-Applied"] = f"wait={int(effective_wait)}"

    # Build final response — re-fetch to avoid stale detached object
    async with session_factory() as session:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one()
        return await _task_response(session, task)


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    request: TaskUpdateRequest,
    session: LockedSessionDep,
    user: CurrentUserDep,
    tsio: TsioDep,
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

    # Handle orphan job cleanup in the same transaction
    if request.status in TERMINAL_STATUSES:
        await session.flush()
        await _soft_delete_orphan_job(session, task.job_id)

    await session.commit()
    await session.refresh(task)

    # Emit events after commit
    emissions: set[Emission] = {await _task_status_emission(session, task)}
    await emit(tsio, emissions)

    return await _task_response(session, task)


@router.get("/workers", response_model=PaginatedResponse[WorkerSummary])
async def list_workers(
    session: LockedSessionDep,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List all workers with their job counts."""
    # Total count
    total_result = await session.execute(select(func.count()).select_from(Worker))
    total = total_result.scalar()

    # Paginated + eager-load jobs
    result = await session.execute(
        select(Worker).options(selectinload(Worker.jobs)).offset(offset).limit(limit)
    )
    workers = result.scalars().all()

    items = [
        WorkerSummary(
            id=worker.id,
            last_heartbeat=worker.last_heartbeat,
            job_count=len(worker.jobs),
        )
        for worker in workers
    ]
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


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
    tsio: TsioDep,
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

    emissions = await cleanup_worker(session, worker)
    await session.commit()
    await emit(tsio, emissions)
