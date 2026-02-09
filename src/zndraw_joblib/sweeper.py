# src/zndraw_joblib/sweeper.py
"""Background sweeper for cleaning up stale workers."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.events import (
    Emission,
    JobsInvalidate,
    build_task_status_emission,
    emit,
)
from zndraw_joblib.models import (
    TERMINAL_STATUSES,
    Job,
    Task,
    TaskStatus,
    Worker,
    WorkerJobLink,
)
from zndraw_joblib.settings import JobLibSettings

logger = logging.getLogger(__name__)


async def _soft_delete_orphan_job(
    session: AsyncSession, job_id: uuid.UUID
) -> set[Emission]:
    """Soft-delete a job if it has no workers and no non-terminal tasks.

    Note: Does NOT commit the transaction - caller must commit.
    """
    # Check if job has any remaining workers
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job_id).limit(1)
    )
    if result.scalar_one_or_none():
        return set()  # Job still has workers

    # Check if job has any non-terminal tasks
    result = await session.execute(
        select(Task)
        .where(
            Task.job_id == job_id,
            Task.status.not_in(TERMINAL_STATUSES),
        )
        .limit(1)
    )
    if result.scalar_one_or_none():
        return set()  # Job has pending/running tasks

    # Job is orphan - soft delete (tasks remain as historical records)
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job:
        job.deleted = True
        session.add(job)
        return {Emission(JobsInvalidate(), f"room:{job.room_id}")}

    return set()


async def cleanup_worker(session: AsyncSession, worker: Worker) -> set[Emission]:
    """Clean up a worker by failing tasks, removing links, and soft-deleting orphan jobs.

    This is the shared cleanup logic used by both delete_worker endpoint and sweeper.
    Note: Does NOT commit the transaction - caller must commit.
    """
    emissions: set[Emission] = set()
    now = datetime.now(timezone.utc)

    # Fail any claimed/running tasks owned by this worker
    result = await session.execute(
        select(Task)
        .options(selectinload(Task.job))
        .where(
            Task.worker_id == worker.id,
            Task.status.in_({TaskStatus.CLAIMED, TaskStatus.RUNNING}),
        )
    )
    worker_tasks = result.scalars().all()
    for task in worker_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Worker disconnected"
        session.add(task)
        emissions.add(
            build_task_status_emission(task, task.job.full_name if task.job else "")
        )

    # Get job IDs this worker is linked to before deleting links
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)
    )
    links = result.scalars().all()
    job_ids = [link.job_id for link in links]

    # Delete all links
    for link in links:
        await session.delete(link)

    # Delete worker
    await session.delete(worker)
    await session.flush()

    # Clean up orphan jobs (no workers and no non-terminal tasks)
    for job_id in job_ids:
        emissions |= await _soft_delete_orphan_job(session, job_id)

    return emissions


async def cleanup_stale_workers(
    session: AsyncSession, timeout: timedelta
) -> tuple[int, set[Emission]]:
    """Find and clean up workers with stale heartbeats.

    Args:
        session: Async database session
        timeout: How long since last heartbeat before a worker is considered stale

    Returns:
        Tuple of (count of workers cleaned up, set of emissions).
    """
    cutoff = datetime.now(timezone.utc) - timeout
    all_emissions: set[Emission] = set()

    # Find all stale workers
    result = await session.execute(select(Worker).where(Worker.last_heartbeat < cutoff))
    stale_workers = result.scalars().all()

    count = 0
    for worker in stale_workers:
        logger.info("Cleaning up stale worker: %s", worker.id)
        emissions = await cleanup_worker(session, worker)
        all_emissions |= emissions
        count += 1

    if count > 0:
        await session.commit()

    return count, all_emissions


async def cleanup_stuck_internal_tasks(
    session: AsyncSession, timeout: timedelta
) -> tuple[int, set[Emission]]:
    """Find and fail @internal tasks stuck in RUNNING beyond timeout.

    Args:
        session: Async database session
        timeout: How long a RUNNING internal task can run before being considered stuck

    Returns:
        Tuple of (count of tasks failed, set of emissions).
    """
    cutoff = datetime.now(timezone.utc) - timeout
    now = datetime.now(timezone.utc)
    emissions: set[Emission] = set()

    result = await session.execute(
        select(Task)
        .join(Job)
        .options(selectinload(Task.job))
        .where(
            Job.room_id == "@internal",
            Task.status == TaskStatus.RUNNING,
            Task.started_at < cutoff,
        )
    )
    stuck_tasks = result.scalars().all()

    count = 0
    for task in stuck_tasks:
        task.status = TaskStatus.FAILED
        task.completed_at = now
        task.error = "Internal worker timeout"
        session.add(task)
        emissions.add(
            build_task_status_emission(task, task.job.full_name if task.job else "")
        )
        count += 1

    if count > 0:
        await session.commit()
        logger.info("Failed %d stuck internal task(s)", count)

    return count, emissions


async def run_sweeper(
    get_session: Callable[[], AsyncGenerator[AsyncSession, None]],
    settings: JobLibSettings,
    tsio: AsyncServerWrapper | None = None,
) -> None:
    """Background task that runs cleanup periodically."""
    timeout = timedelta(seconds=settings.worker_timeout_seconds)
    internal_timeout = timedelta(seconds=settings.internal_task_timeout_seconds)
    interval = settings.sweeper_interval_seconds

    logger.info(
        "Starting sweeper with interval=%ss, worker_timeout=%ss, internal_task_timeout=%ss",
        interval,
        settings.worker_timeout_seconds,
        settings.internal_task_timeout_seconds,
    )

    while True:
        await asyncio.sleep(interval)
        try:
            async for session in get_session():
                count, emissions = await cleanup_stale_workers(session, timeout)
                if count > 0:
                    logger.info("Cleaned up %s stale worker(s)", count)
                await emit(tsio, emissions)

            async for session in get_session():
                count, emissions = await cleanup_stuck_internal_tasks(
                    session, internal_timeout
                )
                if count > 0:
                    logger.info("Failed %s stuck internal task(s)", count)
                await emit(tsio, emissions)
        except Exception as e:
            logger.exception("Error in sweeper: %s", e)
