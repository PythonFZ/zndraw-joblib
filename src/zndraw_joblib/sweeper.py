# src/zndraw_joblib/sweeper.py
"""Background sweeper for cleaning up stale workers."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zndraw_joblib.models import Worker, Task, TaskStatus, Job, WorkerJobLink, TERMINAL_STATUSES
from zndraw_joblib.settings import JobLibSettings

logger = logging.getLogger(__name__)


async def _soft_delete_orphan_job(session: AsyncSession, job_id: uuid.UUID) -> None:
    """Soft-delete a job if it has no workers and no non-terminal tasks.

    Note: Does NOT commit the transaction - caller must commit.
    """
    # Check if job has any remaining workers
    result = await session.execute(
        select(WorkerJobLink).where(WorkerJobLink.job_id == job_id).limit(1)
    )
    if result.scalar_one_or_none():
        return  # Job still has workers

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
        return  # Job has pending/running tasks

    # Job is orphan - soft delete (tasks remain as historical records)
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job:
        job.deleted = True
        session.add(job)


async def _cleanup_worker(session: AsyncSession, worker: Worker) -> None:
    """Clean up a worker by failing tasks, removing links, and soft-deleting orphan jobs.

    This is the shared cleanup logic used by both delete_worker endpoint and sweeper.
    Note: Does NOT commit the transaction - caller must commit.
    """
    now = datetime.now(timezone.utc)

    # Fail any claimed/running tasks owned by this worker
    result = await session.execute(
        select(Task).where(
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
        await _soft_delete_orphan_job(session, job_id)


async def cleanup_stale_workers(session: AsyncSession, timeout: timedelta) -> int:
    """Find and clean up workers with stale heartbeats.

    Args:
        session: Async database session
        timeout: How long since last heartbeat before a worker is considered stale

    Returns:
        Count of workers cleaned up
    """
    cutoff = datetime.now(timezone.utc) - timeout

    # Find all stale workers
    result = await session.execute(select(Worker).where(Worker.last_heartbeat < cutoff))
    stale_workers = result.scalars().all()

    count = 0
    for worker in stale_workers:
        logger.info(f"Cleaning up stale worker: {worker.id}")
        await _cleanup_worker(session, worker)
        count += 1

    if count > 0:
        await session.commit()

    return count


async def run_sweeper(
    get_session: Callable[[], AsyncGenerator[AsyncSession, None]],
    settings: JobLibSettings,
) -> None:
    """Background task that runs cleanup periodically.

    Args:
        get_session: Async generator function that yields database sessions
        settings: Application settings containing timeout and interval config
    """
    timeout = timedelta(seconds=settings.worker_timeout_seconds)
    interval = settings.sweeper_interval_seconds

    logger.info(
        f"Starting sweeper with interval={interval}s, worker_timeout={settings.worker_timeout_seconds}s"
    )

    while True:
        await asyncio.sleep(interval)
        try:
            async for session in get_session():
                count = await cleanup_stale_workers(session, timeout)
                if count > 0:
                    logger.info(f"Cleaned up {count} stale worker(s)")
        except Exception as e:
            logger.exception(f"Error in sweeper: {e}")
