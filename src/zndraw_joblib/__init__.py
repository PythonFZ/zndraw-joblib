# src/zndraw_joblib/__init__.py
"""ZnDraw Job Management Library."""

from zndraw_joblib.client import Category, ClaimedTask, Extension, JobManager
from zndraw_joblib.dependencies import get_settings, get_tsio
from zndraw_joblib.events import (
    Emission,
    FrozenEvent,
    JobsInvalidate,
    JoinJobRoom,
    LeaveJobRoom,
    TaskAvailable,
    TaskStatusEvent,
    build_task_status_emission,
    emit,
)
from zndraw_joblib.exceptions import (
    Forbidden,
    InternalJobNotConfigured,
    InvalidCategory,
    InvalidRoomId,
    InvalidTaskTransition,
    JobNotFound,
    ProblemException,
    SchemaConflict,
    TaskNotFound,
    WorkerNotFound,
    problem_exception_handler,
)
from zndraw_joblib.models import Job, Task, TaskStatus, Worker, WorkerJobLink
from zndraw_joblib.registry import (
    InternalExecutor,
    InternalRegistry,
    register_internal_jobs,
    register_internal_tasks,
)
from zndraw_joblib.router import router
from zndraw_joblib.schemas import PaginatedResponse
from zndraw_joblib.settings import JobLibSettings
from zndraw_joblib.sweeper import (
    cleanup_stale_workers,
    cleanup_stuck_internal_tasks,
    cleanup_worker,
    run_sweeper,
)

__all__ = [
    # Router
    "router",
    # Models
    "Job",
    "Worker",
    "Task",
    "WorkerJobLink",
    "TaskStatus",
    # Dependencies
    "get_settings",
    "get_tsio",
    # Exceptions
    "ProblemException",
    "problem_exception_handler",
    "JobNotFound",
    "SchemaConflict",
    "InvalidCategory",
    "WorkerNotFound",
    "TaskNotFound",
    "InvalidTaskTransition",
    "InvalidRoomId",
    "Forbidden",
    "InternalJobNotConfigured",
    # Schemas
    "PaginatedResponse",
    # Settings
    "JobLibSettings",
    # Client
    "JobManager",
    "ClaimedTask",
    "Extension",
    "Category",
    # Internal registry
    "register_internal_jobs",
    "register_internal_tasks",
    "InternalExecutor",
    "InternalRegistry",
    # Sweeper
    "run_sweeper",
    "cleanup_stale_workers",
    "cleanup_stuck_internal_tasks",
    "cleanup_worker",
    # Events
    "FrozenEvent",
    "JobsInvalidate",
    "TaskAvailable",
    "TaskStatusEvent",
    "JoinJobRoom",
    "LeaveJobRoom",
    "Emission",
    "build_task_status_emission",
    "emit",
]
