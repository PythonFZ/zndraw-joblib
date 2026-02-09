# src/zndraw_joblib/__init__.py
"""ZnDraw Job Management Library."""

from zndraw_joblib.router import router
from zndraw_joblib.models import Job, Worker, Task, WorkerJobLink, TaskStatus
from zndraw_joblib.dependencies import get_settings, get_tsio
from zndraw_joblib.exceptions import (
    ProblemException,
    problem_exception_handler,
    JobNotFound,
    SchemaConflict,
    InvalidCategory,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
    InvalidRoomId,
    Forbidden,
    InternalJobNotConfigured,
)
from zndraw_joblib.schemas import PaginatedResponse
from zndraw_joblib.settings import JobLibSettings
from zndraw_joblib.client import JobManager, ClaimedTask, Extension, Category
from zndraw_joblib.registry import (
    register_internal_jobs,
    register_internal_tasks,
    InternalExecutor,
    InternalRegistry,
)
from zndraw_joblib.sweeper import (
    run_sweeper,
    cleanup_stale_workers,
    cleanup_stuck_internal_tasks,
)
from zndraw_joblib.events import (
    JobsInvalidate,
    TaskAvailable,
    TaskStatusEvent,
    JoinJobRoom,
    LeaveJobRoom,
    Emission,
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
    # Events
    "JobsInvalidate",
    "TaskAvailable",
    "TaskStatusEvent",
    "JoinJobRoom",
    "LeaveJobRoom",
    "Emission",
]
