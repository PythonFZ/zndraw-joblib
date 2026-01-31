# src/zndraw_joblib/__init__.py
"""ZnDraw Job Management Library."""

from zndraw_joblib.router import router
from zndraw_joblib.models import Job, Worker, Task, WorkerJobLink, TaskStatus
from zndraw_joblib.dependencies import (
    get_db_session,
    get_redis_client,
    get_current_identity,
    get_is_admin,
    get_settings,
)
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
)
from zndraw_joblib.settings import JobLibSettings
from zndraw_joblib.client import JobManager, ClaimedTask, Extension, Category

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
    "get_db_session",
    "get_redis_client",
    "get_current_identity",
    "get_is_admin",
    "get_settings",
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
    # Settings
    "JobLibSettings",
    # Client
    "JobManager",
    "ClaimedTask",
    "Extension",
    "Category",
]
