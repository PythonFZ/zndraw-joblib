# tests/test_init.py
"""Tests for public API exports."""


def test_public_api_exports():
    """All public API should be importable from the main package."""
    from zndraw_joblib import (
        # Router
        router,
        # Models
        Job,
        Worker,
        Task,
        WorkerJobLink,
        TaskStatus,
        # Dependencies
        get_db_session,
        get_current_identity,
        get_is_admin,
        get_settings,
        # Exceptions
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
        # Settings
        JobLibSettings,
        # Client
        JobManager,
        ClaimedTask,
        Extension,
        Category,
    )

    assert router is not None
    assert Job is not None
    assert JobManager is not None
    assert Category.MODIFIER.value == "modifiers"


def test_all_exports_in_dunder_all():
    """__all__ should contain all public exports."""
    import zndraw_joblib

    expected = [
        "router",
        "Job",
        "Worker",
        "Task",
        "WorkerJobLink",
        "TaskStatus",
        "get_db_session",
        "get_current_identity",
        "get_is_admin",
        "get_settings",
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
        "JobLibSettings",
        "JobManager",
        "ClaimedTask",
        "Extension",
        "Category",
    ]

    for name in expected:
        assert name in zndraw_joblib.__all__, f"{name} not in __all__"
        assert hasattr(zndraw_joblib, name), f"{name} not accessible"
