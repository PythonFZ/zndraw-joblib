# src/zndraw_joblib/dependencies.py
from fastapi import Request
from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.registry import InternalRegistry
from zndraw_joblib.settings import JobLibSettings


def get_settings() -> JobLibSettings:
    """Create a settings instance from environment variables.

    Override this dependency in tests to inject custom settings.
    """
    return JobLibSettings()


async def get_internal_registry(request: Request) -> InternalRegistry | None:
    """Return the internal registry from app.state, or None if not configured."""
    return getattr(request.app.state, "internal_registry", None)


async def get_tsio() -> AsyncServerWrapper | None:
    """Return the Socket.IO server wrapper for emitting events.

    Override this dependency in the host app to provide a real
    zndraw-socketio AsyncServerWrapper. Returns None by default,
    which disables all real-time event emissions.
    """
    return None
