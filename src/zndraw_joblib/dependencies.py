# src/zndraw_joblib/dependencies.py
from typing import Annotated

from fastapi import Depends, Request
from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.registry import InternalRegistry
from zndraw_joblib.settings import JobLibSettings


def get_joblib_settings(request: Request) -> JobLibSettings:
    """Return joblib settings from app.state."""
    return request.app.state.joblib_settings


JobLibSettingsDep = Annotated[JobLibSettings, Depends(get_joblib_settings)]


async def get_internal_registry(request: Request) -> InternalRegistry | None:
    """Return the internal registry from app.state, or None if not configured."""
    return getattr(request.app.state, "internal_registry", None)


def get_tsio(request: Request) -> AsyncServerWrapper | None:
    """Return the Socket.IO server wrapper from app.state.

    Returns None if tsio is not configured, which disables
    all real-time event emissions.
    """
    return getattr(request.app.state, "tsio", None)
