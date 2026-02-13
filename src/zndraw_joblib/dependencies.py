# src/zndraw_joblib/dependencies.py
from typing import Annotated

from fastapi import Depends, Path, Request
from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.exceptions import InvalidRoomId
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


def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global and @internal)."""
    if room_id in ("@global", "@internal"):
        return
    if not room_id or "@" in room_id or ":" in room_id:
        raise InvalidRoomId.exception(
            detail=f"Room ID '{room_id}' contains invalid characters (@ or :)"
        )


async def verify_writable_room(room_id: str = Path()) -> str:
    """Verify the room is writable. Override in host app for lock checks.

    Default implementation only validates the room_id format.
    Host apps override this via ``app.dependency_overrides[verify_writable_room]``
    to add lock checks (e.g. admin lock, edit lock).
    """
    validate_room_id(room_id)
    return room_id


WritableRoomDep = Annotated[str, Depends(verify_writable_room)]
