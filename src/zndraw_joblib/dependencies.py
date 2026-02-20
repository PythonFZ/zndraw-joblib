# src/zndraw_joblib/dependencies.py
import hashlib
import json
from typing import Annotated, Any, Protocol, runtime_checkable

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


@runtime_checkable
class ResultBackend(Protocol):
    """Protocol for storing and retrieving cached provider results."""

    async def store(self, key: str, data: bytes, ttl: int) -> None: ...

    async def get(self, key: str) -> bytes | None: ...

    async def delete(self, key: str) -> None: ...

    async def acquire_inflight(self, key: str, ttl: int) -> bool:
        """Return True if lock acquired (SET NX semantics)."""
        ...

    async def release_inflight(self, key: str) -> None: ...

    async def wait_for_key(self, key: str, timeout: float) -> bytes | None:
        """Wait for a cache key to be populated.

        Subscribes to a notification channel, checks cache (race-safe),
        then awaits notification or timeout.

        Returns
        -------
        bytes | None
            Cached data if it arrives within timeout, None otherwise.
        """
        ...

    async def notify_key(self, key: str) -> None:
        """Notify waiters that a cache key has been populated."""
        ...


async def get_result_backend() -> ResultBackend:
    """Return the configured ResultBackend.

    Host apps must override this dependency via
    ``app.dependency_overrides[get_result_backend]``.
    """
    raise NotImplementedError(
        "ResultBackend not configured — host app must override get_result_backend"
    )


ResultBackendDep = Annotated[ResultBackend, Depends(get_result_backend)]


def request_hash(params: dict[str, Any]) -> str:
    """Return a SHA-256 hex digest of the canonicalized JSON representation."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
