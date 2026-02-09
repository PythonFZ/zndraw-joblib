# src/zndraw_joblib/dependencies.py
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Callable

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from zndraw_auth.db import get_session_maker
from zndraw_auth.settings import AuthSettings, get_auth_settings

from zndraw_socketio import AsyncServerWrapper

from zndraw_joblib.registry import InternalRegistry
from zndraw_joblib.settings import JobLibSettings


def get_settings() -> JobLibSettings:
    """Create a settings instance from environment variables.

    Override this dependency in tests to inject custom settings.
    """
    return JobLibSettings()


async def get_db_lock(request: Request) -> asyncio.Lock:
    """Return the database lock from app.state.

    The lock must be initialized via the joblib_lifespan context manager.
    """
    return request.app.state.db_lock


def get_async_session_maker(
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> async_sessionmaker[AsyncSession]:
    """Return the async session maker for the configured database.

    Override this single dependency to redirect all joblib database access
    to a different engine (e.g., test in-memory SQLite).
    """
    return get_session_maker(auth_settings.database_url)


async def get_session_factory(
    session_maker: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_async_session_maker)
    ],
) -> Callable:
    """Returns a factory that creates short-lived async sessions on demand.

    Use this when you need multiple independent sessions within a single
    request (e.g., long-polling loops where holding a session open is wasteful).
    """

    @asynccontextmanager
    async def create_session():
        async with session_maker() as session:
            yield session

    return create_session


async def get_locked_async_session(
    session_maker: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_async_session_maker)
    ],
    joblib_settings: Annotated[JobLibSettings, Depends(get_settings)],
    db_lock: Annotated[asyncio.Lock, Depends(get_db_lock)],
) -> AsyncGenerator[AsyncSession, None]:
    """Session dependency that optionally acquires a lock for SQLite compatibility.

    When enable_db_lock=True (default), this ensures only one request accesses
    the database at a time, preventing session state corruption with SQLite's
    single-writer model.

    For PostgreSQL deployments, set ZNDRAW_JOBLIB_ENABLE_DB_LOCK=false to
    disable locking and allow full concurrency.

    If the lock cannot be acquired within db_lock_timeout_seconds, returns 503.
    """
    if joblib_settings.enable_db_lock:
        try:
            await asyncio.wait_for(
                db_lock.acquire(),
                timeout=joblib_settings.db_lock_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database busy, please try again later",
            )
        try:
            async with session_maker() as session:
                yield session
        finally:
            db_lock.release()
    else:
        async with session_maker() as session:
            yield session


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
