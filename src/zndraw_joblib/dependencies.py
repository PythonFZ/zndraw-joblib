# src/zndraw_joblib/dependencies.py
import asyncio
from functools import lru_cache
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from zndraw_auth import get_async_session
from zndraw_auth.settings import AuthSettings, get_auth_settings

from zndraw_joblib.settings import JobLibSettings

# Global lock for serializing database access (SQLite compatibility)
_db_lock = asyncio.Lock()


@lru_cache
def get_settings() -> JobLibSettings:
    """Returns cached settings instance."""
    return JobLibSettings()


async def get_locked_async_session(
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
    joblib_settings: Annotated[JobLibSettings, Depends(get_settings)],
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
                _db_lock.acquire(),
                timeout=joblib_settings.db_lock_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database busy, please try again later",
            )
        try:
            async for session in get_async_session(auth_settings):
                yield session
        finally:
            _db_lock.release()
    else:
        # PostgreSQL mode: no locking needed
        async for session in get_async_session(auth_settings):
            yield session
