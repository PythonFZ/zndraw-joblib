# src/zndraw_joblib/dependencies.py
from functools import lru_cache
from typing import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession

from zndraw_joblib.settings import JobLibSettings


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Stub: Override via app.dependency_overrides."""
    raise NotImplementedError("Dependency not configured")
    yield  # type: ignore


async def get_current_identity() -> str:
    """Stub: Override via app.dependency_overrides. Returns user/worker ID from JWT."""
    raise NotImplementedError("Dependency not configured")


async def get_is_admin() -> bool:
    """Stub: Override via app.dependency_overrides. Returns True if user is admin."""
    raise NotImplementedError("Dependency not configured")


@lru_cache
def get_settings() -> JobLibSettings:
    """Returns cached settings instance."""
    return JobLibSettings()
