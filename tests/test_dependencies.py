# tests/test_dependencies.py
import pytest
from zndraw_joblib.dependencies import (
    get_db_session,
    get_current_identity,
    get_is_admin,
    get_settings,
)
from zndraw_joblib.settings import JobLibSettings


@pytest.mark.asyncio
async def test_get_db_session_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        async for _ in get_db_session():
            pass


@pytest.mark.asyncio
async def test_get_current_identity_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        await get_current_identity()


@pytest.mark.asyncio
async def test_get_is_admin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        await get_is_admin()


def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, JobLibSettings)
