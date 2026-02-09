# tests/test_dependencies.py
from zndraw_joblib.dependencies import get_settings
from zndraw_joblib.settings import JobLibSettings


def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, JobLibSettings)


def test_get_settings_returns_fresh_instance():
    """get_settings returns a new instance each call (overridable in tests)."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 == settings2
    assert settings1 is not settings2


def test_get_internal_registry_import():
    from zndraw_joblib.dependencies import get_internal_registry

    assert callable(get_internal_registry)


async def test_get_tsio_returns_none_by_default():
    """Default tsio dependency returns None (no socketio configured)."""
    from zndraw_joblib.dependencies import get_tsio

    result = await get_tsio()
    assert result is None
