# tests/test_dependencies.py
from zndraw_joblib.dependencies import get_settings
from zndraw_joblib.settings import JobLibSettings


def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, JobLibSettings)


def test_get_settings_is_cached():
    """get_settings should return the same instance (cached)."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_get_internal_registry_import():
    from zndraw_joblib.dependencies import get_internal_registry
    assert callable(get_internal_registry)
