# tests/test_settings.py
from zndraw_joblib.settings import JobLibSettings


def test_default_settings():
    settings = JobLibSettings()
    assert settings.allowed_categories == ["modifiers", "selections", "analysis"]
    assert settings.worker_timeout_seconds == 60
    assert settings.sweeper_interval_seconds == 30
    assert settings.long_poll_max_wait_seconds == 120
    assert settings.enable_db_lock is True
    assert settings.db_lock_timeout_seconds == 30.0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ZNDRAW_JOBLIB_ALLOWED_CATEGORIES", '["custom"]')
    monkeypatch.setenv("ZNDRAW_JOBLIB_WORKER_TIMEOUT_SECONDS", "120")
    settings = JobLibSettings()
    assert settings.allowed_categories == ["custom"]
    assert settings.worker_timeout_seconds == 120


def test_db_lock_settings_from_env(monkeypatch):
    """Test that database lock settings can be configured via environment."""
    monkeypatch.setenv("ZNDRAW_JOBLIB_ENABLE_DB_LOCK", "false")
    monkeypatch.setenv("ZNDRAW_JOBLIB_DB_LOCK_TIMEOUT_SECONDS", "60.0")
    settings = JobLibSettings()
    assert settings.enable_db_lock is False
    assert settings.db_lock_timeout_seconds == 60.0
