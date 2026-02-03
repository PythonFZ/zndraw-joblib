# tests/test_settings.py
from zndraw_joblib.settings import JobLibSettings


def test_default_settings():
    settings = JobLibSettings()
    assert settings.allowed_categories == ["modifiers", "selections", "analysis"]
    assert settings.worker_timeout_seconds == 60
    assert settings.sweeper_interval_seconds == 30
    assert settings.long_poll_max_wait_seconds == 120


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ZNDRAW_JOBLIB_ALLOWED_CATEGORIES", '["custom"]')
    monkeypatch.setenv("ZNDRAW_JOBLIB_WORKER_TIMEOUT_SECONDS", "120")
    settings = JobLibSettings()
    assert settings.allowed_categories == ["custom"]
    assert settings.worker_timeout_seconds == 120
