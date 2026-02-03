# src/zndraw_joblib/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class JobLibSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZNDRAW_JOBLIB_")

    allowed_categories: list[str] = ["modifiers", "selections", "analysis"]
    worker_timeout_seconds: int = 60
    sweeper_interval_seconds: int = 30
    long_poll_max_wait_seconds: int = 120

    # Task claim retry settings (for handling concurrent claim contention)
    claim_max_attempts: int = 10
    claim_base_delay_seconds: float = 0.01  # 10ms

    # Database locking settings (for SQLite compatibility)
    # Set enable_db_lock=False for PostgreSQL deployments
    enable_db_lock: bool = True
    db_lock_timeout_seconds: float = 30.0
