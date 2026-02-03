# src/zndraw_joblib/dependencies.py
from functools import lru_cache

from zndraw_joblib.settings import JobLibSettings


@lru_cache
def get_settings() -> JobLibSettings:
    """Returns cached settings instance."""
    return JobLibSettings()
