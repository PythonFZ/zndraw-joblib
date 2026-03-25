# src/zndraw_joblib/settings.py
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class JobLibSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZNDRAW_JOBLIB_",
        pyproject_toml_table_header=("tool", "zndraw", "joblib"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add pyproject.toml as a configuration source.

        Priority (highest to lowest): init, env vars, pyproject.toml.
        """
        from pydantic_settings import PyprojectTomlConfigSettingsSource  # noqa: PLC0415

        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )

    allowed_categories: list[str] = ["modifiers", "selections", "analysis"]
    worker_timeout_seconds: int = 60
    sweeper_interval_seconds: int = 30
    long_poll_max_wait_seconds: int = 60

    # Task claim retry settings (for handling concurrent claim contention)
    claim_max_attempts: int = 10
    claim_base_delay_seconds: float = 0.01  # 10ms

    # Internal taskiq worker settings
    internal_task_timeout_seconds: int = 3600  # 1 hour

    # Provider settings
    allowed_provider_categories: list[str] | None = None  # None = unrestricted
    provider_result_ttl_seconds: int = 300
    provider_inflight_ttl_seconds: int = 30
    provider_long_poll_default_seconds: int = 5
    provider_long_poll_max_seconds: int = 30
