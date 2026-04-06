from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - fallback only used in stripped-down environments
    from pydantic import BaseModel

    class SettingsConfigDict(dict):
        pass

    class BaseSettings(BaseModel):
        def __init__(self, **data):
            env_data = {}
            for field_name, field_info in self.__class__.model_fields.items():
                alias = field_info.alias or field_name
                if alias in os.environ and field_name not in data:
                    env_data[field_name] = os.environ[alias]
            env_data.update(data)
            super().__init__(**env_data)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    llm_backend: str = Field(default="openai", alias="LLM_BACKEND")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    sqlite_db_path: Path = Field(default=Path("data/app.db"), alias="SQLITE_DB_PATH")
    default_ticker: str = Field(default="NVDA", alias="DEFAULT_TICKER")
    artifact_dir_name: str = Field(default="analysis", alias="ARTIFACT_DIR_NAME")
    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")
    database_backend: str = Field(default="sqlite", alias="DATABASE_BACKEND")
    cloud_storage_bucket: str = Field(default="", alias="CLOUD_STORAGE_BUCKET")
    cloud_sql_connection_name: str = Field(default="", alias="CLOUD_SQL_CONNECTION_NAME")
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    vertex_location: str = Field(default="us-central1", alias="VERTEX_LOCATION")
    # SEC asks clients to identify themselves with a descriptive User-Agent.
    sec_user_agent: str = Field(
        default="company-diligence-agent/0.1 (local development contact: research@example.com)",
        alias="SEC_USER_AGENT",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def ensure_base_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_base_dirs()
    return settings
