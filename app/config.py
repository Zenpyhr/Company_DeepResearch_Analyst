from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    sqlite_db_path: Path = Field(default=Path("data/app.db"), alias="SQLITE_DB_PATH")
    default_ticker: str = Field(default="NVDA", alias="DEFAULT_TICKER")
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
