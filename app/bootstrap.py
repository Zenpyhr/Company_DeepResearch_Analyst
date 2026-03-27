from __future__ import annotations

from app.config import get_settings
from app.logging import get_logger
from app.paths import ensure_company_dirs


logger = get_logger(__name__)


def bootstrap_project() -> None:
    settings = get_settings()
    ensure_company_dirs(settings.default_ticker)
    logger.info("Project bootstrap complete for %s", settings.default_ticker)
    logger.info("Data directory: %s", settings.data_dir.resolve())
    logger.info("SQLite path: %s", settings.sqlite_db_path.resolve())


if __name__ == "__main__":
    bootstrap_project()
