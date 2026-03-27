from __future__ import annotations

from datetime import datetime, timezone

from app.config import get_settings
from app.logging import get_logger
from app.paths import ensure_company_dirs
from schemas.models import Company
from storage.database import initialize_database
from storage.repositories import upsert_company


logger = get_logger(__name__)


def bootstrap_storage() -> None:
    # This is a one-command local setup for the structured storage layer.
    # It creates the company folders, initializes SQLite, and seeds NVIDIA.
    settings = get_settings()
    ensure_company_dirs(settings.default_ticker)
    initialize_database()
    upsert_company(
        Company(
            ticker=settings.default_ticker,
            company_name="NVIDIA Corporation",
            industry="Semiconductors",
            website="https://www.nvidia.com",
            last_refreshed_at=datetime.now(timezone.utc),
        )
    )
    logger.info("Storage bootstrap complete for %s", settings.default_ticker)


if __name__ == "__main__":
    bootstrap_storage()
