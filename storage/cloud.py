from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.logging import get_logger


logger = get_logger(__name__)

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - cloud dependencies are optional during local dev
    storage = None  # type: ignore[assignment]


def upload_artifact(path: str | Path, company_ticker: str) -> str | None:
    settings = get_settings()
    if settings.storage_backend != "gcs" or not settings.cloud_storage_bucket or storage is None:
        return None

    local_path = Path(path)
    if not local_path.exists():
        return None

    client = storage.Client(project=settings.gcp_project_id or None)
    bucket = client.bucket(settings.cloud_storage_bucket)
    blob_name = f"companies/{company_ticker.lower()}/artifacts/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{settings.cloud_storage_bucket}/{blob_name}"
    logger.info("Uploaded artifact to %s", uri)
    return uri
