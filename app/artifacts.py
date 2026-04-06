from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging import get_logger
from app.paths import artifact_data_dir
from storage.cloud import upload_artifact


logger = get_logger(__name__)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_artifact_dir(ticker: str) -> Path:
    path = artifact_data_dir(ticker)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_artifact(ticker: str, stem: str, payload: dict[str, Any]) -> str:
    path = ensure_artifact_dir(ticker) / f"{stem}_{_timestamp_slug()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("Wrote JSON artifact to %s", path)
    return upload_artifact(path, ticker) or str(path)


def write_markdown_artifact(ticker: str, stem: str, content: str) -> str:
    path = ensure_artifact_dir(ticker) / f"{stem}_{_timestamp_slug()}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote markdown artifact to %s", path)
    return upload_artifact(path, ticker) or str(path)


def write_csv_artifact(ticker: str, stem: str, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None

    path = ensure_artifact_dir(ticker) / f"{stem}_{_timestamp_slug()}.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Wrote CSV artifact to %s", path)
    return upload_artifact(path, ticker) or str(path)
