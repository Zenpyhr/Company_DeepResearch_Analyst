from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import requests

from app.config import get_settings
from app.logging import get_logger
from schemas.models import Company


logger = get_logger(__name__)

# Keep a small local registry for polished defaults, but allow runtime lookup for new tickers.
COMPANY_REGISTRY: dict[str, dict[str, str]] = {
    "NVDA": {
        "company_name": "NVIDIA Corporation",
        "cik": "0001045810",
        "industry": "Semiconductors",
        "website": "https://www.nvidia.com",
    }
}

SEC_TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


@lru_cache(maxsize=1)
def _sec_company_lookup() -> dict[str, dict[str, str]]:
    response = requests.get(SEC_TICKER_LOOKUP_URL, headers=_sec_headers(), timeout=30)
    response.raise_for_status()
    payload = response.json()

    lookup: dict[str, dict[str, str]] = {}
    for row in payload.values():
        ticker = str(row.get("ticker") or "").strip().upper()
        cik = str(row.get("cik_str") or "").strip()
        company_name = str(row.get("title") or "").strip()
        if not ticker or not cik or not company_name:
            continue
        lookup[ticker] = {
            "company_name": company_name,
            "cik": cik.zfill(10),
        }
    return lookup


def _dynamic_company_metadata(ticker: str) -> dict[str, str] | None:
    normalized = ticker.strip().upper()
    if normalized in COMPANY_REGISTRY:
        return COMPANY_REGISTRY[normalized]

    try:
        sec_lookup = _sec_company_lookup()
    except Exception as exc:  # pragma: no cover - exercised at runtime
        logger.warning("Failed to fetch SEC ticker lookup while resolving %s: %s", normalized, exc)
        return None

    metadata = sec_lookup.get(normalized)
    if metadata:
        COMPANY_REGISTRY[normalized] = metadata
    return metadata


def resolve_company(ticker: str) -> Company:
    normalized = ticker.strip().upper()
    metadata = _dynamic_company_metadata(normalized)
    if metadata is None:
        raise ValueError(
            f"Ticker {ticker!r} is not configured locally and could not be resolved from SEC company metadata."
        )

    return Company(
        ticker=normalized,
        company_name=metadata["company_name"],
        cik=metadata["cik"],
        industry=metadata.get("industry"),
        website=metadata.get("website"),
        last_refreshed_at=datetime.now(timezone.utc),
    )
