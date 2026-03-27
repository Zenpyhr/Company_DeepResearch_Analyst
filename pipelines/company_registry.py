from __future__ import annotations

from datetime import datetime, timezone

from schemas.models import Company


# MVP registry: we start with NVIDIA only, but this module gives us one place
# to expand into more companies later without changing the whole pipeline.
COMPANY_REGISTRY: dict[str, dict[str, str]] = {
    "NVDA": {
        "company_name": "NVIDIA Corporation",
        "cik": "0001045810",
        "industry": "Semiconductors",
        "website": "https://www.nvidia.com",
    }
}


def resolve_company(ticker: str) -> Company:
    normalized = ticker.strip().upper()
    if normalized not in COMPANY_REGISTRY:
        raise ValueError(f"Ticker {ticker!r} is not configured in the local company registry yet.")

    metadata = COMPANY_REGISTRY[normalized]
    return Company(
        ticker=normalized,
        company_name=metadata["company_name"],
        cik=metadata["cik"],
        industry=metadata.get("industry"),
        website=metadata.get("website"),
        last_refreshed_at=datetime.now(timezone.utc),
    )
