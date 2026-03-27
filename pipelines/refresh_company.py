from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.logging import get_logger
from pipelines.company_registry import resolve_company
from pipelines.companyfacts_client import fetch_companyfacts
from pipelines.market_data import fetch_market_history
from pipelines.press_releases import fetch_press_releases
from pipelines.sec_client import fetch_recent_filings
from storage.bootstrap import bootstrap_storage
from storage.repositories import (
    insert_financial_metric,
    insert_source,
    reset_company_data,
    upsert_company,
    upsert_market_record,
)


logger = get_logger(__name__)


def refresh_company_data(ticker: str) -> dict[str, int]:
    # End-to-end refresh entrypoint for the selected company.
    # For MVP we are focused on one company at a time, so this pipeline updates that local knowledge base.
    bootstrap_storage()
    company = resolve_company(ticker)
    company.last_refreshed_at = datetime.now(timezone.utc)
    upsert_company(company)
    reset_company_data(company.ticker)

    filings = fetch_recent_filings(company)
    for artifact in filings:
        insert_source(artifact.source)

    companyfacts_source, metrics = fetch_companyfacts(company)
    insert_source(companyfacts_source)
    for metric in metrics:
        insert_financial_metric(metric)

    market_source, market_records = fetch_market_history(company)
    insert_source(market_source)
    for record in market_records:
        upsert_market_record(record)

    press_releases = fetch_press_releases(company)
    for release in press_releases:
        insert_source(release)

    summary = {
        "filings": len(filings),
        "financial_metrics": len(metrics),
        "market_rows": len(market_records),
        "press_releases": len(press_releases),
    }
    logger.info("Refresh complete for %s: %s", company.ticker, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the local knowledge base for one company.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to refresh. MVP currently supports NVDA.")
    args = parser.parse_args()
    refresh_company_data(args.ticker)


if __name__ == "__main__":
    main()
