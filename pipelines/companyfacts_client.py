from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests

from app.config import get_settings
from app.logging import get_logger
from app.paths import raw_data_dir
from schemas.models import Company, FinancialMetric, Source


logger = get_logger(__name__)
SEC_DATA_BASE = "https://data.sec.gov"

# We map a small set of business-friendly metric names to possible SEC XBRL tags.
# Many companies report the same concept under slightly different tags, so we merge candidates.
METRIC_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "cash_and_cash_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
}

PREFERRED_FORMS = {"10-K", "10-Q"}


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _companyfacts_dir(ticker: str) -> Path:
    path = raw_data_dir(ticker) / "companyfacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_companyfacts(company: Company) -> tuple[Source, list[FinancialMetric]]:
    padded_cik = company.cik.zfill(10)
    url = f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{padded_cik}.json"
    logger.info("Fetching CompanyFacts for %s from %s", company.ticker, url)
    response = requests.get(url, headers=_sec_headers(), timeout=30)
    response.raise_for_status()
    payload = response.json()

    raw_path = _companyfacts_dir(company.ticker) / f"{company.ticker.lower()}_companyfacts.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    source = Source(
        company_ticker=company.ticker,
        source_type="companyfacts",
        title=f"{company.ticker} SEC CompanyFacts",
        source_url=url,
        raw_path=str(raw_path),
        metadata_json={"entity_name": payload.get("entityName")},
    )

    metrics = extract_key_metrics(company, payload, url)
    return source, metrics


def extract_key_metrics(company: Company, payload: dict, source_url: str) -> list[FinancialMetric]:
    facts = payload.get("facts", {}).get("us-gaap", {})
    extracted: list[FinancialMetric] = []

    for metric_name, candidate_tags in METRIC_TAGS.items():
        selected_entries = _collect_metric_entries(facts, candidate_tags)

        for entry_info in selected_entries:
            entry = entry_info["entry"]
            unit_name = entry_info["unit"]
            selected_tag = entry_info["tag"]
            fiscal_period = f"{entry.get('fy', 'unknown')}-{entry.get('fp', 'NA')}"
            value = entry.get("val")
            if value is None:
                continue

            extracted.append(
                FinancialMetric(
                    company_ticker=company.ticker,
                    fiscal_period=fiscal_period,
                    metric_name=metric_name,
                    metric_value=float(value),
                    unit=unit_name,
                    source_url=source_url,
                    as_of_date=_parse_date(entry.get("end") or entry.get("filed")),
                    metadata_json={
                        "xbrl_tag": selected_tag,
                        "form": entry.get("form"),
                        "frame": entry.get("frame"),
                        "filed": entry.get("filed"),
                        "start": entry.get("start"),
                        "end": entry.get("end"),
                    },
                )
            )

    return extracted


def _collect_metric_entries(facts: dict, candidate_tags: list[str], max_entries: int = 6) -> list[dict]:
    # Merge entries from all possible XBRL tags, then keep the most recent unique periods.
    merged: list[dict] = []
    for tag in candidate_tags:
        tag_payload = facts.get(tag)
        if not tag_payload:
            continue
        for unit_name, entries in tag_payload.get("units", {}).items():
            for entry in entries:
                if not _is_valid_entry(entry):
                    continue
                merged.append({"tag": tag, "unit": unit_name, "entry": entry})

    merged.sort(
        key=lambda item: (
            _parse_date(item["entry"].get("end") or item["entry"].get("filed")) or datetime.min,
            item["entry"].get("fy", 0),
            item["entry"].get("filed", ""),
        ),
        reverse=True,
    )

    deduped: list[dict] = []
    seen_periods: set[tuple[int | str, str]] = set()
    for item in merged:
        entry = item["entry"]
        period_key = (entry.get("fy", "unknown"), entry.get("fp", "NA"))
        if period_key in seen_periods:
            continue
        seen_periods.add(period_key)
        deduped.append(item)
        if len(deduped) >= max_entries:
            break
    return deduped


def _is_valid_entry(entry: dict) -> bool:
    # Prefer standard quarterly/annual facts filed in 10-K / 10-Q.
    return bool(entry.get("fp") and entry.get("fy") and entry.get("form") in PREFERRED_FORMS)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
