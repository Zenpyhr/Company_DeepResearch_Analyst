from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import get_settings
from app.logging import get_logger
from app.paths import raw_data_dir
from schemas.models import Company, Source


logger = get_logger(__name__)
SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
DEFAULT_FORMS = ("10-K", "10-Q", "8-K")


@dataclass(slots=True)
class FilingArtifact:
    source: Source
    html_path: Path
    text_path: Path


def _sec_headers(url: str) -> dict[str, str]:
    # SEC asks for a descriptive User-Agent so they can identify traffic.
    # We avoid forcing the wrong Host header because submissions live on data.sec.gov
    # while filing documents are often served from www.sec.gov.
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": urlparse(url).netloc,
    }


def _filings_dir(ticker: str) -> Path:
    path = raw_data_dir(ticker) / "sec_filings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_submissions(company: Company) -> dict:
    padded_cik = company.cik.zfill(10)
    url = f"{SEC_DATA_BASE}/submissions/CIK{padded_cik}.json"
    logger.info("Fetching SEC submissions for %s from %s", company.ticker, url)
    response = requests.get(url, headers=_sec_headers(url), timeout=30)
    response.raise_for_status()
    return response.json()


def _clean_filing_text(html: str) -> str:
    # SEC filings are HTML-heavy, so we reduce them to readable text for later chunking.
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _candidate_filing_urls(company: Company, accession_compact: str, primary_doc: str) -> list[str]:
    # SEC metadata is usually enough, but some filings are easiest to resolve through the
    # filing index JSON page. We try the direct primary document first, then a small fallback set.
    base_path = f"/Archives/edgar/data/{int(company.cik)}/{accession_compact}"
    return [
        f"{SEC_BASE}{base_path}/{primary_doc}",
        f"{SEC_BASE}{base_path}/index.html",
    ]


def _fetch_filing_response(company: Company, accession_compact: str, primary_doc: str) -> tuple[str, str]:
    for candidate_url in _candidate_filing_urls(company, accession_compact, primary_doc):
        logger.info("Trying SEC filing URL for %s: %s", company.ticker, candidate_url)
        response = requests.get(candidate_url, headers=_sec_headers(candidate_url), timeout=30)
        if response.status_code == 200 and response.text.strip():
            return candidate_url, response.text

    index_url = f"{SEC_BASE}/Archives/edgar/data/{int(company.cik)}/{accession_compact}/index.json"
    logger.info("Primary filing URL failed, checking filing index %s", index_url)
    index_response = requests.get(index_url, headers=_sec_headers(index_url), timeout=30)
    index_response.raise_for_status()
    index_payload = index_response.json()

    directory = index_payload.get("directory", {})
    item_names = [item.get("name", "") for item in directory.get("item", [])]
    likely_docs = [
        name for name in item_names
        if name.lower().endswith((".htm", ".html", ".txt")) and not name.lower().startswith("primary_doc")
    ]
    if not likely_docs:
        raise ValueError(f"No usable filing documents found in SEC index for accession {accession_compact}")

    resolved_doc = likely_docs[0]
    resolved_url = f"{SEC_BASE}/Archives/edgar/data/{int(company.cik)}/{accession_compact}/{resolved_doc}"
    logger.info("Resolved fallback SEC filing URL for %s: %s", company.ticker, resolved_url)
    response = requests.get(resolved_url, headers=_sec_headers(resolved_url), timeout=30)
    response.raise_for_status()
    return resolved_url, response.text


def fetch_recent_filings(company: Company, form_types: tuple[str, ...] = DEFAULT_FORMS, limit_per_form: int = 3) -> list[FilingArtifact]:
    submissions = fetch_submissions(company)
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    per_form_count: dict[str, int] = {form: 0 for form in form_types}
    artifacts: list[FilingArtifact] = []
    filings_dir = _filings_dir(company.ticker)

    for form, accession, primary_doc, filing_date in zip(forms, accession_numbers, primary_docs, filing_dates):
        if form not in form_types:
            continue
        if per_form_count[form] >= limit_per_form:
            continue

        accession_compact = accession.replace("-", "")
        resolved_url, filing_html = _fetch_filing_response(company, accession_compact, primary_doc)

        html_path = filings_dir / f"{filing_date}_{form}_{accession_compact}.html"
        text_path = filings_dir / f"{filing_date}_{form}_{accession_compact}.txt"
        html_path.write_text(filing_html, encoding="utf-8")
        text_path.write_text(_clean_filing_text(filing_html), encoding="utf-8")

        source = Source(
            company_ticker=company.ticker,
            source_type="sec_filing",
            title=f"{company.ticker} {form} filed on {filing_date}",
            source_url=resolved_url,
            raw_path=str(text_path),
            metadata_json={
                "form_type": form,
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": primary_doc,
                "resolved_url": resolved_url,
                "company_name": submissions.get("name"),
            },
        )
        artifacts.append(FilingArtifact(source=source, html_path=html_path, text_path=text_path))
        per_form_count[form] += 1

    return artifacts
