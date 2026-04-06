from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from storage.database import connection_scope


def _loads_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def fetch_document_chunks(company_ticker: str, source_types: list[str] | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM chunks WHERE company_ticker = ?"
    parameters: list[Any] = [company_ticker]
    if source_types:
        placeholders = ",".join("?" for _ in source_types)
        query += f" AND source_type IN ({placeholders})"
        parameters.extend(source_types)
    query += " ORDER BY published_at DESC, source_id, chunk_order"

    with connection_scope() as connection:
        rows = connection.execute(query, parameters).fetchall()

    records = [dict(row) for row in rows]
    for record in records:
        record["metadata_json"] = _loads_metadata(record.get("metadata_json"))
    return records


def search_document_chunks(
    company_ticker: str,
    question: str,
    top_k: int = 5,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 2]
    chunks = fetch_document_chunks(company_ticker, source_types=source_types)
    scored: list[tuple[float, dict[str, Any]]] = []

    for chunk in chunks:
        haystack_parts = [
            str(chunk.get("title") or ""),
            str(chunk.get("chunk_text") or ""),
            str(chunk.get("source_type") or ""),
        ]
        metadata = chunk.get("metadata_json", {})
        haystack_parts.extend([str(metadata.get("section_label") or ""), str(metadata.get("section_title") or "")])
        haystack = " ".join(haystack_parts).lower()
        score = 0.0
        for token in tokens:
            count = haystack.count(token)
            if count:
                score += count
                if token in str(metadata.get("section_label") or "").lower():
                    score += 1.5
                if token in str(metadata.get("section_title") or "").lower():
                    score += 1.0
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].get("chunk_order", 0)))
    return [chunk for _, chunk in scored[:top_k]]


def fetch_financial_metrics(
    company_ticker: str,
    metric_names: list[str] | None = None,
    periods: list[str] | None = None,
    limit_per_metric: int | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM financial_metrics WHERE company_ticker = ?"
    parameters: list[Any] = [company_ticker]
    if metric_names:
        placeholders = ",".join("?" for _ in metric_names)
        query += f" AND metric_name IN ({placeholders})"
        parameters.extend(metric_names)
    if periods:
        placeholders = ",".join("?" for _ in periods)
        query += f" AND fiscal_period IN ({placeholders})"
        parameters.extend(periods)
    query += " ORDER BY metric_name, as_of_date DESC, fiscal_period DESC"

    with connection_scope() as connection:
        rows = connection.execute(query, parameters).fetchall()

    records = [dict(row) for row in rows]
    for record in records:
        record["metadata_json"] = _loads_metadata(record.get("metadata_json"))

    if not limit_per_metric:
        return records

    limited: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        metric_name = str(record.get("metric_name", ""))
        if counts[metric_name] >= limit_per_metric:
            continue
        limited.append(record)
        counts[metric_name] += 1
    return limited


def fetch_market_data(
    company_ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM market_data WHERE company_ticker = ?"
    parameters: list[Any] = [company_ticker]
    if start_date:
        query += " AND date >= ?"
        parameters.append(start_date)
    if end_date:
        query += " AND date <= ?"
        parameters.append(end_date)
    query += " ORDER BY date DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    with connection_scope() as connection:
        rows = connection.execute(query, parameters).fetchall()

    records = [dict(row) for row in rows]
    for record in records:
        record["metadata_json"] = _loads_metadata(record.get("metadata_json"))
    return records


def derive_market_window(company_ticker: str, anchor_dates: list[str], days_before: int = 3, days_after: int = 3) -> list[dict[str, Any]]:
    if not anchor_dates:
        return fetch_market_data(company_ticker, limit=20)

    windows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for raw_anchor in anchor_dates:
        try:
            anchor = datetime.fromisoformat(raw_anchor.replace("Z", "+00:00"))
        except ValueError:
            try:
                anchor = datetime.fromisoformat(raw_anchor[:10])
            except ValueError:
                continue

        start_date = (anchor - timedelta(days=days_before)).date().isoformat()
        end_date = (anchor + timedelta(days=days_after)).date().isoformat()
        for row in fetch_market_data(company_ticker, start_date=start_date, end_date=end_date):
            date_value = str(row.get("date", ""))[:10]
            if date_value in seen_dates:
                continue
            seen_dates.add(date_value)
            windows.append(row)

    windows.sort(key=lambda row: row.get("date", ""))
    return windows


def infer_anchor_dates_from_metrics(metric_rows: list[dict[str, Any]]) -> list[str]:
    anchors: list[str] = []
    for row in metric_rows:
        as_of_date = row.get("as_of_date")
        if as_of_date:
            anchors.append(str(as_of_date))
    return anchors


def safe_pct_change(current: float | int | None, previous: float | int | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return ((float(current) - float(previous)) / abs(float(previous))) * 100.0


def safe_stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
