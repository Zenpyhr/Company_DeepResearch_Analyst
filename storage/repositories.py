from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from schemas.models import Chunk, Company, Event, FinancialMetric, MarketRecord, Source
from storage.database import connection_scope


# Small helpers keep timestamp and metadata handling consistent across tables.
def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def upsert_company(company: Company) -> None:
    # Upsert lets us refresh company metadata without creating duplicate rows.
    with connection_scope() as connection:
        connection.execute(
            """
            INSERT INTO companies (ticker, company_name, cik, industry, website, last_refreshed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                cik = excluded.cik,
                industry = excluded.industry,
                website = excluded.website,
                last_refreshed_at = excluded.last_refreshed_at,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (
                company.ticker,
                company.company_name,
                company.cik,
                company.industry,
                company.website,
                _to_iso(company.last_refreshed_at),
            ),
        )


def reset_company_data(company_ticker: str) -> None:
    # Make refreshes idempotent for one company by clearing dependent rows first.
    # We keep the company identity record, but rebuild the fetched/derived layers.
    with connection_scope() as connection:
        connection.execute("DELETE FROM chunks WHERE company_ticker = ?", (company_ticker,))
        connection.execute("DELETE FROM events WHERE company_ticker = ?", (company_ticker,))
        connection.execute("DELETE FROM financial_metrics WHERE company_ticker = ?", (company_ticker,))
        connection.execute("DELETE FROM market_data WHERE company_ticker = ?", (company_ticker,))
        connection.execute("DELETE FROM sources WHERE company_ticker = ?", (company_ticker,))


def delete_chunks_for_company(company_ticker: str, source_types: list[str] | None = None) -> None:
    # Reprocessing documents should replace old chunk rows rather than append duplicates.
    with connection_scope() as connection:
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            connection.execute(
                f"DELETE FROM chunks WHERE company_ticker = ? AND source_type IN ({placeholders})",
                [company_ticker, *source_types],
            )
        else:
            connection.execute("DELETE FROM chunks WHERE company_ticker = ?", (company_ticker,))


def insert_source(source: Source) -> int:
    # Every raw filing, press release, or dataset should be registered as a source first.
    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sources (
                company_ticker, source_type, title, source_url, published_at, raw_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source.company_ticker,
                source.source_type,
                source.title,
                source.source_url,
                _to_iso(source.published_at),
                source.raw_path,
                _to_json(source.metadata_json),
            ),
        )
        return int(cursor.lastrowid)


def fetch_sources(company_ticker: str, source_types: list[str] | None = None) -> list[dict[str, Any]]:
    # Simple query helper used by later processing steps such as chunk generation.
    query = "SELECT * FROM sources WHERE company_ticker = ?"
    parameters: list[Any] = [company_ticker]
    if source_types:
        placeholders = ",".join("?" for _ in source_types)
        query += f" AND source_type IN ({placeholders})"
        parameters.extend(source_types)
    query += " ORDER BY id"

    with connection_scope() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def insert_chunk(chunk: Chunk) -> int:
    # Chunk text is stored here so we can retrieve the actual passage after FAISS finds a match.
    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO chunks (
                company_ticker, source_id, source_type, title, source_url, published_at,
                chunk_text, chunk_order, embedding_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                chunk.company_ticker,
                chunk.source_id,
                chunk.source_type,
                chunk.title,
                chunk.source_url,
                _to_iso(chunk.published_at),
                chunk.chunk_text,
                chunk.chunk_order,
                chunk.embedding_id,
                _to_json(chunk.metadata_json),
            ),
        )
        return int(cursor.lastrowid)


def insert_financial_metric(metric: FinancialMetric) -> int:
    # This is part of the quantitative layer used for questions like revenue or margin trends.
    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO financial_metrics (
                company_ticker, fiscal_period, metric_name, metric_value, unit,
                source_url, as_of_date, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                metric.company_ticker,
                metric.fiscal_period,
                metric.metric_name,
                metric.metric_value,
                metric.unit,
                metric.source_url,
                _to_iso(metric.as_of_date),
                _to_json(metric.metadata_json),
            ),
        )
        return int(cursor.lastrowid)


def upsert_market_record(record: MarketRecord) -> None:
    # Market rows are keyed by company + date so refreshes can safely overwrite old values.
    with connection_scope() as connection:
        connection.execute(
            """
            INSERT INTO market_data (
                company_ticker, date, open, high, low, close, volume, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                metadata_json = excluded.metadata_json;
            """,
            (
                record.company_ticker,
                _to_iso(record.date),
                record.open,
                record.high,
                record.low,
                record.close,
                record.volume,
                _to_json(record.metadata_json),
            ),
        )


def insert_event(event: Event) -> int:
    # Events store major extracted developments that help with timeline-style reasoning.
    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO events (
                company_ticker, timestamp, source_type, event_type, event_subtype, description,
                evidence_text, source_url, confidence, verification_status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event.company_ticker,
                _to_iso(event.timestamp),
                event.source_type,
                event.event_type,
                event.event_subtype,
                event.description,
                event.evidence_text,
                event.source_url,
                event.confidence,
                event.verification_status,
                _to_json(event.metadata_json),
            ),
        )
        return int(cursor.lastrowid)
