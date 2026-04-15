from __future__ import annotations

from typing import Any

from schemas.models import SchemaColumnProfile, SchemaContext, SchemaTableProfile
from storage.database import connection_scope


TABLE_DESCRIPTIONS: dict[str, str] = {
    "financial_metrics": "Structured company financial facts by ticker, metric, period, and as-of date.",
    "market_data": "Daily stock market history with open, high, low, close, and volume.",
    "chunks": "Text chunks cut from filings and press releases for qualitative retrieval.",
    "sources": "Original source documents and datasets collected for each company.",
    "companies": "Tracked company records with ticker, name, and refresh timestamp.",
}

COLUMN_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("financial_metrics", "metric_name"): "Business metric name such as revenue or net_income.",
    ("financial_metrics", "metric_value"): "Numeric value for the selected business metric.",
    ("financial_metrics", "fiscal_period"): "Reported fiscal period label such as 2026-Q3 or 2026-FY.",
    ("financial_metrics", "as_of_date"): "Date tied to the reported metric row.",
    ("market_data", "date"): "Trading date for the market row.",
    ("market_data", "close"): "Closing stock price for the trading day.",
    ("market_data", "volume"): "Trading volume for the day.",
    ("chunks", "chunk_text"): "Retrieved text excerpt from a filing or press release.",
    ("chunks", "source_type"): "Qualitative source type, usually sec_filing or press_release.",
    ("sources", "source_type"): "Collected source category such as sec_filing, press_release, companyfacts, or market_data.",
    ("sources", "published_at"): "Publication or filing date for the source record.",
}

DOMAIN_GLOSSARY: dict[str, str] = {
    "top line": "Use financial_metrics.metric_name = 'revenue'.",
    "bottom line": "Use financial_metrics.metric_name = 'net_income'.",
    "earnings": "Often maps to net_income or eps_diluted in financial_metrics.",
    "eps": "Use financial_metrics.metric_name = 'eps_diluted'.",
    "stock price": "Use market_data.close.",
    "trading volume": "Use market_data.volume.",
    "filings": "Use sources or chunks where source_type = 'sec_filing'.",
    "press releases": "Use sources or chunks where source_type = 'press_release'.",
    "risk factors": "Often appears in chunks.metadata_json or chunk_text from filing sections.",
}

DEFAULT_SQL_TABLES = ["financial_metrics", "market_data", "chunks", "sources"]


def _sample_values_for_column(table_name: str, column_name: str, *, limit: int = 5) -> list[str]:
    query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {int(limit)}"
    with connection_scope() as connection:
        rows = connection.execute(query).fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _enum_values_for_column(table_name: str, column_name: str, *, max_values: int = 50) -> list[str]:
    query = (
        f"SELECT DISTINCT {column_name} FROM {table_name} "
        f"WHERE {column_name} IS NOT NULL LIMIT {int(max_values) + 1}"
    )
    with connection_scope() as connection:
        rows = connection.execute(query).fetchall()
    values = [str(row[0]) for row in rows if row and row[0] is not None]
    return values if 0 < len(values) <= max_values else []


def _table_columns(table_name: str) -> list[SchemaColumnProfile]:
    with connection_scope() as connection:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()

    profiles: list[SchemaColumnProfile] = []
    for row in rows:
        column_name = str(row["name"])
        enum_values = _enum_values_for_column(table_name, column_name)
        profiles.append(
            SchemaColumnProfile(
                table_name=table_name,
                column_name=column_name,
                data_type=str(row["type"] or "TEXT"),
                description=COLUMN_DESCRIPTIONS.get((table_name, column_name)),
                sample_values=_sample_values_for_column(table_name, column_name),
                is_enum=bool(enum_values),
                enum_values=enum_values,
            )
        )
    return profiles


def _relevant_tables_for_question(question: str, tables: list[str]) -> list[str]:
    lowered = question.lower()
    selected: list[str] = []

    if any(term in lowered for term in ["revenue", "income", "profit", "margin", "eps", "metric", "quarter", "financial"]):
        selected.append("financial_metrics")
    if any(term in lowered for term in ["stock", "price", "market", "volume", "return", "trading"]):
        selected.append("market_data")
    if any(term in lowered for term in ["filing", "press", "release", "risk", "chunk", "section"]):
        selected.extend(["chunks", "sources"])
    if any(term in lowered for term in ["count", "how many", "breakdown", "distribution", "source type"]):
        selected.append("sources")
    return [table for table in dict.fromkeys(selected) if table in tables] or tables


def _glossary_hits_for_question(question: str) -> dict[str, str]:
    lowered = question.lower()
    return {term: meaning for term, meaning in DOMAIN_GLOSSARY.items() if term in lowered}


def build_schema_context(question: str, ticker: str, allowed_tables: list[str] | None = None) -> SchemaContext:
    tables = allowed_tables or DEFAULT_SQL_TABLES
    relevant_tables = _relevant_tables_for_question(question, tables)
    table_profiles = [
        SchemaTableProfile(
            table_name=table_name,
            description=TABLE_DESCRIPTIONS.get(table_name),
            columns=_table_columns(table_name),
        )
        for table_name in relevant_tables
    ]
    return SchemaContext(
        ticker=ticker,
        question=question,
        relevant_tables=relevant_tables,
        tables=table_profiles,
        glossary_hits=_glossary_hits_for_question(question),
        notes=["Use company_ticker filtering whenever the selected table contains that column."],
    )

