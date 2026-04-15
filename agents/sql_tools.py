from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    class _SimpleToolWrapper:
        def __init__(self, fn):
            self._fn = fn
            self.__name__ = getattr(fn, "__name__", "tool")
            self.__doc__ = getattr(fn, "__doc__", None)

        def invoke(self, args: dict[str, Any] | None = None):
            return self._fn(**(args or {}))

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    def tool(*, args_schema=None):  # type: ignore[override]
        def decorator(fn):
            return _SimpleToolWrapper(fn)

        return decorator

from agents.llm import OptionalLLM
from app.artifacts import write_json_artifact
from prompts.sql_prompts import SQL_GENERATOR_SYSTEM_PROMPT, SQL_REPAIR_SYSTEM_PROMPT
from schemas.models import SQLQueryCandidate, SQLQueryResult
from storage.schema_catalog import DEFAULT_SQL_TABLES, build_schema_context
from storage.sql_executor import execute_readonly_sql
from storage.sql_validator import extract_table_names, validate_select_sql


LLM = OptionalLLM()

METRIC_ALIASES: dict[str, list[str]] = {
    "revenue": ["revenue", "sales", "top line"],
    "net_income": ["net income", "profit", "bottom line", "earnings"],
    "gross_profit": ["gross profit", "gross margin"],
    "eps_diluted": ["eps", "earnings per share"],
    "research_and_development": ["research", "r&d"],
    "cash_and_cash_equivalents": ["cash", "liquidity"],
}


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _context_to_prompt_text(context: dict[str, Any]) -> str:
    table_lines: list[str] = []
    for table in context.get("tables", []):
        columns = ", ".join(
            f"{column.get('column_name')} ({column.get('data_type')})"
            for column in table.get("columns", [])
        )
        table_lines.append(f"- {table.get('table_name')}: {columns}")
    glossary_hits = context.get("glossary_hits", {}) or {}
    glossary_text = "\n".join(f"- {term}: {meaning}" for term, meaning in glossary_hits.items()) or "- none"
    return (
        "Schema context:\n"
        + "\n".join(table_lines)
        + "\nGlossary hits:\n"
        + glossary_text
    )


def _metric_names_for_question(question: str) -> list[str]:
    lowered = question.lower()
    return [metric for metric, aliases in METRIC_ALIASES.items() if any(alias in lowered for alias in aliases)]


def _heuristic_sql(question: str, ticker: str, allowed_tables: list[str], max_rows: int) -> SQLQueryCandidate:
    lowered = question.lower()
    safe_ticker = _sql_quote(ticker)

    if ("how many" in lowered or "count" in lowered or "breakdown" in lowered or "distribution" in lowered) and "sources" in allowed_tables:
        return SQLQueryCandidate(
            sql=(
                "SELECT source_type, COUNT(*) AS record_count "
                "FROM sources "
                f"WHERE company_ticker = {safe_ticker} "
                "GROUP BY source_type "
                "ORDER BY record_count DESC "
                f"LIMIT {max_rows}"
            ),
            reasoning_note="Count source records by source type for the selected ticker.",
            tables_used=["sources"],
        )

    if "average" in lowered and any(term in lowered for term in ["close", "price", "stock"]) and "market_data" in allowed_tables:
        return SQLQueryCandidate(
            sql=(
                "SELECT AVG(close) AS average_close, MIN(date) AS window_start, MAX(date) AS window_end "
                "FROM market_data "
                f"WHERE company_ticker = {safe_ticker} "
                "AND close IS NOT NULL "
                f"LIMIT {max_rows}"
            ),
            reasoning_note="Compute the average closing price over the available market rows.",
            tables_used=["market_data"],
        )

    metric_names = _metric_names_for_question(question)
    if metric_names and "financial_metrics" in allowed_tables:
        metrics_sql = ", ".join(_sql_quote(metric_name) for metric_name in metric_names)
        return SQLQueryCandidate(
            sql=(
                "SELECT fiscal_period, metric_name, metric_value, unit, as_of_date, source_url "
                "FROM financial_metrics "
                f"WHERE company_ticker = {safe_ticker} "
                f"AND metric_name IN ({metrics_sql}) "
                "ORDER BY as_of_date DESC, fiscal_period DESC "
                f"LIMIT {max_rows}"
            ),
            reasoning_note="Return recent metric rows that match the question intent.",
            tables_used=["financial_metrics"],
        )

    if "market_data" in allowed_tables:
        return SQLQueryCandidate(
            sql=(
                "SELECT date, close, volume "
                "FROM market_data "
                f"WHERE company_ticker = {safe_ticker} "
                "ORDER BY date DESC "
                f"LIMIT {max_rows}"
            ),
            reasoning_note="Fallback to recent market rows for the selected ticker.",
            tables_used=["market_data"],
        )

    return SQLQueryCandidate(
        sql=(
            "SELECT fiscal_period, metric_name, metric_value, unit, as_of_date, source_url "
            "FROM financial_metrics "
            f"WHERE company_ticker = {safe_ticker} "
            "ORDER BY as_of_date DESC, fiscal_period DESC "
            f"LIMIT {max_rows}"
        ),
        reasoning_note="Fallback to recent financial metric rows.",
        tables_used=["financial_metrics"],
    )


def _llm_sql_candidate(
    *,
    question: str,
    ticker: str,
    schema_context: dict[str, Any],
    max_rows: int,
    previous_sql: str | None = None,
    error_message: str | None = None,
) -> SQLQueryCandidate | None:
    if not LLM.available:
        return None

    prompt_parts = [
        f"Question: {question}",
        f"Ticker: {ticker}",
        f"Maximum rows: {max_rows}",
        _context_to_prompt_text(schema_context),
    ]
    if previous_sql and error_message:
        prompt_parts.append(f"Previous SQL: {previous_sql}")
        prompt_parts.append(f"Execution error: {error_message}")
        system_prompt = SQL_REPAIR_SYSTEM_PROMPT
    else:
        system_prompt = SQL_GENERATOR_SYSTEM_PROMPT

    payload = LLM.complete_json(system_prompt=system_prompt, user_prompt="\n\n".join(prompt_parts), max_tokens=350)
    if not payload:
        return None
    try:
        return SQLQueryCandidate.model_validate(payload)
    except Exception:
        return None


def _build_sql_summary(rows: list[dict[str, Any]], tables_used: list[str]) -> str:
    if not rows:
        return f"SQL analysis ran over {', '.join(tables_used)} but returned no rows."
    column_names = [str(column) for column in rows[0].keys()]
    return f"SQL analysis returned {len(rows)} row(s) from {', '.join(tables_used)} with columns: {', '.join(column_names[:6])}."


class SchemaContextArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    question: str
    allowed_tables: list[str] = Field(default_factory=lambda: list(DEFAULT_SQL_TABLES))


@tool(args_schema=SchemaContextArgs)
def schema_context_tool(ticker: str, question: str, allowed_tables: list[str] | None = None) -> dict[str, Any]:
    """Build schema and glossary context for a Text2SQL request."""
    context = build_schema_context(question, ticker, allowed_tables=allowed_tables or list(DEFAULT_SQL_TABLES))
    return context.model_dump()


class SQLQueryArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    question: str
    mode: str = Field(default="single_query")
    allowed_tables: list[str] = Field(default_factory=lambda: list(DEFAULT_SQL_TABLES))
    max_rows: int = Field(default=20, ge=1, le=200)
    max_retries: int = Field(default=2, ge=0, le=3)


@tool(args_schema=SQLQueryArgs)
def sql_query_tool(
    ticker: str,
    question: str,
    mode: str = "single_query",
    allowed_tables: list[str] | None = None,
    max_rows: int = 20,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate, validate, and execute a safe read-only SQLite query."""
    allowed = allowed_tables or list(DEFAULT_SQL_TABLES)
    schema_context = schema_context_tool.invoke({"ticker": ticker, "question": question, "allowed_tables": allowed})
    candidate = _llm_sql_candidate(
        question=question,
        ticker=ticker,
        schema_context=schema_context,
        max_rows=max_rows,
    ) or _heuristic_sql(question, ticker, allowed, max_rows)

    attempts: list[str] = []
    last_error: str | None = None
    executed_result: SQLQueryResult | None = None

    for attempt_index in range(max_retries + 1):
        is_valid, validation_error = validate_select_sql(candidate.sql, allowed_tables=allowed, max_limit=max_rows)
        if not is_valid:
            last_error = validation_error
        else:
            try:
                rows, elapsed_ms = execute_readonly_sql(candidate.sql)
                executed_result = SQLQueryResult(
                    ticker=ticker,
                    question=question,
                    sql=candidate.sql,
                    status="ok",
                    row_count=len(rows),
                    rows=rows,
                    notes=[
                        candidate.reasoning_note or "SQL query executed successfully.",
                        f"Execution time: {elapsed_ms:.1f} ms",
                        f"Mode: {mode}",
                    ],
                )
                break
            except Exception as exc:  # pragma: no cover - exercised through repair path
                last_error = str(exc)

        attempts.append(f"Attempt {attempt_index + 1} failed: {last_error}")
        repair_candidate = _llm_sql_candidate(
            question=question,
            ticker=ticker,
            schema_context=schema_context,
            max_rows=max_rows,
            previous_sql=candidate.sql,
            error_message=last_error,
        )
        if repair_candidate is None:
            break
        candidate = repair_candidate

    if executed_result is None:
        executed_result = SQLQueryResult(
            ticker=ticker,
            question=question,
            sql=candidate.sql,
            status="error",
            row_count=0,
            rows=[],
            notes=attempts,
            error_message=last_error or "SQL generation failed.",
        )

    artifact_path = write_json_artifact(
        ticker,
        "sql_query_result",
        {
            "question": question,
            "mode": mode,
            "schema_context": schema_context,
            "result": executed_result.model_dump(mode="json"),
        },
    )
    finding = {
        "finding_type": "sql_analysis",
        "summary": _build_sql_summary(executed_result.rows, candidate.tables_used or extract_table_names(candidate.sql)),
        "supporting_records": executed_result.rows[:5],
        "metrics": {
            "row_count": executed_result.row_count,
            "tables_used": candidate.tables_used or extract_table_names(candidate.sql),
            "sql": executed_result.sql,
            "status": executed_result.status,
        },
        "chart_artifact_path": artifact_path,
    }
    return {
        "schema_context": schema_context,
        "sql_result": executed_result.model_dump(),
        "finding": finding,
        "artifact_path": artifact_path,
    }

