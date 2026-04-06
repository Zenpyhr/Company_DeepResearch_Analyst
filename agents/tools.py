from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - fallback only used in stripped-down environments
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

from app.artifacts import write_csv_artifact, write_json_artifact, write_markdown_artifact
from app.logging import get_logger
from pipelines.refresh_company import refresh_company_data
from pipelines.text_processing import process_company_documents
from schemas.models import FinalAnswer, ToolResult
from storage.bootstrap import bootstrap_storage
from storage.query_service import (
    derive_market_window,
    fetch_financial_metrics,
    fetch_market_data,
    infer_anchor_dates_from_metrics,
    safe_pct_change,
    safe_stddev,
    search_document_chunks,
)


logger = get_logger(__name__)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _json_safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe_value(record) for record in records]


class RefreshCompanyArgs(BaseModel):
    ticker: str = Field(default="NVDA", description="Ticker to refresh and process.")


@tool(args_schema=RefreshCompanyArgs)
def refresh_company_data_tool(ticker: str = "NVDA") -> dict[str, Any]:
    """Collect fresh NVDA source data and regenerate qualitative chunks."""
    bootstrap_storage()
    refresh_summary = refresh_company_data(ticker)
    process_summary = process_company_documents(ticker)
    result = {
        "ticker": ticker,
        "refresh_summary": refresh_summary,
        "process_summary": process_summary,
    }
    logger.info("Refresh tool completed for %s: %s", ticker, result)
    return result


class DocumentRetrievalArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    question: str
    top_k: int = Field(default=5, ge=1, le=12)
    source_types: list[str] = Field(default_factory=lambda: ["sec_filing", "press_release"])


@tool(args_schema=DocumentRetrievalArgs)
def retrieve_document_context_tool(ticker: str, question: str, top_k: int = 5, source_types: list[str] | None = None) -> dict[str, Any]:
    """Retrieve relevant SEC and press-release chunks for the question."""
    records = _json_safe_records(search_document_chunks(ticker, question, top_k=top_k, source_types=source_types))
    result = ToolResult(
        tool_name="retrieve_document_context_tool",
        company_ticker=ticker,
        summary=f"Retrieved {len(records)} qualitative records for the question.",
        records=records,
    )
    return result.model_dump()


class FinancialRetrievalArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    metric_names: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    limit_per_metric: int = Field(default=4, ge=1, le=12)


@tool(args_schema=FinancialRetrievalArgs)
def retrieve_financial_metrics_tool(
    ticker: str,
    metric_names: list[str] | None = None,
    periods: list[str] | None = None,
    limit_per_metric: int = 4,
) -> dict[str, Any]:
    """Retrieve structured financial metrics from SQLite."""
    records = _json_safe_records(
        fetch_financial_metrics(ticker, metric_names=metric_names, periods=periods, limit_per_metric=limit_per_metric)
    )
    result = ToolResult(
        tool_name="retrieve_financial_metrics_tool",
        company_ticker=ticker,
        summary=f"Retrieved {len(records)} financial metric rows.",
        records=records,
    )
    return result.model_dump()


class MarketRetrievalArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    start_date: str | None = None
    end_date: str | None = None
    anchor_dates: list[str] = Field(default_factory=list)
    window_days: int = Field(default=3, ge=1, le=30)
    limit: int = Field(default=40, ge=1, le=200)


@tool(args_schema=MarketRetrievalArgs)
def retrieve_market_data_tool(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    anchor_dates: list[str] | None = None,
    window_days: int = 3,
    limit: int = 40,
) -> dict[str, Any]:
    """Retrieve market rows directly or around anchor dates."""
    if anchor_dates:
        records = derive_market_window(ticker, anchor_dates=anchor_dates, days_before=window_days, days_after=window_days)
    else:
        records = fetch_market_data(ticker, start_date=start_date, end_date=end_date, limit=limit)
    records = _json_safe_records(records)

    result = ToolResult(
        tool_name="retrieve_market_data_tool",
        company_ticker=ticker,
        summary=f"Retrieved {len(records)} market rows.",
        records=records,
    )
    return result.model_dump()


class FinancialTrendArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    metric_rows: list[dict[str, Any]] = Field(default_factory=list)


@tool(args_schema=FinancialTrendArgs)
def financial_trend_tool(ticker: str, metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute period-over-period changes and trend summaries over metric rows."""
    if not metric_rows:
        return {"findings": [], "artifact_path": None, "summary": "No financial rows available for analysis."}

    df = pd.DataFrame(metric_rows)
    if df.empty:
        return {"findings": [], "artifact_path": None, "summary": "No financial rows available for analysis."}

    df["as_of_date"] = pd.to_datetime(df.get("as_of_date"), errors="coerce")
    df = df.sort_values(["metric_name", "as_of_date", "fiscal_period"], ascending=[True, False, False])

    findings: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for metric_name, group in df.groupby("metric_name"):
        group_records = _json_safe_records(group.to_dict(orient="records"))
        latest = group_records[0]
        previous = group_records[1] if len(group_records) > 1 else None
        latest_value = latest.get("metric_value")
        previous_value = previous.get("metric_value") if previous else None
        pct_change = safe_pct_change(latest_value, previous_value)
        summary = f"{metric_name.replace('_', ' ').title()} latest value is {latest_value:,.2f}"
        if previous_value is not None:
            summary += f" versus {previous_value:,.2f} in {previous.get('fiscal_period')}"
        if pct_change is not None:
            summary += f" ({pct_change:+.1f}% change)."
        else:
            summary += "."

        findings.append(
            {
                "finding_type": "financial_trend",
                "summary": summary,
                "supporting_records": group_records[:3],
                "metrics": {
                    "metric_name": metric_name,
                    "latest_fiscal_period": latest.get("fiscal_period"),
                    "latest_value": latest_value,
                    "previous_fiscal_period": previous.get("fiscal_period") if previous else None,
                    "previous_value": previous_value,
                    "pct_change": pct_change,
                },
            }
        )
        artifact_rows.extend(
            {
                "metric_name": record.get("metric_name"),
                "fiscal_period": record.get("fiscal_period"),
                "as_of_date": str(record.get("as_of_date") or ""),
                "metric_value": record.get("metric_value"),
            }
            for record in group_records[:6]
        )

    artifact_path = write_csv_artifact(ticker, "financial_trends", artifact_rows)
    return {
        "findings": findings,
        "artifact_path": artifact_path,
        "summary": f"Computed {len(findings)} financial trend findings.",
    }


class MarketReactionArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    market_rows: list[dict[str, Any]] = Field(default_factory=list)
    anchor_dates: list[str] = Field(default_factory=list)


@tool(args_schema=MarketReactionArgs)
def market_reaction_tool(ticker: str, market_rows: list[dict[str, Any]], anchor_dates: list[str] | None = None) -> dict[str, Any]:
    """Analyze market movement and volatility over a retrieved market window."""
    if not market_rows:
        return {"findings": [], "artifact_path": None, "summary": "No market rows available for analysis."}

    df = pd.DataFrame(market_rows)
    if df.empty or "close" not in df:
        return {"findings": [], "artifact_path": None, "summary": "No market rows available for analysis."}

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.sort_values("date")
    close_values = df["close"].dropna().astype(float).tolist()
    if len(close_values) < 2:
        return {"findings": [], "artifact_path": None, "summary": "Not enough market rows for analysis."}

    start_close = close_values[0]
    end_close = close_values[-1]
    total_return = safe_pct_change(end_close, start_close)
    daily_returns = [safe_pct_change(curr, prev) or 0.0 for prev, curr in zip(close_values[:-1], close_values[1:])]
    volatility = safe_stddev(daily_returns)

    summary = f"Over the retrieved window, NVDA moved from {start_close:,.2f} to {end_close:,.2f}"
    if total_return is not None:
        summary += f" ({total_return:+.1f}% total return)"
    if volatility is not None:
        summary += f" with {volatility:.2f} daily-return standard deviation."
    else:
        summary += "."

    artifact_rows = [
        {"date": str(row.get("date") or ""), "close": row.get("close"), "volume": row.get("volume")}
        for row in _json_safe_records(df.to_dict(orient="records"))
    ]
    artifact_path = write_csv_artifact(ticker, "market_reaction", artifact_rows)
    finding = {
        "finding_type": "market_reaction",
        "summary": summary,
        "supporting_records": _json_safe_records(df.tail(6).to_dict(orient="records")),
        "metrics": {
            "window_start_close": start_close,
            "window_end_close": end_close,
            "total_return_pct": total_return,
            "daily_return_stddev": volatility,
            "anchor_dates": anchor_dates or [],
        },
    }
    return {"findings": [finding], "artifact_path": artifact_path, "summary": "Computed market reaction analysis."}


class TextThemeArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    chunk_rows: list[dict[str, Any]] = Field(default_factory=list)
    question: str


@tool(args_schema=TextThemeArgs)
def text_theme_tool(ticker: str, chunk_rows: list[dict[str, Any]], question: str) -> dict[str, Any]:
    """Perform deterministic text analysis over retrieved chunk rows."""
    if not chunk_rows:
        return {"findings": [], "artifact_path": None, "summary": "No qualitative records available for analysis."}

    question_terms = [token for token in question.lower().split() if len(token) > 3]
    section_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    supporting_records: list[dict[str, Any]] = []

    for row in chunk_rows:
        metadata = row.get("metadata_json", {}) or {}
        section_label = str(metadata.get("section_label") or row.get("source_type") or "unknown")
        section_counter[section_label] += 1
        text = str(row.get("chunk_text") or "").lower()
        for term in question_terms:
            if term in text:
                keyword_counter[term] += text.count(term)
        if len(supporting_records) < 5:
            supporting_records.append(_json_safe_value(row))

    dominant_section, dominant_count = section_counter.most_common(1)[0]
    top_keywords = keyword_counter.most_common(5)
    summary = f"Retrieved text evidence concentrates most heavily in {dominant_section} ({dominant_count} chunks)."
    if top_keywords:
        summary += " Most repeated question-linked terms: " + ", ".join(f"{term} ({count})" for term, count in top_keywords) + "."

    chart_rows = [{"label": label, "count": count} for label, count in section_counter.most_common()]
    artifact_path = write_csv_artifact(ticker, "text_theme_counts", chart_rows)
    finding = {
        "finding_type": "text_theme",
        "summary": summary,
        "supporting_records": supporting_records,
        "metrics": {
            "dominant_section": dominant_section,
            "dominant_section_count": dominant_count,
            "top_keywords": top_keywords,
        },
    }
    return {"findings": [finding], "artifact_path": artifact_path, "summary": "Computed text-theme analysis."}


class ChartArgs(BaseModel):
    ticker: str = Field(default="NVDA")
    title: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    x_field: str
    y_field: str


@tool(args_schema=ChartArgs)
def chart_tool(ticker: str, title: str, rows: list[dict[str, Any]], x_field: str, y_field: str) -> dict[str, Any]:
    """Generate and persist a Vega-Lite chart specification."""
    if not rows:
        return {"chart_spec": None, "artifact_path": None}

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "data": {"values": rows},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": x_field, "type": "ordinal"},
            "y": {"field": y_field, "type": "quantitative"},
            "tooltip": [{"field": key, "type": "nominal"} for key in rows[0].keys()],
        },
    }
    artifact_path = write_json_artifact(ticker, "chart_spec", spec)
    return {"chart_spec": spec, "artifact_path": artifact_path}


def final_answer_builder(
    *,
    ticker: str,
    question: str,
    findings: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
    llm_summary: str | None = None,
) -> tuple[FinalAnswer, str]:
    sources: list[str] = []
    key_points: list[str] = []
    support_snippets: list[str] = []
    for record in evidence_bundle.get("qualitative_records", [])[:5]:
        source_url = record.get("source_url")
        title = record.get("title")
        if source_url and source_url not in sources:
            sources.append(str(source_url))
        if title and len(support_snippets) < 3:
            support_snippets.append(f"- {title}")
    for record in evidence_bundle.get("quantitative_records", [])[:5]:
        source_url = record.get("source_url")
        if source_url and source_url not in sources:
            sources.append(str(source_url))

    for finding in findings[:5]:
        summary = finding.get("summary")
        if summary:
            key_points.append(str(summary))

    if llm_summary:
        answer_text = llm_summary.strip()
    else:
        answer_parts = ["Grounded analyst view based on collected NVDA evidence:"]
        answer_parts.extend(f"- {point}" for point in key_points[:3])
        if support_snippets:
            answer_parts.append("Primary source context included:")
            answer_parts.extend(support_snippets[:2])
        answer_text = "\n".join(answer_parts)

    final_answer = FinalAnswer(
        company_ticker=ticker,
        question=question,
        answer=answer_text,
        key_points=key_points[:5],
        sources=sources[:8],
        confidence_note="This answer is grounded in retrieved evidence and deterministic EDA over the local NVDA dataset.",
    )

    markdown_lines = [
        f"# NVDA Analyst Memo",
        "",
        f"## Question",
        question,
        "",
        f"## Answer",
        final_answer.answer,
        "",
        f"## Key Points",
    ]
    markdown_lines.extend(f"- {point}" for point in final_answer.key_points)
    markdown_lines.extend(["", "## Sources"])
    markdown_lines.extend(f"- {source}" for source in final_answer.sources)
    artifact_path = write_markdown_artifact(ticker, "analyst_memo", "\n".join(markdown_lines))
    return final_answer, artifact_path
