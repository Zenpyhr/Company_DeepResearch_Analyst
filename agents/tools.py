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

TEXT_THEME_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "which",
    "most",
    "still",
    "stand",
    "about",
    "around",
    "over",
    "recent",
    "reported",
    "reporting",
    "support",
    "supports",
    "suggest",
    "suggests",
    "nvidia",
    "nvda",
    "does",
    "did",
    "how",
}

RISK_THEME_PATTERNS: dict[str, list[str]] = {
    "competition": ["competition", "competitive", "market share"],
    "supply_chain": ["supply", "capacity", "manufactur", "lead time", "supplier"],
    "demand_forecasting": ["demand", "forecast", "estimate customer demand", "mismatch"],
    "regulation_export": ["regulation", "regulatory", "export", "government", "compliance"],
    "customer_concentration": ["customer concentration", "large customer", "concentration"],
    "cloud_execution": ["cloud", "service", "deployment", "software", "adoption"],
}

GROWTH_THEME_PATTERNS: dict[str, list[str]] = {
    "ai_demand": ["ai", "artificial intelligence", "inference", "training"],
    "data_center": ["data center", "data-cent", "dgx", "infrastructure"],
    "software_platform": ["cuda", "software", "sdk", "library", "enterprise"],
    "partnerships": ["partner", "partnership", "collaboration", "ecosystem"],
    "product_platform": ["blackwell", "grace", "rubin", "gpu", "platform"],
}


def _json_safe_value(value: Any) -> Any:
    # Tool outputs are later:
    # - stored in graph state,
    # - rendered in Streamlit,
    # - sometimes serialized into JSON for prompts/artifacts.
    # This helper normalizes pandas/datetime values early so every downstream stage can
    # treat tool records as safe, plain Python data instead of worrying about serializer
    # edge cases.
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
    # This is the qualitative retrieval entrypoint.
    # It does not interpret the text; it only returns the most relevant chunk rows so the
    # Collector can package them into the evidence bundle and the EDA stage can analyze them.
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
    # This is the main quantitative retrieval entrypoint for accounting-style questions.
    # The caller chooses metric names and period depth, and the tool returns normalized
    # rows that are ready for deterministic analysis.
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
    # Market data is treated as a second quantitative modality because the time window is
    # often different from the accounting periods. We support both direct date filters and
    # anchor-date windows so the workflow can tie price movement back to financial events.
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
    # This tool is a good place to study how "EDA" is implemented in practice.
    # Input: raw metric rows retrieved from SQLite.
    # Output: structured findings with summaries, supporting records, and numeric metrics.
    # It also writes an artifact so the analysis can be inspected outside the final answer.
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
        # Each metric is analyzed independently so the analyst later receives a set of
        # small, concrete findings instead of one oversized blob of mixed numeric context.
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
    # This is the market-focused EDA tool. It converts a raw sequence of daily rows into
    # a compact statement about return and volatility, which is much easier for the final
    # analyst stage to combine with textual and financial findings.
    if not market_rows:
        return {"findings": [], "artifact_path": None, "summary": "No market rows available for analysis."}

    df = pd.DataFrame(market_rows)
    if df.empty or "close" not in df:
        return {"findings": [], "artifact_path": None, "summary": "No market rows available for analysis."}

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df = df.sort_values("date")
    close_values = df["close"].dropna().astype(float).tolist()
    if len(close_values) < 2:
        return {"findings": [], "artifact_path": None, "summary": "Not enough market rows for analysis."}

    start_close = close_values[0]
    end_close = close_values[-1]
    total_return = safe_pct_change(end_close, start_close)
    daily_returns = [safe_pct_change(curr, prev) or 0.0 for prev, curr in zip(close_values[:-1], close_values[1:])]
    volatility = safe_stddev(daily_returns)
    anchor_label = None
    if anchor_dates:
        anchor_label = sorted({str(anchor)[:10] for anchor in anchor_dates if anchor}, reverse=True)[0]

    if anchor_label:
        summary = f"Around the reporting anchor on {anchor_label}, NVDA moved from {start_close:,.2f} to {end_close:,.2f}"
    else:
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


def _top_themes_for_text(
    *,
    text: str,
    use_risk_themes: bool,
    use_growth_themes: bool,
) -> list[str]:
    matched: list[str] = []
    if use_risk_themes:
        for theme_name, patterns in RISK_THEME_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                matched.append(theme_name)
    if use_growth_themes:
        for theme_name, patterns in GROWTH_THEME_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                matched.append(theme_name)
    return matched


def _format_theme_counts(theme_counts: list[tuple[str, int]], *, limit: int = 3) -> str:
    trimmed = theme_counts[:limit]
    return ", ".join(f"{theme.replace('_', ' ')} ({count})" for theme, count in trimmed)


@tool(args_schema=TextThemeArgs)
def text_theme_tool(ticker: str, chunk_rows: list[dict[str, Any]], question: str) -> dict[str, Any]:
    """Perform deterministic text analysis over retrieved chunk rows."""
    # This is the qualitative EDA tool. Instead of asking an LLM to "summarize the text,"
    # we compute simple deterministic signals first: repeated sections and question-linked
    # terms. That makes the qualitative path more auditable and assignment-friendly.
    if not chunk_rows:
        return {"findings": [], "artifact_path": None, "summary": "No qualitative records available for analysis."}

    question_terms = [
        token
        for token in pd.unique(pd.Series(question.lower().replace("?", " ").replace(",", " ").split())).tolist()
        if isinstance(token, str) and len(token) > 3 and token not in TEXT_THEME_STOPWORDS
    ]
    section_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()
    source_theme_counters: dict[str, Counter[str]] = {
        "sec_filing": Counter(),
        "press_release": Counter(),
    }
    supporting_records: list[dict[str, Any]] = []
    lowered_question = question.lower()
    use_risk_themes = "risk" in lowered_question
    use_growth_themes = any(term in lowered_question for term in ["growth", "ai", "demand", "press", "release"])

    for row in chunk_rows:
        metadata = row.get("metadata_json", {}) or {}
        source_type = str(row.get("source_type") or "unknown")
        section_label = str(metadata.get("section_label") or row.get("source_type") or "unknown")
        section_counter[section_label] += 1
        source_counter[source_type] += 1
        text = str(row.get("chunk_text") or "").lower()
        for term in question_terms:
            if term in text:
                keyword_counter[term] += text.count(term)
        matched_themes = _top_themes_for_text(
            text=text,
            use_risk_themes=use_risk_themes,
            use_growth_themes=use_growth_themes,
        )
        for theme_name in matched_themes:
            theme_counter[theme_name] += 1
            if source_type in source_theme_counters:
                source_theme_counters[source_type][theme_name] += 1
        if len(supporting_records) < 5:
            supporting_records.append(_json_safe_value(row))

    dominant_section, dominant_count = section_counter.most_common(1)[0]
    top_keywords = keyword_counter.most_common(5)
    top_themes = theme_counter.most_common(5)
    filing_top_themes = source_theme_counters["sec_filing"].most_common(5)
    press_release_top_themes = source_theme_counters["press_release"].most_common(5)
    summary = f"Retrieved text evidence concentrates most heavily in {dominant_section} ({dominant_count} chunks)."
    if filing_top_themes and press_release_top_themes:
        summary += (
            " Filing themes: "
            + _format_theme_counts(filing_top_themes)
            + ". Press-release themes: "
            + _format_theme_counts(press_release_top_themes)
            + "."
        )
    elif filing_top_themes:
        summary += " Filing themes: " + _format_theme_counts(filing_top_themes) + "."
    elif press_release_top_themes:
        summary += " Press-release themes: " + _format_theme_counts(press_release_top_themes) + "."
    elif top_themes:
        summary += " Most visible themes: " + _format_theme_counts(top_themes, limit=5) + "."
    elif top_keywords:
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
            "top_themes": top_themes,
            "source_theme_breakdown": {
                "sec_filing": filing_top_themes,
                "press_release": press_release_top_themes,
            },
            "filing_top_themes": filing_top_themes,
            "press_release_top_themes": press_release_top_themes,
            "source_counts": source_counter.most_common(),
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


def _rank_findings_for_question(question: str, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered_question = question.lower()
    wants_risk = "risk" in lowered_question
    wants_market = any(term in lowered_question for term in ["stock", "market", "price", "reaction"])
    wants_growth = any(term in lowered_question for term in ["revenue", "growth", "financial", "income", "margin", "profit"])

    metric_priority = {
        "revenue": 0,
        "gross_profit": 1,
        "net_income": 2,
        "cash_and_cash_equivalents": 3,
        "research_and_development": 4,
        "eps_diluted": 5,
    }

    def score(finding: dict[str, Any]) -> tuple[int, int, str]:
        finding_type = str(finding.get("finding_type") or "")
        metrics = finding.get("metrics", {}) or {}
        metric_name = str(metrics.get("metric_name") or "")

        if wants_market and not wants_growth and not wants_risk:
            priority_map = {
                "market_reaction": 0,
                "financial_trend": 1,
                "text_theme": 2,
            }
            return (priority_map.get(finding_type, 9), metric_priority.get(metric_name, 9), metric_name)

        if wants_risk and not wants_growth and not wants_market:
            priority_map = {
                "text_theme": 0,
                "financial_trend": 1,
                "market_reaction": 2,
            }
            return (priority_map.get(finding_type, 9), metric_priority.get(metric_name, 9), metric_name)

        if wants_growth and wants_risk:
            if finding_type == "financial_trend":
                growth_priority = {
                    "revenue": 0,
                    "gross_profit": 2,
                    "net_income": 3,
                    "cash_and_cash_equivalents": 4,
                    "research_and_development": 5,
                    "eps_diluted": 6,
                }
                return (0 if metric_name == "revenue" else 2, growth_priority.get(metric_name, 9), metric_name)
            if finding_type == "text_theme":
                return (1, 0, metric_name)
            if finding_type == "market_reaction":
                return (4, 0, metric_name)
            return (9, 9, metric_name)

        if wants_growth:
            if finding_type == "financial_trend":
                return (0, metric_priority.get(metric_name, 9), metric_name)
            if finding_type == "text_theme":
                return (1, 0, metric_name)
            if finding_type == "market_reaction":
                return (2, 0, metric_name)
            return (9, 9, metric_name)

        priority_map = {
            "financial_trend": 0,
            "text_theme": 1,
            "market_reaction": 2,
        }
        return (priority_map.get(finding_type, 9), metric_priority.get(metric_name, 9), metric_name)

    return sorted(findings, key=score)


def _build_mixed_narrative_point(
    *,
    question: str,
    text_finding: dict[str, Any] | None,
    evidence_bundle: dict[str, Any],
) -> str | None:
    if not text_finding:
        return None

    qualitative_records = evidence_bundle.get("qualitative_records", []) or []
    source_counter: Counter[str] = Counter(
        str(record.get("source_type") or "")
        for record in qualitative_records
        if record.get("source_type")
    )
    metrics = text_finding.get("metrics", {}) or {}
    top_themes = metrics.get("top_themes", []) or []
    filing_themes = metrics.get("filing_top_themes", []) or metrics.get("source_theme_breakdown", {}).get("sec_filing", [])
    press_release_themes = metrics.get("press_release_top_themes", []) or metrics.get("source_theme_breakdown", {}).get("press_release", [])
    theme_names = [str(theme[0]).replace("_", " ") for theme in top_themes[:3] if isinstance(theme, (list, tuple)) and theme]
    if not theme_names:
        return str(text_finding.get("summary") or "") or None

    if source_counter.get("sec_filing", 0) and source_counter.get("press_release", 0):
        filing_theme_names = [
            str(theme[0]).replace("_", " ")
            for theme in filing_themes[:3]
            if isinstance(theme, (list, tuple)) and theme
        ]
        press_theme_names = [
            str(theme[0]).replace("_", " ")
            for theme in press_release_themes[:3]
            if isinstance(theme, (list, tuple)) and theme
        ]
        if filing_theme_names and press_theme_names:
            return (
                f"Filings emphasize {', '.join(filing_theme_names)} risk, while press releases lean more toward "
                f"{', '.join(press_theme_names)} messaging."
            )
        return (
            "Filings continue to emphasize risk and operating constraints, while press releases lean more toward growth messaging. "
            f"The strongest shared themes in the retrieved text are {', '.join(theme_names)}."
        )
    if source_counter.get("sec_filing", 0):
        filing_theme_names = [
            str(theme[0]).replace("_", " ")
            for theme in filing_themes[:3]
            if isinstance(theme, (list, tuple)) and theme
        ]
        if filing_theme_names:
            return f"Filing evidence remains the main qualitative support and highlights {', '.join(filing_theme_names)} risk."
        return f"Filing evidence remains the main qualitative support and highlights themes such as {', '.join(theme_names)}."
    if source_counter.get("press_release", 0):
        press_theme_names = [
            str(theme[0]).replace("_", " ")
            for theme in press_release_themes[:3]
            if isinstance(theme, (list, tuple)) and theme
        ]
        if press_theme_names:
            return f"Press-release evidence is the main qualitative support and highlights themes such as {', '.join(press_theme_names)}."
        return f"Press-release evidence is the main qualitative support and highlights themes such as {', '.join(theme_names)}."
    return f"The strongest qualitative themes in the retrieved evidence are {', '.join(theme_names)}."


def _build_fallback_answer_text(question: str, key_points: list[str], support_snippets: list[str]) -> str:
    lowered_question = question.lower()
    wants_mixed = any(term in lowered_question for term in ["risk", "growth", "press", "release", "filing", "filings"])
    if wants_mixed and key_points:
        lead_points = key_points[:3]
        answer_parts = [
            "Grounded analyst view based on collected NVDA evidence:",
            f"The strongest support comes from {lead_points[0].lower()}",
        ]
        if len(lead_points) > 1:
            answer_parts.append(f"Qualitative evidence adds that {lead_points[1].lower()}")
        if len(lead_points) > 2:
            answer_parts.append(f"A second financial signal shows that {lead_points[2].lower()}")
        if support_snippets:
            answer_parts.append("Primary source context included:")
            answer_parts.extend(support_snippets[:2])
        return "\n".join(answer_parts)

    answer_parts = ["Grounded analyst view based on collected NVDA evidence:"]
    answer_parts.extend(f"- {point}" for point in key_points[:3])
    if support_snippets:
        answer_parts.append("Primary source context included:")
        answer_parts.extend(support_snippets[:2])
    return "\n".join(answer_parts)


def final_answer_builder(
    *,
    ticker: str,
    question: str,
    findings: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
    llm_summary: str | None = None,
) -> tuple[FinalAnswer, str]:
    # This helper is the last step that turns structured evidence into a deliverable.
    # It intentionally works from findings/evidence rather than the raw question alone,
    # which keeps the final answer grounded in earlier stages of the pipeline.
    ranked_findings = _rank_findings_for_question(question, list(findings))
    selected_findings = ranked_findings[:3]

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

    text_finding = next((finding for finding in ranked_findings if finding.get("finding_type") == "text_theme"), None)
    lowered_question = question.lower()
    wants_mixed = any(term in lowered_question for term in ["risk", "growth", "press", "release", "filing", "filings"])
    mixed_narrative_point = _build_mixed_narrative_point(question=question, text_finding=text_finding, evidence_bundle=evidence_bundle)

    for finding in selected_findings:
        summary = finding.get("summary")
        if summary:
            key_points.append(str(summary))

    if wants_mixed and mixed_narrative_point:
        key_points = [point for point in key_points if point != str(text_finding.get("summary") if text_finding else "")]
        key_points.insert(1 if key_points else 0, mixed_narrative_point)

    if llm_summary:
        answer_text = llm_summary.strip()
    else:
        answer_text = _build_fallback_answer_text(question, key_points, support_snippets)

    final_answer = FinalAnswer(
        company_ticker=ticker,
        question=question,
        answer=answer_text,
        key_points=key_points[:3],
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
