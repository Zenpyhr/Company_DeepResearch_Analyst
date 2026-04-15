from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# These literals keep source/category values consistent across the app.
# That makes downstream filtering and tool responses much easier to reason about.
SourceType = Literal["sec_filing", "press_release", "companyfacts", "market_data", "derived"]
QuestionType = Literal["qualitative", "quantitative", "mixed", "out_of_scope"]
QuestionCategory = Literal["financial_trend", "market_reaction", "risk_narrative", "mixed", "out_of_scope"]
QuestionSubIntent = Literal["financial_trend", "market_reaction", "risk_narrative"]
PlanningSource = Literal["llm", "heuristic"]
RoutingSource = Literal["llm", "heuristic_fallback"]
SQLQueryMode = Literal["single_query", "analysis"]


class BaseSchema(BaseModel):
    # Forbid unknown fields so bad or incomplete records fail early.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Company(BaseSchema):
    # The top-level identity for the selected public company.
    ticker: str = Field(..., description="Public ticker symbol, e.g. NVDA")
    company_name: str = Field(..., description="Full company name")
    cik: str | None = Field(default=None, description="SEC CIK if known")
    industry: str | None = None
    website: str | None = None
    last_refreshed_at: datetime | None = None


class Source(BaseSchema):
    # One original document or dataset we collected.
    # Examples: a 10-K filing, a press release page, or a CompanyFacts payload.
    company_ticker: str
    source_type: SourceType
    title: str
    source_url: str
    published_at: datetime | None = None
    raw_path: str | None = None
    # Flexible metadata lets us keep source-specific fields without exploding the schema.
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseSchema):
    # A chunk is a small text passage cut from a larger source document.
    # We keep the text and metadata in SQLite, while FAISS will later store the embeddings.
    company_ticker: str
    source_id: int | None = None
    source_type: SourceType | None = None
    title: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    chunk_text: str
    chunk_order: int
    embedding_id: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class FinancialMetric(BaseSchema):
    # One structured business metric for a known fiscal period.
    # This is part of the quantitative layer used for SQL-style lookup.
    company_ticker: str
    fiscal_period: str
    metric_name: str
    metric_value: float
    unit: str | None = None
    source_url: str | None = None
    as_of_date: datetime | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class MarketRecord(BaseSchema):
    # One row of daily market data such as OHLCV.
    company_ticker: str
    date: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class Event(BaseSchema):
    # Events are structured analytical facts extracted from qualitative text.
    # They are not pure numeric data; they act as a bridge between text evidence and reporting.
    company_ticker: str
    timestamp: datetime | None = None
    source_type: SourceType
    event_type: str
    event_subtype: str | None = None
    description: str
    evidence_text: str | None = None
    source_url: str | None = None
    confidence: float | None = None
    verification_status: str = "unverified"
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseSchema):
    # This is produced before collection to guide the workflow.
    company_ticker: str
    question: str
    question_type: QuestionType
    question_category: QuestionCategory | None = None
    sub_intents: list[QuestionSubIntent] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    tools_to_call: list[str] = Field(default_factory=list)
    notes: str | None = None


class RetrievalPlan(BaseSchema):
    # Structured retrieval instructions produced by the orchestrator.
    needs_qualitative: bool = False
    needs_quantitative: bool = False
    source_types: list[Literal["sec_filing", "press_release"]] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    limit_per_metric: int = Field(default=4, ge=1, le=12)
    needs_market_data: bool = False
    anchor_date_count: int = Field(default=1, ge=1, le=4)
    market_window_days: int = Field(default=2, ge=1, le=30)


class EDAPlan(BaseSchema):
    # Structured EDA-tool selection produced by the orchestrator.
    selected_tools: list[str] = Field(default_factory=list)
    chart_metric: str | None = None
    notes: list[str] = Field(default_factory=list)


class RetryDecision(BaseSchema):
    # Retry decision after the first EDA pass.
    retry_requested: bool = False
    missing_modalities: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    reason: str | None = None


class OrchestrationPlan(BaseSchema):
    # High-level decision object that drives Collector and EDA behavior.
    company_ticker: str
    question: str
    question_type: QuestionType
    question_category: QuestionCategory
    sub_intents: list[QuestionSubIntent] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan
    eda_plan: EDAPlan
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    confidence_notes: str | None = None
    planning_source: PlanningSource = "heuristic"


class SchemaColumnProfile(BaseSchema):
    table_name: str
    column_name: str
    data_type: str
    description: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    is_enum: bool = False
    enum_values: list[str] = Field(default_factory=list)


class SchemaTableProfile(BaseSchema):
    table_name: str
    description: str | None = None
    columns: list[SchemaColumnProfile] = Field(default_factory=list)


class SchemaContext(BaseSchema):
    ticker: str
    question: str
    relevant_tables: list[str] = Field(default_factory=list)
    tables: list[SchemaTableProfile] = Field(default_factory=list)
    glossary_hits: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SQLQueryRequest(BaseSchema):
    question: str
    ticker: str
    mode: SQLQueryMode = "single_query"
    allowed_tables: list[str] = Field(default_factory=list)
    max_rows: int = Field(default=50, ge=1, le=200)


class SQLQueryCandidate(BaseSchema):
    sql: str
    reasoning_note: str | None = None
    tables_used: list[str] = Field(default_factory=list)


class SQLQueryResult(BaseSchema):
    ticker: str
    question: str
    sql: str
    status: Literal["ok", "error"] = "ok"
    row_count: int = 0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error_message: str | None = None


class ToolRequest(BaseSchema):
    # Standard shape for calling a tool or skill.
    tool_name: str
    company_ticker: str
    query: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseSchema):
    # Standard shape for returning structured tool output back to the agent workflow.
    tool_name: str
    company_ticker: str
    status: Literal["ok", "error"] = "ok"
    summary: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


class EvidenceBundle(BaseSchema):
    # A shared evidence package passed from the Collector to EDA/Analyst.
    company_ticker: str
    question: str
    qualitative_records: list[dict[str, Any]] = Field(default_factory=list)
    quantitative_records: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_notes: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class AnalysisFinding(BaseSchema):
    # A concrete, evidence-backed EDA result.
    finding_type: str
    summary: str
    supporting_records: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    chart_artifact_path: str | None = None


class AnalysisBundle(BaseSchema):
    # The output of the EDA stage before the final analyst synthesis.
    company_ticker: str
    question: str
    findings: list[AnalysisFinding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    requires_additional_research: bool = False
    missing_modalities: list[str] = Field(default_factory=list)


class FinalAnswer(BaseSchema):
    # Final grounded response shape returned to the UI.
    company_ticker: str
    question: str
    answer: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence_note: str | None = None
