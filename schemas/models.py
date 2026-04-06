from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# These literals keep source/category values consistent across the app.
# That makes downstream filtering and tool responses much easier to reason about.
SourceType = Literal["sec_filing", "press_release", "companyfacts", "market_data", "derived"]
QuestionType = Literal["qualitative", "quantitative", "mixed"]
QuestionCategory = Literal["financial_trend", "market_reaction", "risk_narrative", "mixed"]


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
    # This will be produced by the Planner Agent to guide the rest of the workflow.
    company_ticker: str
    question: str
    question_type: QuestionType
    question_category: QuestionCategory | None = None
    goals: list[str] = Field(default_factory=list)
    tools_to_call: list[str] = Field(default_factory=list)
    notes: str | None = None


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
    # A shared evidence package passed from Researcher to EDA/Analyst.
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
