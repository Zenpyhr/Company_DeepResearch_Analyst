from __future__ import annotations

import json
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - fallback only used when langgraph is unavailable
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]

from agents.llm import OptionalLLM
from agents.tools import (
    chart_tool,
    final_answer_builder,
    financial_trend_tool,
    market_reaction_tool,
    refresh_company_data_tool,
    retrieve_document_context_tool,
    retrieve_financial_metrics_tool,
    retrieve_market_data_tool,
    text_theme_tool,
)
from app.config import get_settings
from app.logging import get_logger
from prompts.agent_prompts import ANALYST_SYSTEM_PROMPT, EDA_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT
from schemas.models import AnalysisBundle, AnalysisFinding, EvidenceBundle, ResearchPlan, ToolResult
from storage.query_service import infer_anchor_dates_from_metrics


logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    company_ticker: str
    research_plan: dict[str, Any]
    evidence_bundle: dict[str, Any]
    analysis_bundle: dict[str, Any]
    final_answer: dict[str, Any]
    chart_spec: dict[str, Any] | None
    chart_artifact_path: str | None
    memo_artifact_path: str | None
    loop_count: int
    retry_requested: bool
    missing_modalities: list[str]
    execution_log: list[str]


LLM = OptionalLLM()


def _classify_question(question: str) -> tuple[str, str]:
    lowered = question.lower()
    financial_keywords = {"revenue", "income", "profit", "eps", "cash", "margin", "quarter", "quarters", "financial", "growth"}
    market_keywords = {"stock", "price", "market", "volume", "return", "reaction", "trading", "shares"}
    qualitative_keywords = {"risk", "risks", "narrative", "management", "filing", "press", "release", "business", "strategy", "demand"}
    mixed_keywords = {"memo", "report", "support", "thesis", "outlook", "analyst"}

    has_financial = any(keyword in lowered for keyword in financial_keywords)
    has_market = any(keyword in lowered for keyword in market_keywords)
    has_qualitative = any(keyword in lowered for keyword in qualitative_keywords)
    has_mixed = any(keyword in lowered for keyword in mixed_keywords)

    if has_market and not has_qualitative and not has_mixed:
        return "market_reaction", "quantitative"
    if has_qualitative and not has_financial and not has_market and not has_mixed:
        return "risk_narrative", "qualitative"
    if has_mixed or ((has_financial or has_market) and has_qualitative):
        return "mixed", "mixed"
    return "financial_trend", "quantitative"


def _metric_names_for_question(question: str) -> list[str]:
    lowered = question.lower()
    candidates = {
        "revenue": ["revenue", "sales", "top line"],
        "net_income": ["income", "net income", "profit", "bottom line"],
        "gross_profit": ["gross profit", "gross margin"],
        "eps_diluted": ["eps", "earnings per share"],
        "research_and_development": ["research", "r&d", "development"],
        "cash_and_cash_equivalents": ["cash", "liquidity"],
    }
    selected = [metric for metric, aliases in candidates.items() if any(alias in lowered for alias in aliases)]
    return selected or ["revenue", "net_income", "gross_profit", "eps_diluted"]


def _build_research_plan(question: str, ticker: str) -> ResearchPlan:
    question_category, question_type = _classify_question(question)
    goals: list[str] = []
    tools_to_call: list[str] = []
    if question_category == "financial_trend":
        goals = ["Retrieve recent financial metrics", "Compute period-over-period trend analysis"]
        tools_to_call = ["retrieve_financial_metrics_tool", "financial_trend_tool"]
    elif question_category == "market_reaction":
        goals = ["Retrieve relevant market data", "Measure market reaction and volatility"]
        tools_to_call = ["retrieve_market_data_tool", "market_reaction_tool"]
    elif question_category == "risk_narrative":
        goals = ["Retrieve filing and press-release context", "Identify repeated themes and risks"]
        tools_to_call = ["retrieve_document_context_tool", "text_theme_tool"]
    else:
        goals = [
            "Retrieve both qualitative and quantitative evidence",
            "Analyze trends before writing a grounded memo",
        ]
        tools_to_call = [
            "retrieve_document_context_tool",
            "retrieve_financial_metrics_tool",
            "retrieve_market_data_tool",
            "financial_trend_tool",
            "text_theme_tool",
        ]

    notes = "Deterministic routing is used by default; an LLM may refine the narrative if configured."
    if LLM.available:
        prompt = (
            f"Question: {question}\n"
            f"Current heuristic category: {question_category}\n"
            f"Current heuristic type: {question_type}\n"
            "Briefly refine the research goals in one or two sentences."
        )
        llm_notes = LLM.complete(system_prompt=RESEARCHER_SYSTEM_PROMPT, user_prompt=prompt, max_tokens=150)
        if llm_notes:
            notes = llm_notes.strip()

    return ResearchPlan(
        company_ticker=ticker,
        question=question,
        question_type=question_type,  # type: ignore[arg-type]
        question_category=question_category,  # type: ignore[arg-type]
        goals=goals,
        tools_to_call=tools_to_call,
        notes=notes,
    )


def _merge_records(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in [*existing, *incoming]:
        key = tuple(record.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


def _collector_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    plan = _build_research_plan(question, ticker)
    evidence_bundle = EvidenceBundle.model_validate(
        state.get("evidence_bundle")
        or {
            "company_ticker": ticker,
            "question": question,
            "qualitative_records": [],
            "quantitative_records": [],
            "retrieval_notes": [],
            "tool_results": [],
        }
    )
    missing_modalities = state.get("missing_modalities", [])
    next_loop_count = state.get("loop_count", 0)
    if missing_modalities and state.get("analysis_bundle") and next_loop_count < 1:
        next_loop_count += 1
    needs_qualitative = plan.question_type in {"qualitative", "mixed"} or "qualitative" in missing_modalities
    needs_quantitative = plan.question_type in {"quantitative", "mixed"} or "quantitative" in missing_modalities

    execution_log = list(state.get("execution_log", []))
    if needs_qualitative:
        qualitative_result = retrieve_document_context_tool.invoke(
            {"ticker": ticker, "question": question, "top_k": 6, "source_types": ["sec_filing", "press_release"]}
        )
        evidence_bundle.qualitative_records = _merge_records(
            evidence_bundle.qualitative_records,
            qualitative_result["records"],
            ["id", "source_id", "chunk_order"],
        )
        evidence_bundle.tool_results.append(ToolResult.model_validate(qualitative_result))
        evidence_bundle.retrieval_notes.append(qualitative_result["summary"])
        execution_log.append("retrieve_document_context_tool")

    if needs_quantitative:
        metric_names = _metric_names_for_question(question)
        metrics_result = retrieve_financial_metrics_tool.invoke(
            {"ticker": ticker, "metric_names": metric_names, "limit_per_metric": 4}
        )
        metric_rows = [{**record, "record_kind": "financial_metric"} for record in metrics_result["records"]]
        evidence_bundle.quantitative_records = _merge_records(
            evidence_bundle.quantitative_records,
            metric_rows,
            ["metric_name", "fiscal_period", "record_kind"],
        )
        evidence_bundle.tool_results.append(ToolResult.model_validate(metrics_result))
        evidence_bundle.retrieval_notes.append(metrics_result["summary"])
        execution_log.append("retrieve_financial_metrics_tool")

        lowered = question.lower()
        if plan.question_category in {"market_reaction", "mixed"} or any(keyword in lowered for keyword in ["stock", "price", "market", "reaction"]):
            anchor_dates = infer_anchor_dates_from_metrics(metrics_result["records"])
            market_result = retrieve_market_data_tool.invoke(
                {"ticker": ticker, "anchor_dates": anchor_dates, "window_days": 5, "limit": 40}
            )
            market_rows = [{**record, "record_kind": "market_data"} for record in market_result["records"]]
            evidence_bundle.quantitative_records = _merge_records(
                evidence_bundle.quantitative_records,
                market_rows,
                ["date", "record_kind"],
            )
            evidence_bundle.tool_results.append(ToolResult.model_validate(market_result))
            evidence_bundle.retrieval_notes.append(market_result["summary"])
            execution_log.append("retrieve_market_data_tool")

    if (
        not evidence_bundle.qualitative_records
        and not evidence_bundle.quantitative_records
        and state.get("loop_count", 0) == 0
    ):
        try:
            refresh_result = refresh_company_data_tool.invoke({"ticker": ticker})
            evidence_bundle.retrieval_notes.append(
                f"Refresh/process step completed: {refresh_result['refresh_summary']} and {refresh_result['process_summary']}"
            )
            execution_log.append("refresh_company_data_tool")
            if needs_qualitative:
                qualitative_result = retrieve_document_context_tool.invoke(
                    {"ticker": ticker, "question": question, "top_k": 6, "source_types": ["sec_filing", "press_release"]}
                )
                evidence_bundle.qualitative_records = _merge_records(
                    evidence_bundle.qualitative_records,
                    qualitative_result["records"],
                    ["id", "source_id", "chunk_order"],
                )
                evidence_bundle.tool_results.append(ToolResult.model_validate(qualitative_result))
                evidence_bundle.retrieval_notes.append(qualitative_result["summary"])
                execution_log.append("retrieve_document_context_tool")
            if needs_quantitative:
                metrics_result = retrieve_financial_metrics_tool.invoke(
                    {"ticker": ticker, "metric_names": _metric_names_for_question(question), "limit_per_metric": 4}
                )
                metric_rows = [{**record, "record_kind": "financial_metric"} for record in metrics_result["records"]]
                evidence_bundle.quantitative_records = _merge_records(
                    evidence_bundle.quantitative_records,
                    metric_rows,
                    ["metric_name", "fiscal_period", "record_kind"],
                )
                evidence_bundle.tool_results.append(ToolResult.model_validate(metrics_result))
                evidence_bundle.retrieval_notes.append(metrics_result["summary"])
                execution_log.append("retrieve_financial_metrics_tool")
                lowered = question.lower()
                if plan.question_category in {"market_reaction", "mixed"} or any(keyword in lowered for keyword in ["stock", "price", "market", "reaction"]):
                    anchor_dates = infer_anchor_dates_from_metrics(metrics_result["records"])
                    market_result = retrieve_market_data_tool.invoke(
                        {"ticker": ticker, "anchor_dates": anchor_dates, "window_days": 5, "limit": 40}
                    )
                    market_rows = [{**record, "record_kind": "market_data"} for record in market_result["records"]]
                    evidence_bundle.quantitative_records = _merge_records(
                        evidence_bundle.quantitative_records,
                        market_rows,
                        ["date", "record_kind"],
                    )
                    evidence_bundle.tool_results.append(ToolResult.model_validate(market_result))
                    evidence_bundle.retrieval_notes.append(market_result["summary"])
                    execution_log.append("retrieve_market_data_tool")
        except Exception as exc:
            evidence_bundle.retrieval_notes.append(f"Refresh skipped or failed: {exc}")

    return {
        "company_ticker": ticker,
        "research_plan": plan.model_dump(),
        "evidence_bundle": evidence_bundle.model_dump(),
        "loop_count": next_loop_count,
        "retry_requested": False,
        "execution_log": execution_log,
    }


def _eda_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    plan = ResearchPlan.model_validate(state["research_plan"])
    evidence_bundle = EvidenceBundle.model_validate(state["evidence_bundle"])
    execution_log = list(state.get("execution_log", []))

    quantitative_records = evidence_bundle.quantitative_records
    metric_rows = [record for record in quantitative_records if record.get("record_kind") == "financial_metric"]
    market_rows = [record for record in quantitative_records if record.get("record_kind") == "market_data"]
    qualitative_records = evidence_bundle.qualitative_records

    findings: list[AnalysisFinding] = []
    notes: list[str] = []
    chart_spec: dict[str, Any] | None = None
    chart_artifact_path: str | None = None

    if plan.question_category in {"financial_trend", "mixed"} and metric_rows:
        trend_result = financial_trend_tool.invoke({"ticker": ticker, "metric_rows": metric_rows})
        findings.extend(AnalysisFinding.model_validate(finding) for finding in trend_result["findings"])
        if trend_result.get("artifact_path"):
            notes.append(f"Financial trend artifact saved to {trend_result['artifact_path']}")
        execution_log.append("financial_trend_tool")

        first_metric = metric_rows[0].get("metric_name") if metric_rows else None
        if first_metric:
            chart_rows = [
                {"fiscal_period": row.get("fiscal_period"), "metric_value": row.get("metric_value")}
                for row in metric_rows
                if row.get("metric_name") == first_metric
            ]
            chart_result = chart_tool.invoke(
                {
                    "ticker": ticker,
                    "title": f"{first_metric.replace('_', ' ').title()} trend",
                    "rows": list(reversed(chart_rows)),
                    "x_field": "fiscal_period",
                    "y_field": "metric_value",
                }
            )
            chart_spec = chart_result.get("chart_spec")
            chart_artifact_path = chart_result.get("artifact_path")
            execution_log.append("chart_tool")

    if plan.question_category in {"market_reaction", "mixed"} and market_rows:
        market_result = market_reaction_tool.invoke(
            {
                "ticker": ticker,
                "market_rows": market_rows,
                "anchor_dates": infer_anchor_dates_from_metrics(metric_rows),
            }
        )
        findings.extend(AnalysisFinding.model_validate(finding) for finding in market_result["findings"])
        if not chart_spec and market_rows:
            chart_rows = [
                {"date": str(row.get("date", ""))[:10], "close": row.get("close")}
                for row in market_rows
                if row.get("close") is not None
            ]
            chart_result = chart_tool.invoke(
                {
                    "ticker": ticker,
                    "title": "NVDA market window",
                    "rows": chart_rows,
                    "x_field": "date",
                    "y_field": "close",
                }
            )
            chart_spec = chart_result.get("chart_spec")
            chart_artifact_path = chart_result.get("artifact_path")
            execution_log.append("chart_tool")
        execution_log.append("market_reaction_tool")

    if plan.question_category in {"risk_narrative", "mixed"} and qualitative_records:
        text_result = text_theme_tool.invoke({"ticker": ticker, "chunk_rows": qualitative_records, "question": question})
        findings.extend(AnalysisFinding.model_validate(finding) for finding in text_result["findings"])
        execution_log.append("text_theme_tool")

    missing_modalities: list[str] = []
    requires_additional_research = False
    if plan.question_category == "mixed":
        if not qualitative_records:
            missing_modalities.append("qualitative")
        if not quantitative_records:
            missing_modalities.append("quantitative")
    if not findings:
        if not qualitative_records:
            missing_modalities.append("qualitative")
        if not quantitative_records:
            missing_modalities.append("quantitative")

    if missing_modalities and state.get("loop_count", 0) < 1:
        requires_additional_research = True
        notes.append(f"EDA requested another retrieval pass for: {', '.join(sorted(set(missing_modalities)))}.")

    if LLM.available and findings:
        prompt = (
            f"Question: {question}\n"
            f"Research plan: {json.dumps(plan.model_dump(mode='json'), indent=2, default=str)}\n"
            f"Current findings: {json.dumps([finding.model_dump(mode='json') for finding in findings], indent=2, default=str)}\n"
            "Write one short note describing the strongest EDA takeaway."
        )
        llm_note = LLM.complete(system_prompt=EDA_SYSTEM_PROMPT, user_prompt=prompt, max_tokens=120)
        if llm_note:
            notes.append(llm_note.strip())

    analysis_bundle = AnalysisBundle(
        company_ticker=ticker,
        question=question,
        findings=findings,
        notes=notes,
        requires_additional_research=requires_additional_research,
        missing_modalities=sorted(set(missing_modalities)),
    )

    return {
        "analysis_bundle": analysis_bundle.model_dump(),
        "chart_spec": chart_spec,
        "chart_artifact_path": chart_artifact_path,
        "missing_modalities": analysis_bundle.missing_modalities,
        "retry_requested": requires_additional_research,
        "execution_log": execution_log,
    }


def _route_after_eda(state: AgentState) -> str:
    analysis_bundle = AnalysisBundle.model_validate(state["analysis_bundle"])
    if analysis_bundle.requires_additional_research and state.get("loop_count", 0) < 1:
        return "collector"
    return "analyst"


def _analyst_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    evidence_bundle = EvidenceBundle.model_validate(state["evidence_bundle"])
    analysis_bundle = AnalysisBundle.model_validate(state["analysis_bundle"])
    execution_log = list(state.get("execution_log", []))

    llm_summary: str | None = None
    if LLM.available:
        prompt = (
            f"Question: {question}\n"
            f"Evidence bundle: {json.dumps(evidence_bundle.model_dump(mode='json'), indent=2, default=str)}\n"
            f"EDA findings: {json.dumps(analysis_bundle.model_dump(mode='json'), indent=2, default=str)}\n"
            "Write a concise grounded analyst answer with 2-3 short paragraphs."
        )
        llm_summary = LLM.complete(system_prompt=ANALYST_SYSTEM_PROMPT, user_prompt=prompt, max_tokens=350)

    final_answer, memo_artifact_path = final_answer_builder(
        ticker=ticker,
        question=question,
        findings=[finding.model_dump() for finding in analysis_bundle.findings],
        evidence_bundle=evidence_bundle.model_dump(),
        llm_summary=llm_summary,
    )
    execution_log.append("final_answer_builder")

    return {
        "final_answer": final_answer.model_dump(),
        "memo_artifact_path": memo_artifact_path,
        "execution_log": execution_log,
    }


class _FallbackWorkflow:
    def invoke(self, initial_state: AgentState) -> AgentState:
        state = dict(initial_state)
        state.update(_collector_node(state))
        state.update(_eda_node(state))
        if _route_after_eda(state) == "collector":
            state.update(_collector_node(state))
            state.update(_eda_node(state))
        state.update(_analyst_node(state))
        return state


def build_workflow():
    if StateGraph is None:
        logger.info("LangGraph is unavailable in this environment; using fallback sequential workflow.")
        return _FallbackWorkflow()

    graph = StateGraph(AgentState)
    graph.add_node("collector", _collector_node)
    graph.add_node("eda", _eda_node)
    graph.add_node("analyst", _analyst_node)
    graph.add_edge(START, "collector")
    graph.add_edge("collector", "eda")
    graph.add_conditional_edges("eda", _route_after_eda, {"collector": "collector", "analyst": "analyst"})
    graph.add_edge("analyst", END)
    return graph.compile()


WORKFLOW = build_workflow()


def run_analyst_workflow(question: str, company_ticker: str | None = None) -> dict[str, Any]:
    ticker = company_ticker or get_settings().default_ticker
    initial_state: AgentState = {
        "question": question,
        "company_ticker": ticker,
        "loop_count": 0,
        "execution_log": [],
    }

    return WORKFLOW.invoke(initial_state)
