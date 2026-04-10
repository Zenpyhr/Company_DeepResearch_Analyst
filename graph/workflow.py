from __future__ import annotations

import json
import re
from copy import deepcopy
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
from prompts.agent_prompts import ANALYST_SYSTEM_PROMPT, EDA_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from schemas.models import (
    AnalysisBundle,
    AnalysisFinding,
    EDAPlan,
    EvidenceBundle,
    OrchestrationPlan,
    ResearchPlan,
    RetrievalPlan,
    RetryDecision,
    ToolResult,
)
from storage.query_service import infer_anchor_dates_from_metrics


logger = get_logger(__name__)

ALLOWED_SOURCE_TYPES = {"sec_filing", "press_release"}
ALLOWED_EDA_TOOLS = {"financial_trend_tool", "market_reaction_tool", "text_theme_tool", "chart_tool"}


class AgentState(TypedDict, total=False):
    question: str
    company_ticker: str
    orchestration_plan: dict[str, Any]
    research_plan: dict[str, Any]
    evidence_bundle: dict[str, Any]
    analysis_bundle: dict[str, Any]
    retry_decision: dict[str, Any]
    final_answer: dict[str, Any]
    chart_spec: dict[str, Any] | None
    chart_artifact_path: str | None
    memo_artifact_path: str | None
    loop_count: int
    refresh_attempted: bool
    retry_requested: bool
    routing_source: str
    selected_tools: list[str]
    requested_sources: list[str]
    selected_sources: list[str]
    retry_reason: str | None
    clarification_needed: bool
    clarification_question: str | None
    clarification_reason: str | None
    execution_log: list[str]


LLM = OptionalLLM()


def _contains_term(lowered: str, tokens: set[str], terms: set[str]) -> bool:
    for term in terms:
        if " " in term:
            if term in lowered:
                return True
        elif term in tokens:
            return True
    return False


def _classify_question(question: str) -> tuple[str, str]:
    lowered = question.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    financial_keywords = {"revenue", "income", "profit", "eps", "cash", "margin", "quarter", "quarters", "financial", "growth"}
    market_keywords = {"stock", "price", "market", "volume", "return", "reaction", "trading", "shares"}
    qualitative_keywords = {"risk", "risks", "narrative", "management", "filing", "filings", "press", "release", "releases", "business", "strategy", "demand"}
    mixed_keywords = {"memo", "thesis", "outlook", "analyst", "growth versus risk", "growth vs risk"}

    has_financial = _contains_term(lowered, tokens, financial_keywords)
    has_market = _contains_term(lowered, tokens, market_keywords)
    has_qualitative = _contains_term(lowered, tokens, qualitative_keywords)
    has_mixed = _contains_term(lowered, tokens, mixed_keywords)

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


def _derive_sub_intents(question: str, question_category: str, question_type: str) -> list[str]:
    lowered = question.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    intents: list[str] = []

    financial_terms = {"revenue", "income", "profit", "eps", "cash", "margin", "growth", "financial", "quarter", "quarters"}
    market_terms = {"stock", "price", "market", "reaction", "return", "trading", "shares"}
    risk_terms = {"risk", "risks", "filing", "filings", "press", "release", "releases", "narrative", "demand", "strategy"}

    if question_category == "mixed":
        if any(term in lowered or term in tokens for term in financial_terms):
            intents.append("financial_trend")
        if any(term in lowered or term in tokens for term in risk_terms):
            intents.append("risk_narrative")
        if any(term in lowered or term in tokens for term in market_terms):
            intents.append("market_reaction")
        if not intents:
            intents = ["financial_trend", "risk_narrative"] if question_type == "mixed" else ["financial_trend"]
        return intents

    if question_category in {"financial_trend", "market_reaction", "risk_narrative"}:
        return [question_category]
    return ["financial_trend"]


def _detect_clarification_need(question: str) -> tuple[bool, str | None, str | None]:
    lowered = question.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if not lowered:
        return True, "What do you want to analyze: financial performance, market reaction, risk disclosures, or growth narrative?", "The question is empty."

    generic_patterns = [
        "tell me about",
        "what's going on",
        "whats going on",
        "what is going on",
        "give me an update",
        "analyze nvidia",
        "look at nvidia",
    ]
    domain_tokens = {
        "revenue",
        "income",
        "profit",
        "eps",
        "market",
        "stock",
        "price",
        "reaction",
        "risk",
        "risks",
        "filing",
        "filings",
        "press",
        "release",
        "growth",
        "demand",
        "guidance",
        "margin",
        "cash",
    }
    if any(pattern in lowered for pattern in generic_patterns):
        return (
            True,
            "Do you want to focus on financial performance, market reaction, risk disclosures, or the growth narrative?",
            "The question is broad and does not clearly state which type of analysis matters most.",
        )
    if len(tokens) <= 4 and not any(token in domain_tokens for token in tokens):
        return (
            True,
            "Could you clarify whether you want financials, stock reaction, risks, or growth-related evidence?",
            "The question is too short to choose the best evidence path confidently.",
        )
    return False, None, None


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _heuristic_retrieval_plan(question: str, question_category: str, question_type: str) -> RetrievalPlan:
    lowered = question.lower()
    source_types = ["sec_filing", "press_release"] if question_type in {"qualitative", "mixed"} else []
    if question_category == "risk_narrative" and "filing" in lowered:
        source_types = ["sec_filing"]
    explicit_market_request = any(
        token in lowered for token in ["stock", "price", "market", "reaction"]
    )
    needs_market_data = question_category == "market_reaction" or explicit_market_request
    metric_names = _metric_names_for_question(question) if question_type in {"quantitative", "mixed"} else []
    return RetrievalPlan(
        needs_qualitative=question_type in {"qualitative", "mixed"},
        needs_quantitative=question_type in {"quantitative", "mixed"},
        source_types=[source for source in source_types if source in ALLOWED_SOURCE_TYPES],
        metric_names=metric_names,
        limit_per_metric=4,
        needs_market_data=needs_market_data,
        anchor_date_count=1 if question_category == "market_reaction" else 2,
        market_window_days=2 if needs_market_data else 3,
    )


def _heuristic_eda_plan(question: str, question_category: str, retrieval_plan: RetrievalPlan) -> EDAPlan:
    selected_tools: list[str] = []
    if question_category in {"financial_trend", "mixed"} and retrieval_plan.needs_quantitative:
        selected_tools.append("financial_trend_tool")
    if question_category in {"market_reaction", "mixed"} and retrieval_plan.needs_market_data:
        selected_tools.append("market_reaction_tool")
    if question_category in {"risk_narrative", "mixed"} and retrieval_plan.needs_qualitative:
        selected_tools.append("text_theme_tool")
    if any(tool in selected_tools for tool in ["financial_trend_tool", "market_reaction_tool"]):
        selected_tools.append("chart_tool")
    chart_metric = retrieval_plan.metric_names[0] if retrieval_plan.metric_names and "financial_trend_tool" in selected_tools else None
    return EDAPlan(
        selected_tools=selected_tools,
        chart_metric=chart_metric,
        notes=["Heuristic EDA planning is active unless the LLM planner overrides it."],
    )


def _heuristic_orchestration_plan(question: str, ticker: str) -> OrchestrationPlan:
    question_category, question_type = _classify_question(question)
    sub_intents = _derive_sub_intents(question, question_category, question_type)
    retrieval_plan = _heuristic_retrieval_plan(question, question_category, question_type)
    eda_plan = _heuristic_eda_plan(question, question_category, retrieval_plan)
    return OrchestrationPlan(
        company_ticker=ticker,
        question=question,
        question_type=question_type,  # type: ignore[arg-type]
        question_category=question_category,  # type: ignore[arg-type]
        sub_intents=sub_intents,  # type: ignore[arg-type]
        retrieval_plan=retrieval_plan,
        eda_plan=eda_plan,
        retry_policy={"allow_retry": True, "max_retries": 1},
        notes=[
            f"Heuristic category: {question_category}",
            f"Heuristic type: {question_type}",
            f"Heuristic sub-intents: {', '.join(sub_intents)}",
        ],
        confidence_notes="Deterministic planning fallback is available if LLM planning fails.",
        planning_source="heuristic",
    )


def _normalize_orchestration_plan(payload: dict[str, Any], fallback: OrchestrationPlan) -> OrchestrationPlan | None:
    merged = _deep_merge(fallback.model_dump(mode="json"), payload)
    retrieval_plan = merged.get("retrieval_plan", {})
    eda_plan = merged.get("eda_plan", {})

    retrieval_plan["source_types"] = [
        source for source in retrieval_plan.get("source_types", []) if source in ALLOWED_SOURCE_TYPES
    ]
    if retrieval_plan.get("needs_qualitative") and not retrieval_plan["source_types"]:
        retrieval_plan["source_types"] = fallback.retrieval_plan.source_types
    retrieval_plan["limit_per_metric"] = max(1, min(int(retrieval_plan.get("limit_per_metric", 4)), 12))
    retrieval_plan["anchor_date_count"] = max(1, min(int(retrieval_plan.get("anchor_date_count", 1)), 4))
    retrieval_plan["market_window_days"] = max(1, min(int(retrieval_plan.get("market_window_days", 2)), 30))
    retrieval_plan["metric_names"] = [str(metric) for metric in retrieval_plan.get("metric_names", []) if str(metric).strip()]
    if retrieval_plan.get("needs_quantitative") and not retrieval_plan["metric_names"]:
        retrieval_plan["metric_names"] = fallback.retrieval_plan.metric_names
    if not retrieval_plan.get("needs_quantitative"):
        retrieval_plan["needs_market_data"] = False
    lowered_question = fallback.question.lower()
    explicit_market_request = any(token in lowered_question for token in ["stock", "price", "market", "reaction"])
    if fallback.question_category != "market_reaction" and not explicit_market_request:
        retrieval_plan["needs_market_data"] = False

    eda_plan["selected_tools"] = [tool for tool in eda_plan.get("selected_tools", []) if tool in ALLOWED_EDA_TOOLS]
    if not eda_plan["selected_tools"]:
        eda_plan["selected_tools"] = fallback.eda_plan.selected_tools
    if not retrieval_plan.get("needs_qualitative"):
        eda_plan["selected_tools"] = [tool for tool in eda_plan["selected_tools"] if tool != "text_theme_tool"]
    if not retrieval_plan.get("needs_market_data"):
        eda_plan["selected_tools"] = [tool for tool in eda_plan["selected_tools"] if tool != "market_reaction_tool"]
    if not retrieval_plan.get("needs_quantitative"):
        eda_plan["selected_tools"] = [tool for tool in eda_plan["selected_tools"] if tool != "financial_trend_tool"]
    if "chart_tool" in eda_plan["selected_tools"] and not any(
        tool in eda_plan["selected_tools"] for tool in ["financial_trend_tool", "market_reaction_tool"]
    ):
        eda_plan["selected_tools"] = [tool for tool in eda_plan["selected_tools"] if tool != "chart_tool"]
    chart_metric = eda_plan.get("chart_metric")
    if chart_metric and chart_metric not in retrieval_plan.get("metric_names", []):
        eda_plan["chart_metric"] = retrieval_plan.get("metric_names", [None])[0]
    if "chart_tool" in eda_plan["selected_tools"] and not eda_plan.get("chart_metric") and retrieval_plan.get("metric_names"):
        eda_plan["chart_metric"] = retrieval_plan["metric_names"][0]
    if "chart_tool" not in eda_plan["selected_tools"]:
        eda_plan["chart_metric"] = None

    merged["retrieval_plan"] = retrieval_plan
    merged["eda_plan"] = eda_plan
    if not merged.get("sub_intents"):
        merged["sub_intents"] = fallback.sub_intents
    merged["planning_source"] = "llm"

    try:
        return OrchestrationPlan.model_validate(merged)
    except Exception as exc:  # pragma: no cover - validation path exercised indirectly
        logger.warning("LLM orchestration plan validation failed, falling back to heuristic plan: %s", exc)
        return None


def _llm_orchestration_plan(question: str, ticker: str, fallback: OrchestrationPlan) -> OrchestrationPlan | None:
    if not LLM.available:
        return None

    user_prompt = (
        f"Question: {question}\n"
        f"Ticker: {ticker}\n"
        f"Fallback heuristic plan JSON: {json.dumps(fallback.model_dump(mode='json'), indent=2)}\n"
        "Return JSON only with these top-level keys: "
        "question_category, question_type, sub_intents, retrieval_plan, eda_plan, retry_policy, notes, confidence_notes.\n"
        "Constraints:\n"
        "- question_category must be one of financial_trend, market_reaction, risk_narrative, mixed\n"
        "- question_type must be one of qualitative, quantitative, mixed\n"
        "- sub_intents must be a list containing any of financial_trend, market_reaction, risk_narrative\n"
        "- retrieval_plan.source_types can only include sec_filing and press_release\n"
        "- eda_plan.selected_tools can only include financial_trend_tool, market_reaction_tool, text_theme_tool, chart_tool\n"
        "- do not request market data unless the question genuinely needs price or reaction context\n"
        "- do not request both qualitative and quantitative unless both are actually needed\n"
        "- never answer the question; only plan the workflow\n"
    )
    payload = LLM.complete_json(system_prompt=ORCHESTRATOR_SYSTEM_PROMPT, user_prompt=user_prompt, max_tokens=700)
    if not payload:
        return None
    payload["company_ticker"] = ticker
    payload["question"] = question
    return _normalize_orchestration_plan(payload, fallback)


def _research_plan_from_orchestration(plan: OrchestrationPlan) -> ResearchPlan:
    goals: list[str] = []
    tools_to_call = list(plan.eda_plan.selected_tools)
    if plan.retrieval_plan.needs_qualitative:
        goals.append("Retrieve qualitative evidence from company documents")
        tools_to_call.insert(0, "retrieve_document_context_tool")
    if plan.retrieval_plan.needs_quantitative:
        goals.append("Retrieve quantitative financial evidence")
        tools_to_call.insert(0, "retrieve_financial_metrics_tool")
    if plan.retrieval_plan.needs_market_data:
        goals.append("Retrieve market evidence tied to the question")
        tools_to_call.insert(1 if tools_to_call and tools_to_call[0] == "retrieve_financial_metrics_tool" else 0, "retrieve_market_data_tool")
    if not goals:
        goals.append("Retrieve the minimum evidence needed to answer the question safely")
    note_text = " ".join(plan.notes).strip() or None
    return ResearchPlan(
        company_ticker=plan.company_ticker,
        question=plan.question,
        question_type=plan.question_type,
        question_category=plan.question_category,
        sub_intents=plan.sub_intents,
        goals=goals,
        tools_to_call=list(dict.fromkeys(tools_to_call)),
        notes=note_text,
    )


def _planner_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    clarification_needed, clarification_question, clarification_reason = _detect_clarification_need(question)
    heuristic_plan = _heuristic_orchestration_plan(question, ticker)
    orchestration_plan = _llm_orchestration_plan(question, ticker, heuristic_plan) or heuristic_plan
    if clarification_needed and clarification_reason:
        orchestration_plan = orchestration_plan.model_copy(
            update={
                "notes": [
                    *orchestration_plan.notes,
                    f"Clarification suggested: {clarification_reason}",
                ]
            }
        )
    research_plan = _research_plan_from_orchestration(orchestration_plan)
    routing_source = "llm" if orchestration_plan.planning_source == "llm" else "heuristic_fallback"
    selected_tools = list(dict.fromkeys(orchestration_plan.eda_plan.selected_tools))
    if orchestration_plan.retrieval_plan.needs_qualitative:
        selected_tools.insert(0, "retrieve_document_context_tool")
    if orchestration_plan.retrieval_plan.needs_quantitative:
        selected_tools.insert(0, "retrieve_financial_metrics_tool")
    if orchestration_plan.retrieval_plan.needs_market_data:
        selected_tools.append("retrieve_market_data_tool")

    return {
        "company_ticker": ticker,
        "orchestration_plan": orchestration_plan.model_dump(),
        "research_plan": research_plan.model_dump(),
        "routing_source": routing_source,
        "selected_tools": list(dict.fromkeys(selected_tools)),
        "requested_sources": orchestration_plan.retrieval_plan.source_types,
        "selected_sources": orchestration_plan.retrieval_plan.source_types,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question,
        "clarification_reason": clarification_reason,
        "execution_log": [*state.get("execution_log", []), f"planner:{routing_source}"],
    }


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
    plan = OrchestrationPlan.model_validate(state["orchestration_plan"])
    retrieval_plan = plan.retrieval_plan
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
    retry_decision = RetryDecision.model_validate(state.get("retry_decision") or {})
    next_loop_count = state.get("loop_count", 0)
    if retry_decision.retry_requested and state.get("analysis_bundle") and next_loop_count < 1:
        next_loop_count += 1

    execution_log = list(state.get("execution_log", []))
    effective_source_types = retrieval_plan.source_types or []
    if retry_decision.missing_sources:
        effective_source_types = [
            source for source in dict.fromkeys([*effective_source_types, *retry_decision.missing_sources]) if source in ALLOWED_SOURCE_TYPES
        ]
    effective_metric_names = retrieval_plan.metric_names or []
    if retry_decision.missing_metrics:
        effective_metric_names = [metric for metric in dict.fromkeys([*effective_metric_names, *retry_decision.missing_metrics]) if metric]

    if retrieval_plan.needs_qualitative:
        qualitative_result = retrieve_document_context_tool.invoke(
            {
                "ticker": ticker,
                "question": question,
                "top_k": 6,
                "source_types": effective_source_types,
            }
        )
        evidence_bundle.qualitative_records = _merge_records(
            evidence_bundle.qualitative_records,
            qualitative_result["records"],
            ["id", "source_id", "chunk_order"],
        )
        evidence_bundle.tool_results.append(ToolResult.model_validate(qualitative_result))
        evidence_bundle.retrieval_notes.append(qualitative_result["summary"])
        execution_log.append("retrieve_document_context_tool")

    actual_retrieved_sources = sorted(
        {
            str(record.get("source_type"))
            for record in evidence_bundle.qualitative_records
            if record.get("source_type")
        }
    )

    if retrieval_plan.needs_quantitative:
        metrics_result = retrieve_financial_metrics_tool.invoke(
            {
                "ticker": ticker,
                "metric_names": effective_metric_names,
                "limit_per_metric": retrieval_plan.limit_per_metric,
            }
        )
        metric_rows = [{**record, "record_kind": "financial_metric"} for record in metrics_result["records"]]
        evidence_bundle.quantitative_records = _merge_records(
            evidence_bundle.quantitative_records,
            metric_rows,
            ["id", "record_kind"],
        )
        evidence_bundle.tool_results.append(ToolResult.model_validate(metrics_result))
        evidence_bundle.retrieval_notes.append(metrics_result["summary"])
        execution_log.append("retrieve_financial_metrics_tool")

        if retrieval_plan.needs_market_data:
            anchor_dates = infer_anchor_dates_from_metrics(
                metrics_result["records"],
                max_dates=retrieval_plan.anchor_date_count,
            )
            market_result = retrieve_market_data_tool.invoke(
                {
                    "ticker": ticker,
                    "anchor_dates": anchor_dates,
                    "window_days": retrieval_plan.market_window_days,
                    "limit": 20,
                }
            )
            market_rows = [{**record, "record_kind": "market_data"} for record in market_result["records"]]
            evidence_bundle.quantitative_records = _merge_records(
                evidence_bundle.quantitative_records,
                market_rows,
                ["id", "record_kind"],
            )
            evidence_bundle.tool_results.append(ToolResult.model_validate(market_result))
            evidence_bundle.retrieval_notes.append(market_result["summary"])
            execution_log.append("retrieve_market_data_tool")

    if (
        not evidence_bundle.qualitative_records
        and not evidence_bundle.quantitative_records
        and not state.get("refresh_attempted", False)
    ):
        try:
            refresh_result = refresh_company_data_tool.invoke({"ticker": ticker})
            evidence_bundle.retrieval_notes.append(
                f"Refresh/process step completed: {refresh_result['refresh_summary']} and {refresh_result['process_summary']}"
            )
            execution_log.append("refresh_company_data_tool")
            return _collector_node(
                {
                    **state,
                    "loop_count": next_loop_count,
                    "refresh_attempted": True,
                    "evidence_bundle": evidence_bundle.model_dump(),
                    "execution_log": execution_log,
                    "retry_decision": RetryDecision(retry_requested=False).model_dump(),
                }
            )
        except Exception as exc:
            evidence_bundle.retrieval_notes.append(f"Refresh skipped or failed: {exc}")

    return {
        "company_ticker": ticker,
        "evidence_bundle": evidence_bundle.model_dump(),
        "loop_count": next_loop_count,
        "retry_requested": False,
        "execution_log": execution_log,
        "requested_sources": effective_source_types,
        "selected_sources": actual_retrieved_sources,
    }


def _build_retry_decision(
    *,
    plan: OrchestrationPlan,
    evidence_bundle: EvidenceBundle,
    findings: list[AnalysisFinding],
    loop_count: int,
) -> RetryDecision:
    missing_modalities: list[str] = []
    missing_sources: list[str] = []
    missing_metrics: list[str] = []
    reasons: list[str] = []

    if plan.retrieval_plan.needs_qualitative and not evidence_bundle.qualitative_records:
        missing_modalities.append("qualitative")
        missing_sources.extend(plan.retrieval_plan.source_types)
    if plan.retrieval_plan.needs_quantitative and not evidence_bundle.quantitative_records:
        missing_modalities.append("quantitative")
        missing_metrics.extend(plan.retrieval_plan.metric_names)

    present_tools = {finding.finding_type for finding in findings}
    if "financial_trend_tool" in plan.eda_plan.selected_tools and "financial_trend" not in present_tools:
        missing_metrics.extend(plan.retrieval_plan.metric_names)
        reasons.append("Financial analysis tool had insufficient metric evidence.")
    if "market_reaction_tool" in plan.eda_plan.selected_tools and "market_reaction" not in present_tools:
        missing_modalities.append("quantitative")
        reasons.append("Market reaction tool had insufficient market rows.")
    if "text_theme_tool" in plan.eda_plan.selected_tools and "text_theme" not in present_tools:
        missing_modalities.append("qualitative")
        missing_sources.extend(plan.retrieval_plan.source_types)
        reasons.append("Text analysis tool had insufficient qualitative evidence.")

    retry_requested = bool((missing_modalities or reasons) and loop_count < int(plan.retry_policy.get("max_retries", 1)))
    if not reasons and retry_requested:
        reasons.append("Evidence bundle was incomplete for the planned analysis path.")

    return RetryDecision(
        retry_requested=retry_requested and bool(plan.retry_policy.get("allow_retry", True)),
        missing_modalities=sorted(set(missing_modalities)),
        missing_sources=sorted(set(missing_sources)),
        missing_metrics=sorted(set(missing_metrics)),
        reason=" ".join(reasons).strip() or None,
    )


def _llm_retry_decision(
    *,
    plan: OrchestrationPlan,
    evidence_bundle: EvidenceBundle,
    analysis_findings: list[AnalysisFinding],
    fallback: RetryDecision,
    loop_count: int,
) -> RetryDecision:
    if not LLM.available:
        return fallback

    prompt = (
        f"Question: {plan.question}\n"
        f"Loop count: {loop_count}\n"
        f"Orchestration plan: {json.dumps(plan.model_dump(mode='json'), indent=2)}\n"
        f"Evidence counts: qualitative={len(evidence_bundle.qualitative_records)}, quantitative={len(evidence_bundle.quantitative_records)}\n"
        f"Analysis findings: {json.dumps([finding.model_dump(mode='json') for finding in analysis_findings], indent=2)}\n"
        f"Fallback retry decision JSON: {json.dumps(fallback.model_dump(mode='json'), indent=2)}\n"
        "Return JSON only with keys: retry_requested, missing_modalities, missing_sources, missing_metrics, reason.\n"
        "Never request more than one additional retry and do not answer the question."
    )
    payload = LLM.complete_json(system_prompt=EDA_SYSTEM_PROMPT, user_prompt=prompt, max_tokens=250)
    if not payload:
        return fallback

    merged = _deep_merge(fallback.model_dump(mode="json"), payload)
    merged["retry_requested"] = bool(merged.get("retry_requested")) and loop_count < int(plan.retry_policy.get("max_retries", 1))
    try:
        return RetryDecision.model_validate(merged)
    except Exception as exc:  # pragma: no cover - validation path exercised indirectly
        logger.warning("LLM retry decision validation failed, falling back to deterministic retry logic: %s", exc)
        return fallback


def _eda_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    plan = OrchestrationPlan.model_validate(state["orchestration_plan"])
    evidence_bundle = EvidenceBundle.model_validate(state["evidence_bundle"])
    execution_log = list(state.get("execution_log", []))

    quantitative_records = evidence_bundle.quantitative_records
    metric_rows = [record for record in quantitative_records if record.get("record_kind") == "financial_metric"]
    market_rows = [record for record in quantitative_records if record.get("record_kind") == "market_data"]
    qualitative_records = evidence_bundle.qualitative_records

    findings: list[AnalysisFinding] = []
    notes: list[str] = list(plan.eda_plan.notes)
    chart_spec: dict[str, Any] | None = None
    chart_artifact_path: str | None = None
    selected_tools_executed: list[str] = []

    for tool_name in plan.eda_plan.selected_tools:
        if tool_name == "financial_trend_tool":
            if not metric_rows:
                notes.append("Skipped financial_trend_tool because no financial metric rows were available.")
                continue
            trend_result = financial_trend_tool.invoke({"ticker": ticker, "metric_rows": metric_rows})
            findings.extend(AnalysisFinding.model_validate(finding) for finding in trend_result["findings"])
            if trend_result.get("artifact_path"):
                notes.append(f"Financial trend artifact saved to {trend_result['artifact_path']}")
            execution_log.append("financial_trend_tool")
            selected_tools_executed.append("financial_trend_tool")
            continue

        if tool_name == "market_reaction_tool":
            if not market_rows:
                notes.append("Skipped market_reaction_tool because no market rows were available.")
                continue
            market_result = market_reaction_tool.invoke(
                {
                    "ticker": ticker,
                    "market_rows": market_rows,
                    "anchor_dates": infer_anchor_dates_from_metrics(
                        metric_rows,
                        max_dates=plan.retrieval_plan.anchor_date_count,
                    ),
                }
            )
            findings.extend(AnalysisFinding.model_validate(finding) for finding in market_result["findings"])
            execution_log.append("market_reaction_tool")
            selected_tools_executed.append("market_reaction_tool")
            continue

        if tool_name == "text_theme_tool":
            if not qualitative_records:
                notes.append("Skipped text_theme_tool because no qualitative records were available.")
                continue
            text_result = text_theme_tool.invoke({"ticker": ticker, "chunk_rows": qualitative_records, "question": question})
            findings.extend(AnalysisFinding.model_validate(finding) for finding in text_result["findings"])
            execution_log.append("text_theme_tool")
            selected_tools_executed.append("text_theme_tool")
            continue

        if tool_name == "chart_tool":
            chart_metric = plan.eda_plan.chart_metric
            if chart_metric and metric_rows:
                chart_rows = [
                    {"fiscal_period": row.get("fiscal_period"), "metric_value": row.get("metric_value")}
                    for row in metric_rows
                    if row.get("metric_name") == chart_metric
                ]
                if chart_rows:
                    chart_result = chart_tool.invoke(
                        {
                            "ticker": ticker,
                            "title": f"{chart_metric.replace('_', ' ').title()} trend",
                            "rows": list(reversed(chart_rows)),
                            "x_field": "fiscal_period",
                            "y_field": "metric_value",
                        }
                    )
                    chart_spec = chart_result.get("chart_spec")
                    chart_artifact_path = chart_result.get("artifact_path")
                    execution_log.append("chart_tool")
                    selected_tools_executed.append("chart_tool")
                    continue
            if market_rows and not chart_spec:
                chart_rows = [
                    {"date": str(row.get("date", ""))[:10], "close": row.get("close")}
                    for row in market_rows
                    if row.get("close") is not None
                ]
                if chart_rows:
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
                    selected_tools_executed.append("chart_tool")
                else:
                    notes.append("Skipped chart_tool because there were no chartable rows.")
            else:
                notes.append("Skipped chart_tool because no chartable rows were available.")

    fallback_retry = _build_retry_decision(
        plan=plan,
        evidence_bundle=evidence_bundle,
        findings=findings,
        loop_count=state.get("loop_count", 0),
    )
    retry_decision = _llm_retry_decision(
        plan=plan,
        evidence_bundle=evidence_bundle,
        analysis_findings=findings,
        fallback=fallback_retry,
        loop_count=state.get("loop_count", 0),
    )
    if retry_decision.reason:
        notes.append(retry_decision.reason)

    if LLM.available and findings:
        prompt = (
            f"Question: {question}\n"
            f"Orchestration plan: {json.dumps(plan.model_dump(mode='json'), indent=2)}\n"
            f"Current findings: {json.dumps([finding.model_dump(mode='json') for finding in findings], indent=2)}\n"
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
        requires_additional_research=retry_decision.retry_requested,
        missing_modalities=retry_decision.missing_modalities,
    )

    return {
        "analysis_bundle": analysis_bundle.model_dump(),
        "retry_decision": retry_decision.model_dump(),
        "chart_spec": chart_spec,
        "chart_artifact_path": chart_artifact_path,
        "retry_requested": retry_decision.retry_requested,
        "retry_reason": retry_decision.reason,
        "selected_tools": list(dict.fromkeys(state.get("selected_tools", []) + selected_tools_executed)),
        "execution_log": execution_log,
    }


def _route_after_eda(state: AgentState) -> str:
    retry_decision = RetryDecision.model_validate(state.get("retry_decision") or {})
    if retry_decision.retry_requested and state.get("loop_count", 0) < 1:
        return "collector"
    return "analyst"


def _analyst_node(state: AgentState) -> AgentState:
    question = state["question"]
    ticker = state.get("company_ticker") or get_settings().default_ticker
    evidence_bundle = EvidenceBundle.model_validate(state["evidence_bundle"])
    analysis_bundle = AnalysisBundle.model_validate(state["analysis_bundle"])
    orchestration_plan = OrchestrationPlan.model_validate(state["orchestration_plan"])
    execution_log = list(state.get("execution_log", []))

    llm_summary: str | None = None
    if LLM.available:
        prompt = (
            f"Question: {question}\n"
            f"Orchestration plan: {json.dumps(orchestration_plan.model_dump(mode='json'), indent=2)}\n"
            f"Evidence bundle: {json.dumps(evidence_bundle.model_dump(mode='json'), indent=2)}\n"
            f"EDA findings: {json.dumps(analysis_bundle.model_dump(mode='json'), indent=2)}\n"
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
        state.update(_planner_node(state))
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
    graph.add_node("planner", _planner_node)
    graph.add_node("collector", _collector_node)
    graph.add_node("eda", _eda_node)
    graph.add_node("analyst", _analyst_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "collector")
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
        "refresh_attempted": False,
        "execution_log": [],
    }
    return WORKFLOW.invoke(initial_state)
