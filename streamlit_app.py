from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from agents.tools import refresh_company_data_tool
from app.config import get_settings
from app.logging import get_logger
from graph.workflow import run_analyst_workflow
from schemas.models import AnalysisBundle, EvidenceBundle, FinalAnswer, ResearchPlan
from storage.bootstrap import bootstrap_storage


logger = get_logger(__name__)


def _render_collect_section(plan: ResearchPlan, evidence_bundle: EvidenceBundle) -> None:
    # This tab answers "what did the system collect before it started reasoning?"
    # Keeping the collection stage visible is important for the assignment and for debugging:
    # if the final answer looks weak, this section tells us whether the problem started with
    # missing evidence rather than bad analysis or synthesis.
    st.subheader("Collect")
    st.write("The Collector agent decides which evidence to retrieve before any conclusion is written.")
    st.markdown("**Research goals**")
    for goal in plan.goals:
        st.write(f"- {goal}")

    st.markdown("**Retrieval notes**")
    for note in evidence_bundle.retrieval_notes:
        st.write(f"- {note}")

    if evidence_bundle.tool_results:
        tool_rows = [
            {"tool_name": result.tool_name, "summary": result.summary, "status": result.status, "record_count": len(result.records)}
            for result in evidence_bundle.tool_results
        ]
        st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Qualitative records", len(evidence_bundle.qualitative_records))
    with col2:
        st.metric("Quantitative records", len(evidence_bundle.quantitative_records))


def _render_chart(chart_spec: dict[str, Any] | None) -> None:
    if not chart_spec:
        return
    values = chart_spec.get("data", {}).get("values", [])
    if not values:
        return
    df = pd.DataFrame(values)
    encoding = chart_spec.get("encoding", {})
    x_field = encoding.get("x", {}).get("field")
    y_field = encoding.get("y", {}).get("field")
    title = chart_spec.get("title", "Analysis chart")
    if x_field and y_field and x_field in df.columns and y_field in df.columns:
        st.markdown(f"**{title}**")
        line_df = df[[x_field, y_field]].dropna()
        if not line_df.empty:
            st.line_chart(line_df.set_index(x_field), height=320)


def _render_explore_section(analysis_bundle: AnalysisBundle, chart_spec: dict[str, Any] | None, chart_artifact_path: str | None) -> None:
    # This tab is the EDA checkpoint. It shows tool-produced findings before the final memo,
    # so we can verify the system actually analyzed the retrieved data instead of jumping
    # straight from raw evidence to a conclusion.
    st.subheader("Explore / Analyze")
    st.write("The EDA agent uses tool calls over the retrieved data before the analyst hypothesis is formed.")
    for finding in analysis_bundle.findings:
        with st.container(border=True):
            st.markdown(f"**{finding.finding_type.replace('_', ' ').title()}**")
            st.write(finding.summary)
            if finding.metrics:
                st.json(finding.metrics)
            if finding.supporting_records:
                st.dataframe(pd.DataFrame(finding.supporting_records[:5]), use_container_width=True)

    if analysis_bundle.notes:
        st.markdown("**EDA notes**")
        for note in analysis_bundle.notes:
            st.write(f"- {note}")

    _render_chart(chart_spec)
    if chart_artifact_path:
        st.caption(f"Chart spec artifact: `{chart_artifact_path}`")


def _render_hypothesis_section(final_answer: FinalAnswer, memo_artifact_path: str | None) -> None:
    # This is the final analyst deliverable. By the time we render this tab, the workflow
    # has already completed collection and EDA, so this section should read like a synthesis
    # of earlier evidence rather than a brand-new source of facts.
    st.subheader("Hypothesis")
    st.write("The Analyst agent synthesizes the evidence and EDA findings into a grounded answer.")
    st.markdown(final_answer.answer)
    if final_answer.key_points:
        st.markdown("**Key points**")
        for point in final_answer.key_points:
            st.write(f"- {point}")
    if final_answer.sources:
        st.markdown("**Sources**")
        for source in final_answer.sources:
            st.write(f"- {source}")
    if final_answer.confidence_note:
        st.info(final_answer.confidence_note)
    if memo_artifact_path:
        st.caption(f"Memo artifact: `{memo_artifact_path}`")


def main() -> None:
    # `main()` is the easiest starting point for reading the app because it wires together
    # the full top-level user flow:
    # 1. ensure storage exists,
    # 2. render the Streamlit controls,
    # 3. run the workflow when the user clicks the button,
    # 4. unpack the structured stage outputs,
    # 5. render Collect -> Explore/Analyze -> Hypothesis tabs.
    bootstrap_storage()
    settings = get_settings()
    st.set_page_config(page_title="NVDA Analyst Agent", layout="wide")
    st.title("NVDA Analyst Agent")
    st.caption("Single-company analyst workflow with explicit Collect -> Explore -> Hypothesize stages.")

    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None

    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("Ticker", [settings.default_ticker], index=0)
        if st.button("Refresh Company Data", use_container_width=True):
            with st.spinner("Refreshing company data and processing documents..."):
                refresh_result = refresh_company_data_tool.invoke({"ticker": ticker})
            st.success("Refresh complete.")
            st.json(refresh_result)

        st.markdown("---")
        st.write("The live deployment target for this app is Cloud Run.")

    question = st.text_area(
        "Ask an analyst question",
        value="Do NVIDIA's recent financials support the AI growth narrative, and what risks still stand out?",
        height=120,
    )

    if st.button("Run Analyst Workflow", type="primary", use_container_width=True):
        # This is the single frontend call that kicks off the agent graph.
        # Everything interesting downstream happens inside `run_analyst_workflow()`:
        # classification, retrieval, EDA, synthesis, artifact writing, and optional retry.
        with st.spinner("Running Collector -> EDA -> Analyst workflow..."):
            st.session_state.workflow_result = run_analyst_workflow(question, company_ticker=ticker)

    result = st.session_state.workflow_result
    if not result:
        st.info("Refresh the data if needed, then ask a question to run the workflow.")
        return

    # The workflow returns plain dictionaries because that is the easiest shape to persist
    # through Streamlit session state and LangGraph state. We immediately convert them back
    # into typed Pydantic models here so the UI layer can rely on a stable contract.
    plan = ResearchPlan.model_validate(result["research_plan"])
    evidence_bundle = EvidenceBundle.model_validate(result["evidence_bundle"])
    analysis_bundle = AnalysisBundle.model_validate(result["analysis_bundle"])
    final_answer = FinalAnswer.model_validate(result["final_answer"])

    collect_tab, explore_tab, hypothesis_tab, debug_tab = st.tabs(["Collect", "Explore / Analyze", "Hypothesis", "Debug"])

    with collect_tab:
        _render_collect_section(plan, evidence_bundle)

    with explore_tab:
        _render_explore_section(analysis_bundle, result.get("chart_spec"), result.get("chart_artifact_path"))

    with hypothesis_tab:
        _render_hypothesis_section(final_answer, result.get("memo_artifact_path"))

    with debug_tab:
        st.markdown("**Execution log**")
        for entry in result.get("execution_log", []):
            st.write(f"- {entry}")
        st.markdown("**Workflow payload**")
        st.json(result)


if __name__ == "__main__":
    main()
