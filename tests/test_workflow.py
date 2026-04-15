from __future__ import annotations

import importlib
import os
import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4


class WorkflowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workspace_tmp = Path.cwd() / "data" / "test_tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        cls.temp_dir = workspace_tmp / f"workflow_{uuid4().hex}"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        data_dir = cls.temp_dir / "data"
        os.environ["DATA_DIR"] = str(data_dir)
        os.environ["SQLITE_DB_PATH"] = str(data_dir / "app.db")
        os.environ["DEFAULT_TICKER"] = "NVDA"
        os.environ["OPENAI_API_KEY"] = ""

        from app.config import get_settings

        get_settings.cache_clear()

        from storage.bootstrap import bootstrap_storage
        from storage.repositories import insert_chunk, insert_financial_metric, insert_source, upsert_market_record
        from schemas.models import Chunk, FinancialMetric, MarketRecord, Source

        bootstrap_storage()
        latest_date = datetime(2026, 2, 25, tzinfo=timezone.utc)
        previous_date = datetime(2025, 11, 25, tzinfo=timezone.utc)

        sec_source_id = insert_source(
            Source(
                company_ticker="NVDA",
                source_type="sec_filing",
                title="NVDA 10-K filed on 2026-02-25",
                source_url="https://example.com/sec",
                raw_path="data/example_sec.txt",
                metadata_json={"form_type": "10-K"},
            )
        )
        press_source_id = insert_source(
            Source(
                company_ticker="NVDA",
                source_type="press_release",
                title="NVIDIA expands AI infrastructure",
                source_url="https://example.com/press",
                raw_path="data/example_press.txt",
                metadata_json={},
            )
        )

        insert_chunk(
            Chunk(
                company_ticker="NVDA",
                source_id=sec_source_id,
                source_type="sec_filing",
                title="NVDA 10-K filed on 2026-02-25",
                source_url="https://example.com/sec",
                chunk_text="Item 1A. Risk Factors Competition, supply chain limits, and customer concentration remain important risks.",
                chunk_order=0,
                metadata_json={"section_label": "item_1a_risk_factors", "section_title": "Item 1A. Risk Factors"},
            )
        )
        insert_chunk(
            Chunk(
                company_ticker="NVDA",
                source_id=press_source_id,
                source_type="press_release",
                title="NVIDIA expands AI infrastructure",
                source_url="https://example.com/press",
                chunk_text="NVIDIA said AI infrastructure demand remained strong across data center customers.",
                chunk_order=0,
                metadata_json={"section_label": "full_text", "section_title": "NVIDIA expands AI infrastructure"},
            )
        )

        insert_financial_metric(
            FinancialMetric(
                company_ticker="NVDA",
                fiscal_period="2026-Q3",
                metric_name="revenue",
                metric_value=120.0,
                source_url="https://example.com/companyfacts",
                as_of_date=latest_date,
                metadata_json={},
            )
        )
        insert_financial_metric(
            FinancialMetric(
                company_ticker="NVDA",
                fiscal_period="2026-Q2",
                metric_name="revenue",
                metric_value=100.0,
                source_url="https://example.com/companyfacts",
                as_of_date=previous_date,
                metadata_json={},
            )
        )
        insert_financial_metric(
            FinancialMetric(
                company_ticker="NVDA",
                fiscal_period="2026-Q3",
                metric_name="net_income",
                metric_value=55.0,
                source_url="https://example.com/companyfacts",
                as_of_date=latest_date,
                metadata_json={},
            )
        )
        insert_financial_metric(
            FinancialMetric(
                company_ticker="NVDA",
                fiscal_period="2026-Q2",
                metric_name="net_income",
                metric_value=42.0,
                source_url="https://example.com/companyfacts",
                as_of_date=previous_date,
                metadata_json={},
            )
        )

        for day, close_value in enumerate([100.0, 103.0, 101.0, 106.0, 108.0], start=1):
            upsert_market_record(
                MarketRecord(
                    company_ticker="NVDA",
                    date=datetime(2026, 2, day, tzinfo=timezone.utc),
                    open=close_value - 1,
                    high=close_value + 1,
                    low=close_value - 2,
                    close=close_value,
                    volume=1_000_000 + day * 1000,
                    metadata_json={},
                )
            )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_document_retrieval_returns_relevant_chunk(self) -> None:
        from storage.query_service import search_document_chunks

        rows = search_document_chunks("NVDA", "What risks still stand out for NVIDIA?", top_k=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("risk", rows[0]["chunk_text"].lower())

    def test_financial_metric_fetch_preserves_multiple_periods(self) -> None:
        from storage.query_service import fetch_financial_metrics

        rows = fetch_financial_metrics("NVDA", metric_names=["revenue"], limit_per_metric=4)
        periods = [row["fiscal_period"] for row in rows]
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("2026-Q3", periods)
        self.assertIn("2026-Q2", periods)

    def test_mixed_document_retrieval_preserves_press_release_evidence(self) -> None:
        from storage.query_service import search_document_chunks

        rows = search_document_chunks(
            "NVDA",
            "What do recent filings and press releases suggest about growth versus risk for NVIDIA?",
            top_k=4,
        )
        source_types = {row["source_type"] for row in rows}
        self.assertIn("sec_filing", source_types)
        self.assertIn("press_release", source_types)

    def test_question_classifier_stays_focused_for_revenue_and_market_questions(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")

        revenue_category, revenue_type = graph_workflow._classify_question(  # type: ignore[attr-defined]
            "How has NVIDIA revenue changed over recent reported periods?"
        )
        market_category, market_type = graph_workflow._classify_question(  # type: ignore[attr-defined]
            "How did NVDA stock react around recent reporting periods?"
        )

        self.assertEqual(("financial_trend", "quantitative"), (revenue_category, revenue_type))
        self.assertEqual(("market_reaction", "quantitative"), (market_category, market_type))

    def test_planner_falls_back_to_heuristic_when_llm_plan_is_unavailable(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")

        with patch.object(graph_workflow, "_llm_orchestration_plan", return_value=None):
            result = graph_workflow._planner_node({"question": "How has NVIDIA revenue changed?", "company_ticker": "NVDA"})  # type: ignore[attr-defined]

        self.assertEqual("heuristic_fallback", result["routing_source"])
        self.assertIn("orchestration_plan", result)
        self.assertTrue(result["selected_tools"])

    def test_planner_accepts_llm_orchestration_plan(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        fallback = graph_workflow._heuristic_orchestration_plan("What risks does NVIDIA emphasize most in recent filings?", "NVDA")  # type: ignore[attr-defined]
        llm_plan = fallback.model_copy(
            update={
                "planning_source": "llm",
                "eda_plan": fallback.eda_plan.model_copy(update={"selected_tools": ["text_theme_tool"]}),
            }
        )

        with patch.object(graph_workflow, "_llm_orchestration_plan", return_value=llm_plan):
            result = graph_workflow._planner_node(
                {"question": "What risks does NVIDIA emphasize most in recent filings?", "company_ticker": "NVDA"}
            )  # type: ignore[attr-defined]

        self.assertEqual("llm", result["routing_source"])
        self.assertEqual("llm", result["orchestration_plan"]["planning_source"])
        self.assertIn("text_theme_tool", result["selected_tools"])

    def test_anchor_dates_prefer_filed_dates(self) -> None:
        from storage.query_service import fetch_financial_metrics, infer_anchor_dates_from_metrics

        metric_rows = fetch_financial_metrics("NVDA", metric_names=["revenue"], limit_per_metric=4)
        anchors = infer_anchor_dates_from_metrics(metric_rows, max_dates=2)
        self.assertTrue(anchors)
        self.assertEqual("2026-02-25", anchors[0])

    def test_text_theme_tool_surfaces_concrete_risk_themes(self) -> None:
        from storage.query_service import search_document_chunks
        from agents.tools import text_theme_tool

        rows = search_document_chunks("NVDA", "What risks does NVIDIA emphasize most in recent filings?", top_k=4)
        result = text_theme_tool.invoke(
            {"ticker": "NVDA", "chunk_rows": rows, "question": "What risks does NVIDIA emphasize most in recent filings?"}
        )
        self.assertTrue(result["findings"])
        top_themes = result["findings"][0]["metrics"].get("top_themes", [])
        self.assertTrue(top_themes)

    def test_text_theme_tool_returns_source_specific_theme_breakdown(self) -> None:
        from agents.tools import text_theme_tool

        rows = [
            {
                "source_type": "sec_filing",
                "chunk_text": "Supply constraints and competition remain important risk factors for data center growth.",
                "metadata_json": {"section_label": "item_1a_risk_factors"},
            },
            {
                "source_type": "press_release",
                "chunk_text": "NVIDIA highlighted AI demand and data center partnerships in the latest announcement.",
                "metadata_json": {"section_label": "press_release"},
            },
        ]

        result = text_theme_tool.invoke(
            {
                "ticker": "NVDA",
                "chunk_rows": rows,
                "question": "What do recent filings and press releases suggest about growth versus risk for NVIDIA?",
            }
        )

        metrics = result["findings"][0]["metrics"]
        self.assertTrue(metrics.get("filing_top_themes"))
        self.assertTrue(metrics.get("press_release_top_themes"))

    def test_financial_trend_tool_creates_findings(self) -> None:
        from storage.query_service import fetch_financial_metrics
        from agents.tools import financial_trend_tool

        metric_rows = fetch_financial_metrics("NVDA", metric_names=["revenue"], limit_per_metric=2)
        result = financial_trend_tool.invoke({"ticker": "NVDA", "metric_rows": metric_rows})
        self.assertTrue(result["findings"])
        self.assertIn("change", result["findings"][0]["summary"])

    def test_workflow_returns_findings_and_final_answer(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow.run_analyst_workflow(
            "Do NVIDIA's recent financials support the AI growth narrative, and what risks still stand out?",
            company_ticker="NVDA",
        )
        self.assertIn("orchestration_plan", result)
        self.assertIn("analysis_bundle", result)
        self.assertIn("final_answer", result)
        self.assertIn("routing_source", result)
        self.assertIn("selected_tools", result)
        self.assertIn("requested_sources", result)
        self.assertIn("selected_sources", result)
        self.assertTrue(result["analysis_bundle"]["findings"])
        self.assertTrue(result["final_answer"]["answer"])

    def test_quantitative_question_does_not_report_retrieved_qualitative_sources(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow.run_analyst_workflow(
            "How has NVIDIA revenue changed over recent reported periods?",
            company_ticker="NVDA",
        )
        self.assertEqual("quantitative", result["research_plan"]["question_type"])
        self.assertEqual([], result["selected_sources"])

    def test_mixed_growth_risk_question_does_not_request_market_data_by_default(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        plan = graph_workflow._heuristic_orchestration_plan(
            "Do NVIDIA's recent financials support the AI growth narrative, and what risks still stand out?",
            "NVDA",
        )
        self.assertFalse(plan.retrieval_plan.needs_market_data)
        self.assertNotIn("market_reaction_tool", plan.eda_plan.selected_tools)
        self.assertIn("financial_trend", plan.sub_intents)
        self.assertIn("risk_narrative", plan.sub_intents)

    def test_broad_question_triggers_clarification_suggestion(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow._planner_node({"question": "Tell me about NVIDIA", "company_ticker": "NVDA"})  # type: ignore[attr-defined]
        self.assertTrue(result["clarification_needed"])
        self.assertIn("financial performance", result["clarification_question"])

    def test_out_of_scope_question_triggers_guardrail(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow.run_analyst_workflow(
            "Write me a poem about tomorrow's weather",
            company_ticker="NVDA",
        )
        self.assertTrue(result["out_of_scope"])
        self.assertEqual([], result["selected_tools"])
        self.assertIn("outside the current app scope", result["final_answer"]["answer"])
        self.assertIn("analyst:out_of_scope", result["execution_log"])

    def test_llm_scope_check_can_mark_ambiguous_question_out_of_scope(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        with patch.object(graph_workflow, "_detect_out_of_scope", return_value=(False, None)):
            with patch.object(graph_workflow, "_should_use_llm_scope_check", return_value=True):
                with patch.object(
                    graph_workflow,
                    "_llm_scope_check",
                    return_value=(True, "This question is unrelated to the company-analysis dataset."),
                ):
                    result = graph_workflow._planner_node(
                        {"question": "Should I study P/E ratios or memorize formulas?", "company_ticker": "NVDA"}
                    )  # type: ignore[attr-defined]
        self.assertTrue(result["out_of_scope"])
        self.assertIn("planner:scope_check:llm_scope", result["execution_log"])

    def test_sql_breakdown_question_gets_direct_factual_answer(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow.run_analyst_workflow(
            "How many source records do we have by source type for NVIDIA?",
            company_ticker="NVDA",
        )
        self.assertIn("dataset currently contains", result["final_answer"]["answer"])
        self.assertIn("sec_filing", result["final_answer"]["answer"])
        self.assertNotIn("typical data coverage", result["final_answer"]["answer"])

    def test_negation_constraints_can_limit_sources(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow._planner_node(  # type: ignore[attr-defined]
            {"question": "Don't use filings, only use press releases to analyze NVIDIA risks.", "company_ticker": "NVDA"}
        )
        self.assertEqual(["press_release"], result["orchestration_plan"]["retrieval_plan"]["source_types"])
        self.assertEqual("qualitative", result["orchestration_plan"]["question_type"])
        self.assertNotIn("financial_trend_tool", result["selected_tools"])

    def test_negation_constraints_can_shift_metric_focus(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow._planner_node(  # type: ignore[attr-defined]
            {"question": "Not revenue; focus on cash for NVIDIA.", "company_ticker": "NVDA"}
        )
        metric_names = result["orchestration_plan"]["retrieval_plan"]["metric_names"]
        self.assertIn("cash_and_cash_equivalents", metric_names)
        self.assertNotIn("revenue", metric_names)

    def test_workflow_failure_returns_safe_fallback_result(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        with patch.object(graph_workflow.WORKFLOW, "invoke", side_effect=RuntimeError("forced workflow failure")):
            result = graph_workflow.run_analyst_workflow(
                "How has NVIDIA revenue changed over recent reported periods?",
                company_ticker="NVDA",
            )
        self.assertIn("backend step failed", result["final_answer"]["answer"])
        self.assertEqual([], result["selected_tools"])
        self.assertTrue(any(entry.startswith("workflow:error:") for entry in result["execution_log"]))

    def test_mixed_answer_includes_source_aware_narrative_point(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")
        result = graph_workflow.run_analyst_workflow(
            "What do recent filings and press releases suggest about growth versus risk for NVIDIA?",
            company_ticker="NVDA",
        )
        key_points = result["final_answer"]["key_points"]
        self.assertTrue(any("Filings" in point or "press releases" in point for point in key_points))
        self.assertTrue(any("supply chain" in point.lower() or "ai demand" in point.lower() for point in key_points))

    def test_resolve_company_can_use_sec_lookup_for_new_ticker(self) -> None:
        from pipelines import company_registry

        mock_response = Mock()
        mock_response.json.return_value = {
            "0": {"ticker": "MSFT", "cik_str": 789019, "title": "MICROSOFT CORP"}
        }
        mock_response.raise_for_status.return_value = None

        company_registry._sec_company_lookup.cache_clear()
        with patch("pipelines.company_registry.requests.get", return_value=mock_response):
            company = company_registry.resolve_company("MSFT")

        self.assertEqual("MSFT", company.ticker)
        self.assertEqual("MICROSOFT CORP", company.company_name)
        self.assertEqual("0000789019", company.cik)

    def test_ui_ticker_helpers_support_new_ticker_entry(self) -> None:
        from app.config import get_settings
        from app.paths import ensure_company_dirs
        from streamlit_app import _available_tickers, _resolve_sidebar_ticker

        ensure_company_dirs("MSFT")
        get_settings.cache_clear()

        tickers = _available_tickers("NVDA")
        self.assertIn("MSFT", tickers)
        self.assertEqual("MSFT", _resolve_sidebar_ticker("NVDA", "msft"))
        self.assertEqual("NVDA", _resolve_sidebar_ticker("NVDA", ""))


if __name__ == "__main__":
    unittest.main()
