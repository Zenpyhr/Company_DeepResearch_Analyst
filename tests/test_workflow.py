from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


class WorkflowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(cls.temp_dir.name) / "data"
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
        cls.temp_dir.cleanup()

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
        self.assertIn("analysis_bundle", result)
        self.assertIn("final_answer", result)
        self.assertTrue(result["analysis_bundle"]["findings"])
        self.assertTrue(result["final_answer"]["answer"])


if __name__ == "__main__":
    unittest.main()
