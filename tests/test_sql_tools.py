from __future__ import annotations

import importlib
import os
import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SQLToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workspace_tmp = Path.cwd() / "data" / "test_tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        cls.temp_dir = workspace_tmp / f"sql_tools_{uuid4().hex}"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        data_dir = cls.temp_dir / "data"
        os.environ["DATA_DIR"] = str(data_dir)
        os.environ["SQLITE_DB_PATH"] = str(data_dir / "app.db")
        os.environ["DEFAULT_TICKER"] = "NVDA"
        os.environ["OPENAI_API_KEY"] = ""

        from app.config import get_settings

        get_settings.cache_clear()

        from schemas.models import Chunk, FinancialMetric, MarketRecord, Source
        from storage.bootstrap import bootstrap_storage
        from storage.repositories import insert_chunk, insert_financial_metric, insert_source, upsert_market_record

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
        insert_source(
            Source(
                company_ticker="NVDA",
                source_type="companyfacts",
                title="NVDA companyfacts payload",
                source_url="https://example.com/companyfacts",
                raw_path="data/example_companyfacts.json",
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
                chunk_text="Competition and supply chain risk remain important concerns.",
                chunk_order=0,
                metadata_json={"section_label": "item_1a_risk_factors"},
            )
        )
        insert_chunk(
            Chunk(
                company_ticker="NVDA",
                source_id=press_source_id,
                source_type="press_release",
                title="NVIDIA expands AI infrastructure",
                source_url="https://example.com/press",
                chunk_text="AI infrastructure demand stayed strong across data center customers.",
                chunk_order=0,
                metadata_json={"section_label": "full_text"},
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

        for day, close_value in enumerate([100.0, 103.0, 101.0], start=1):
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

    def test_schema_context_tool_returns_relevant_tables(self) -> None:
        from agents.sql_tools import schema_context_tool

        result = schema_context_tool.invoke(
            {
                "ticker": "NVDA",
                "question": "How many source records do we have by source type for NVIDIA?",
            }
        )
        self.assertIn("sources", result["relevant_tables"])
        self.assertTrue(result["tables"])

    def test_sql_validator_blocks_write_queries(self) -> None:
        from storage.sql_validator import validate_select_sql

        is_valid, error_message = validate_select_sql(
            "DELETE FROM sources WHERE company_ticker = 'NVDA'",
            allowed_tables=["sources"],
        )
        self.assertFalse(is_valid)
        self.assertIn("SELECT", error_message or "")

    def test_sql_query_tool_can_count_sources(self) -> None:
        from agents.sql_tools import sql_query_tool

        result = sql_query_tool.invoke(
            {
                "ticker": "NVDA",
                "question": "How many source records do we have by source type for NVIDIA?",
                "allowed_tables": ["sources", "financial_metrics"],
                "max_rows": 10,
            }
        )
        self.assertEqual("ok", result["sql_result"]["status"])
        self.assertGreaterEqual(result["sql_result"]["row_count"], 2)
        self.assertIn("source_type", result["sql_result"]["rows"][0])

    def test_workflow_selects_sql_tool_for_breakdown_questions(self) -> None:
        graph_workflow = importlib.import_module("graph.workflow")

        plan_result = graph_workflow._planner_node(  # type: ignore[attr-defined]
            {"question": "Give me a breakdown of NVIDIA source records by source type", "company_ticker": "NVDA"}
        )
        self.assertIn("sql_query_tool", plan_result["selected_tools"])


if __name__ == "__main__":
    unittest.main()

