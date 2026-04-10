from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.paths import processed_data_dir
from graph.workflow import run_analyst_workflow
from schemas.models import AnalysisBundle, EvidenceBundle, FinalAnswer, OrchestrationPlan, ResearchPlan, RetryDecision
from storage.bootstrap import bootstrap_storage


DEFAULT_QUESTIONS = [
    "Do NVIDIA's recent financials support the AI growth narrative, and what risks still stand out?",
    "What risks does NVIDIA emphasize most in recent filings?",
    "How has NVIDIA revenue changed over recent reported periods?",
    "How did NVDA stock react around recent reporting periods?",
    "What do recent filings and press releases suggest about growth versus risk for NVIDIA?",
]


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(text: str, max_length: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:max_length].rstrip("-") or "question"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _build_markdown_summary(results: list[dict[str, Any]], ticker: str, run_dir: Path) -> str:
    lines = [
        "# Workflow Evaluation Summary",
        "",
        f"- Ticker: `{ticker}`",
        f"- Run directory: `{run_dir}`",
        f"- Question count: `{len(results)}`",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"## Q{result['index']}: {result['question']}",
                "",
                f"- Status: `{result['status']}`",
                f"- Question category: `{result.get('question_category', 'n/a')}`",
                f"- Question type: `{result.get('question_type', 'n/a')}`",
                f"- Sub-intents: `{', '.join(result.get('sub_intents', [])) or 'n/a'}`",
                f"- Planning source: `{result.get('planning_source', 'n/a')}`",
                f"- Clarification suggested: `{result.get('clarification_needed', False)}`",
                f"- Qualitative records: `{result.get('qualitative_count', 0)}`",
                f"- Quantitative records: `{result.get('quantitative_count', 0)}`",
                f"- Finding count: `{result.get('finding_count', 0)}`",
                f"- Source count: `{result.get('source_count', 0)}`",
            ]
        )

        if result.get("clarification_question"):
            lines.append(f"- Clarification question: `{result['clarification_question']}`")

        if result.get("selected_tools"):
            lines.append("- Selected tools:")
            lines.extend([f"  - {tool}" for tool in result["selected_tools"]])

        if result.get("requested_sources"):
            lines.append("- Requested sources:")
            lines.extend([f"  - {source}" for source in result["requested_sources"]])

        if result.get("selected_sources"):
            lines.append("- Retrieved sources:")
            lines.extend([f"  - {source}" for source in result["selected_sources"]])

        if result.get("retry_reason"):
            lines.append(f"- Retry reason: `{result['retry_reason']}`")

        if result.get("key_points"):
            lines.append("- Key points:")
            lines.extend([f"  - {point}" for point in result["key_points"][:3]])

        if result.get("artifact_json"):
            lines.append(f"- Full payload: `{result['artifact_json']}`")

        if result.get("error"):
            lines.append(f"- Error: `{result['error']}`")

        lines.append("")

    return "\n".join(lines)


def _extract_compact_summary(index: int, question: str, payload: dict[str, Any], artifact_json: str) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(payload["research_plan"])
    orchestration_plan = OrchestrationPlan.model_validate(payload["orchestration_plan"])
    evidence_bundle = EvidenceBundle.model_validate(payload["evidence_bundle"])
    analysis_bundle = AnalysisBundle.model_validate(payload["analysis_bundle"])
    retry_decision = RetryDecision.model_validate(payload.get("retry_decision") or {})
    final_answer = FinalAnswer.model_validate(payload["final_answer"])

    return {
        "index": index,
        "question": question,
        "status": "ok",
        "question_category": plan.question_category,
        "question_type": plan.question_type,
        "sub_intents": orchestration_plan.sub_intents,
        "planning_source": orchestration_plan.planning_source,
        "clarification_needed": bool(payload.get("clarification_needed")),
        "clarification_question": payload.get("clarification_question"),
        "qualitative_count": len(evidence_bundle.qualitative_records),
        "quantitative_count": len(evidence_bundle.quantitative_records),
        "finding_count": len(analysis_bundle.findings),
        "source_count": len(final_answer.sources),
        "selected_tools": payload.get("selected_tools", []),
        "requested_sources": payload.get("requested_sources", []),
        "selected_sources": payload.get("selected_sources", []),
        "retry_reason": retry_decision.reason,
        "key_points": final_answer.key_points,
        "artifact_json": artifact_json,
    }


def run_evaluation(ticker: str, questions: list[str], output_dir: Path | None = None) -> Path:
    bootstrap_storage()
    run_dir = output_dir or (processed_data_dir(ticker) / "evaluations" / _timestamp_slug())
    run_dir.mkdir(parents=True, exist_ok=True)

    compact_results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        stem = f"q{index:02d}_{_slugify(question)}"
        payload_path = run_dir / f"{stem}.json"

        try:
            payload = run_analyst_workflow(question, company_ticker=ticker)
            _json_dump(payload_path, payload)
            compact_results.append(_extract_compact_summary(index, question, payload, str(payload_path)))
            print(f"[ok] Q{index}: {question}")
        except Exception as exc:  # pragma: no cover - runtime utility
            error_payload = {
                "question_index": index,
                "question": question,
                "status": "error",
                "error": str(exc),
            }
            _json_dump(payload_path, error_payload)
            compact_results.append(
                {
                    "index": index,
                    "question": question,
                    "status": "error",
                    "error": str(exc),
                    "artifact_json": str(payload_path),
                }
            )
            print(f"[error] Q{index}: {question} -> {exc}")

    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    _json_dump(summary_json, {"ticker": ticker, "results": compact_results})
    summary_md.write_text(_build_markdown_summary(compact_results, ticker, run_dir), encoding="utf-8")

    print(f"\nSaved evaluation run to: {run_dir}")
    print(f"Summary markdown: {summary_md}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight evaluation harness over the current analyst workflow."
    )
    parser.add_argument("--ticker", default="NVDA", help="Ticker to evaluate. Default: NVDA")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Add a benchmark question. Repeat the flag to add multiple questions. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: data/companies/<ticker>/processed/evaluations/<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = args.questions or DEFAULT_QUESTIONS
    run_evaluation(ticker=args.ticker, questions=questions, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
