from __future__ import annotations

import json
import re
from pathlib import Path

from app.logging import get_logger
from app.paths import processed_data_dir
from schemas.models import Chunk
from storage.repositories import delete_chunks_for_company, fetch_sources, insert_chunk


logger = get_logger(__name__)

# Short, low-information lines from SEC inline XBRL are common and add noise to retrieval.
_NOISY_LINE_PATTERNS = [
    re.compile(r"^\d{4,}$"),
    re.compile(r"^[A-Za-z0-9:_\-#.]+$"),
    re.compile(r"^(xbrli|dei|us-gaap|iso4217|nvda):", re.IGNORECASE),
]

# 10-K / 10-Q item labels we care about most for analyst-style qualitative retrieval.
_TARGET_LONG_FORM_SECTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("item_1_business", re.compile(r"^item\s+1\.?\s+business\b", re.IGNORECASE)),
    ("item_1a_risk_factors", re.compile(r"^item\s+1a\.?\s+risk factors\b", re.IGNORECASE)),
    (
        "item_7_mda",
        re.compile(
            r"^item\s+7\.?\s+management[’'`s\s]+discussion and analysis of financial condition and results of operations\b",
            re.IGNORECASE,
        ),
    ),
    (
        "item_7a_market_risk",
        re.compile(r"^item\s+7a\.?\s+quantitative and qualitative disclosures about market risk\b", re.IGNORECASE),
    ),
]

# General item boundaries help us stop sections at the next filing item, even when we do not keep that next item.
_LONG_FORM_BOUNDARY_PATTERN = re.compile(r"^item\s+(1a|1b|1c|2|3|4|5|6|7|7a|8|9|9a|9b|9c|10|11|12|13|14|15|16)\.?\b", re.IGNORECASE)

# 8-K filings use more specific item numbering such as 2.02 or 5.02.
_EIGHT_K_SECTION_PATTERN = re.compile(r"^item\s+(\d+\.\d+)\.?\s+", re.IGNORECASE)


def process_company_documents(ticker: str, source_types: list[str] | None = None, chunk_size: int = 1200, overlap: int = 200) -> dict[str, int]:
    # Convert stored raw source text into cleaned chunks that can later be embedded.
    # For SEC filings, we first split the text into meaningful filing sections, then chunk within each section.
    source_types = source_types or ["sec_filing", "press_release"]
    processed_dir = processed_data_dir(ticker)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous chunk rows for these sources so reruns replace stale output.
    delete_chunks_for_company(ticker, source_types=source_types)

    chunk_count = 0
    source_count = 0
    chunk_records: list[dict] = []

    for source in fetch_sources(ticker, source_types=source_types):
        raw_path = source.get("raw_path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue

        source_count += 1
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        source_type = source.get("source_type", "")

        if source_type == "sec_filing":
            sections = split_sec_into_sections(raw_text, title=source.get("title", ""))
        else:
            clean_text = clean_document_text(raw_text, source_type)
            sections = [{"section_label": "full_text", "section_title": source.get("title") or "full_text", "text": clean_text}]

        source_chunk_order = 0
        for section in sections:
            section_text = section.get("text", "").strip()
            if not section_text:
                continue
            chunks = split_text_into_chunks(section_text, chunk_size=chunk_size, overlap=overlap)
            for chunk_text in chunks:
                chunk = Chunk(
                    company_ticker=ticker,
                    source_id=source["id"],
                    source_type=source_type,
                    title=source.get("title"),
                    source_url=source.get("source_url"),
                    chunk_text=chunk_text,
                    chunk_order=source_chunk_order,
                    metadata_json={
                        "raw_path": raw_path,
                        "section_label": section.get("section_label"),
                        "section_title": section.get("section_title"),
                    },
                )
                insert_chunk(chunk)
                chunk_records.append(
                    {
                        "source_id": source["id"],
                        "source_type": source_type,
                        "chunk_order": source_chunk_order,
                        "section_label": section.get("section_label"),
                        "section_title": section.get("section_title"),
                        "chunk_text": chunk_text,
                    }
                )
                chunk_count += 1
                source_chunk_order += 1

    output_path = processed_dir / f"{ticker.lower()}_chunks_preview.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in chunk_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    logger.info("Processed %s sources into %s chunks for %s", source_count, chunk_count, ticker)
    return {"sources_processed": source_count, "chunks_created": chunk_count}


def clean_document_text(text: str, source_type: str) -> str:
    # Use slightly different cleanup rules depending on source type.
    if source_type == "sec_filing":
        text = _clean_sec_text(text)
    else:
        text = _clean_generic_text(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sec_into_sections(text: str, title: str = "") -> list[dict[str, str]]:
    # SEC plain-text filings usually still retain "Item ..." lines, even after formatting is lost.
    # We use those lines to keep only the high-value sections and skip cover-page noise.
    clean_text = _clean_sec_text(text)
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

    if "8-K" in title.upper():
        return _split_eight_k_sections(lines)
    return _split_long_form_sections(lines)


def _split_long_form_sections(lines: list[str]) -> list[dict[str, str]]:
    boundary_starts: list[tuple[int, str]] = []
    target_starts: list[tuple[int, str, str]] = []

    for index, line in enumerate(lines):
        boundary_match = _LONG_FORM_BOUNDARY_PATTERN.match(line)
        if boundary_match:
            boundary_label = boundary_match.group(1).lower()
            boundary_starts.append((index, f"item_{boundary_label}"))
        for label, pattern in _TARGET_LONG_FORM_SECTIONS:
            if pattern.match(line):
                target_starts.append((index, label, line))
                break

    deduped_targets = _dedupe_section_starts(target_starts)
    sections: list[dict[str, str]] = []
    for start_index, label, line in deduped_targets:
        end_index = _next_boundary_index(boundary_starts, start_index, len(lines))
        body = "\n".join(lines[start_index:end_index]).strip()
        if len(body) < 500:
            continue
        sections.append({"section_label": label, "section_title": line, "text": body})

    if not sections:
        fallback = "\n".join(lines[80:]) if len(lines) > 80 else "\n".join(lines)
        if fallback.strip():
            sections.append({"section_label": "fallback_body", "section_title": "fallback_body", "text": fallback})
    return sections


def _split_eight_k_sections(lines: list[str]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    starts: list[tuple[int, str, str]] = []

    for index, line in enumerate(lines):
        match = _EIGHT_K_SECTION_PATTERN.match(line)
        if match:
            item_label = match.group(1)
            starts.append((index, f"item_{item_label}", line))

    for position, (start_index, label, line) in enumerate(starts):
        end_index = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start_index:end_index]).strip()
        if len(body) < 200:
            continue
        sections.append({"section_label": label, "section_title": line, "text": body})

    if not sections:
        fallback = "\n".join(lines[30:]) if len(lines) > 30 else "\n".join(lines)
        if fallback.strip():
            sections.append({"section_label": "fallback_body", "section_title": "fallback_body", "text": fallback})
    return sections


def _dedupe_section_starts(starts: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    # Keep the last occurrence of each section label because the first one is often just the table of contents.
    latest_by_label: dict[str, tuple[int, str, str]] = {}
    for start in starts:
        latest_by_label[start[1]] = start
    return sorted(latest_by_label.values(), key=lambda item: item[0])


def _next_boundary_index(boundaries: list[tuple[int, str]], start_index: int, default_end: int) -> int:
    for boundary_index, _ in boundaries:
        if boundary_index > start_index:
            return boundary_index
    return default_end


def _clean_sec_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _NOISY_LINE_PATTERNS):
            continue
        if len(stripped) < 3:
            continue
        if sum(ch.isalpha() for ch in stripped) < 3:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _clean_generic_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    # Simple character-based chunking is enough for MVP and easy to inspect.
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, 0)
    return chunks
