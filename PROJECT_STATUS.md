# Project Status

## Project Overview

This project is a `Single-Company Deep Research Analyst Agent` focused on helping a user research one public company at a time, starting with `NVIDIA (NVDA)`.

The core idea is to combine:
- qualitative company context from SEC filings and press releases
- quantitative company facts from SEC CompanyFacts and market data
- a future agent workflow that uses both kinds of evidence together to generate grounded analyst-style answers

This is not intended to be a trading bot. It is better framed as an `AI company diligence copilot`.

## Project Goal

The target system should support questions such as:
- What are the biggest risks for NVIDIA?
- How has revenue changed over the last few quarters?
- Do the financials support the growth narrative?
- Generate a short analyst memo with evidence.

The architecture is intentionally split into:
- a qualitative retrieval path using vector search over chunked document text
- a quantitative retrieval path using SQL-backed financial and market facts
- an LLM-based planning and synthesis layer that combines both

## High-Level Architecture

The system is being built in layers:

1. Source layer
Raw documents and datasets are fetched and saved locally.

2. Structured storage layer
Structured records are stored in SQLite.

3. Qualitative processing layer
Raw SEC filings and press releases are cleaned, split into sections, chunked, and stored for later retrieval.

4. Future vector retrieval layer
Qualitative chunks will be embedded and indexed in FAISS.
Quantitative facts will continue to come from SQLite.

5. Future agent layer
A Planner -> Researcher -> Analyst workflow will retrieve the needed evidence and synthesize a final answer.

## Current Data Sources

### 1. SEC Filings
Forms currently fetched:
- 10-K
- 10-Q
- 8-K

Purpose:
- qualitative company context
- risk factors
- management discussion
- corporate disclosures

Current raw storage:
- `.html`
- `.txt`

### 2. SEC CompanyFacts
Purpose:
- structured financial facts such as revenue, net income, EPS, gross profit, R&D, and cash

Current raw storage:
- `.json`

### 3. Market Data
Source:
- Yahoo Finance via `yfinance`

Purpose:
- daily stock prices and volume

Current raw storage:
- `.csv`

### 4. Press Releases
Source:
- NVIDIA newsroom page

Purpose:
- official company announcements and company narrative

Current raw storage:
- `.txt`

## Current Storage Model

### Raw File Storage
Located under:
- `data/companies/nvda/raw/`

Contains:
- SEC filing HTML/TXT
- CompanyFacts JSON
- market CSV
- press release TXT

### SQLite Structured Storage
Located at:
- `data/app.db`

Current tables:
- `companies`
- `sources`
- `chunks`
- `financial_metrics`
- `market_data`
- `events`
- `refresh_runs`

### Future Vector Storage
Planned:
- FAISS index for chunk embeddings

This will store embeddings for qualitative chunks only.

## Current Runnable Stages

### 1. Refresh Company
Command:

```powershell
python -m pipelines.refresh_company --ticker NVDA
```

Purpose:
- fetch raw company data
- store source metadata
- parse quantitative metrics into SQLite

What it currently does:
- bootstraps storage if needed
- resolves `NVDA`
- fetches SEC filings
- fetches SEC CompanyFacts
- fetches market data
- fetches press release text
- writes source rows to SQLite
- writes financial metrics to SQLite
- writes market rows to SQLite

### 2. Process Documents
Command:

```powershell
python -m pipelines.process_documents --ticker NVDA
```

Purpose:
- process qualitative documents into retrieval-ready chunks

What it currently does:
- loads raw SEC and press release text
- cleans text
- splits SEC filings by plain-text `Item ...` section headers
- chunks within those sections
- writes chunk rows to SQLite
- writes a preview JSONL file under `processed/`

## Current Status

### What Is Working Well

#### Quantitative side
- `financial_metrics` is in good shape for MVP
- `market_data` is healthy
- the revenue extraction bug was fixed

Current metric families in SQLite:
- `revenue`
- `net_income`
- `gross_profit`
- `eps_diluted`
- `research_and_development`
- `cash_and_cash_equivalents`

These now include recent periods such as:
- `2026-FY`
- `2026-Q3`
- `2026-Q2`
- `2026-Q1`
- `2025-FY`
- `2025-Q3`

#### Qualitative side
- SEC and press release processing works
- chunk generation works
- section-aware SEC chunking is implemented
- SEC chunks are significantly better than the earliest raw output
- press release text extraction is usable

### What Is Still Weak or Incomplete
- only `1` press release source is currently collected
- raw SEC `.txt` files still look ugly as intermediate artifacts
- there is no FAISS vector index yet
- there are no embeddings yet
- the `events` table is not meaningfully populated yet
- MCP tools are not built yet
- the LangGraph workflow is not built yet
- the Streamlit app is not built yet

## Current SEC Processing State

### Earlier state
The SEC `.txt` output looked very bad:
- inline XBRL noise
- taxonomy labels
- cover-page clutter
- table-of-contents fragments

### Current state
SEC processing now:
- cleans the raw text
- identifies meaningful sections by text-based `Item ...` patterns
- keeps useful sections such as:
  - `Item 1. Business`
  - `Item 1A. Risk Factors`
  - `Item 7. MD&A`
  - `Item 7A. Market Risk`
  - `Item X.XX` sections for `8-K`
- chunks within those sections

Important distinction:
- the raw SEC files are still ugly
- the processed chunk output is much better

## Current Chunking Strategy

Current chunking strategy:
- first: section-aware splitting
- then: fixed-size chunking with overlap inside each section

For SEC filings:
- detect filing sections
- keep target sections
- split into chunks

For press releases:
- treat the article body as one section
- split into chunks

Current chunking parameters:
- `chunk_size = 1200`
- `overlap = 200`

Chunks are saved in SQLite `chunks` with metadata including:
- `source_id`
- `source_type`
- `chunk_order`
- `section_label`
- `section_title`
- `raw_path`

## Intuitive Pipeline Separation

### `refresh_company`
Collect the research materials.

### `process_documents`
Prepare the materials for retrieval.

Analogy:
- `refresh_company` = go to the library and bring back books and spreadsheets
- `process_documents` = highlight, section, and cut the books into useful note cards

This separation matters because it lets us improve processing many times without redownloading everything.

## Qualitative vs Quantitative Design

### Qualitative side
Sources:
- SEC filings
- press releases

Flow:
- raw files
- clean text
- chunking
- future embeddings
- future FAISS retrieval
- later pass retrieved chunks to the LLM

### Quantitative side
Sources:
- CompanyFacts
- market data

Flow:
- raw file
- parse into structured rows
- query via SQLite
- later pass returned numeric facts to the LLM

Important clarification:
- the LLM is not supposed to invent or directly write the numbers
- it should request the quantitative data from structured storage
- then reason over the returned facts

## Repo / Data Handling Intent

Current repo intent:
- keep code tracked
- ignore local raw artifacts
- ignore local DB
- likely ignore vector artifacts later

Intended ignored areas include:
- `.venv/`
- `.env`
- `data/app.db`
- `data/companies/*/raw/`
- `data/companies/*/vector/`

## Current Status Snapshot

The project currently has:
- working NVIDIA data ingestion
- working SQLite structured storage
- fixed CompanyFacts metric extraction
- working market data ingestion
- improved press release extraction
- working section-aware SEC chunking
- chunk storage in SQLite

The project does not yet have:
- embeddings
- FAISS
- event extraction
- MCP tools
- LangGraph agent flow
- Streamlit frontend

So the project is currently at:
- data engineering done to a usable MVP foundation
- retrieval and agent intelligence still to be built

## Recommended Immediate Next Steps

### 1. Build embeddings + FAISS
Goal:
- turn the existing chunked qualitative data into searchable vector context

Tasks:
- generate embeddings for chunks
- store embeddings in FAISS
- keep chunk text/metadata in SQLite
- implement mapping from FAISS result back to SQLite chunk row

### 2. Build qualitative retrieval helpers
Goal:
- answer qualitative questions with retrieved chunk evidence

Tasks:
- embed user query
- search FAISS
- fetch top chunk rows from SQLite
- return chunk text + metadata

### 3. Build quantitative query helpers
Goal:
- answer numeric questions from SQLite

Tasks:
- query `financial_metrics`
- query `market_data`
- later add derived metrics like revenue growth and margins

### 4. Build event extraction
Goal:
- create timeline-style structured company developments

Tasks:
- extract major events from filings and press releases
- store them in `events`
- use them for report generation later

### 5. Expose tools
Likely tools:
- `vector_context`
- `financial_metrics`
- `market_data`
- `timeline_events`
- `report`

### 6. Build Planner -> Researcher -> Analyst workflow
Goal:
- let the LLM plan which evidence to retrieve and synthesize it

Likely components:
- LangGraph
- MCP
- OpenAI API

### 7. Build Streamlit UI
Goal:
- select company
- refresh data
- ask questions
- show final answer and evidence

## Recommended Agent Skills

The most useful first skills for the future agent system are:

### 1. `vector_context_skill`
Purpose:
- retrieve qualitative evidence from SEC and press-release chunks

Input:
- company
- natural language question
- top-k

Output:
- top chunk texts
- section labels
- source URLs

### 2. `financial_metrics_skill`
Purpose:
- answer structured financial questions from SQLite

Input:
- company
- metric name
- optional period filter

Output:
- exact financial rows
- periods
- values

### 3. `market_data_skill`
Purpose:
- answer stock price and market trend questions

Input:
- company
- date range or event window

Output:
- market rows
- summary statistics later

### 4. `timeline_events_skill`
Purpose:
- return extracted major company developments once event extraction exists

Input:
- company
- date range
- event type filter

Output:
- event rows with source evidence

### 5. `report_skill`
Purpose:
- synthesize gathered evidence into a concise analyst-style answer or memo

Input:
- evidence bundle from retrieval tools

Output:
- final answer
- key points
- source citations

### 6. `resolve_company_skill`
Purpose:
- map user input to the target company and internal metadata

### 7. `refresh_company_data_skill`
Purpose:
- trigger a company data refresh for the selected ticker

## Recommended Build Order For Agent Skills

1. `financial_metrics_skill`
2. `market_data_skill`
3. `vector_context_skill`
4. `report_skill`
5. `timeline_events_skill`
6. `resolve_company_skill`
7. `refresh_company_data_skill`

Reason:
- first make retrieval work
- then make synthesis work
- then add orchestration around those capabilities
