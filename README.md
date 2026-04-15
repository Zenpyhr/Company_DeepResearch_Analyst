# Project 2: Company Data Analyst Agent

Company analyst agent with an explicit `Collect -> Explore -> Hypothesize` workflow, a Streamlit frontend, a LangGraph backend, deterministic analysis tools, and a new safe Text2SQL path for structured metadata questions.

## Live App

- Public URL:
  - `https://company-deepresearch-analyst-git-555207000332.us-central1.run.app`

## Agent Capabilities

- Answers company-analysis questions about:
  - financial trends
  - market reaction
  - filing / press-release risk and growth themes
  - source-coverage and metadata questions through Text2SQL
- Retrieves real company data from:
  - SEC filings
  - SEC CompanyFacts
  - Yahoo Finance market history
  - NVIDIA press releases
- Uses a LangGraph workflow to:
  - plan the analysis path
  - collect evidence
  - run tool-based EDA
  - produce a grounded final answer with sources
- Applies deterministic constraint handling for explicit user instructions such as:
  - `only use press releases`
  - `don't use filings`
  - `not revenue, focus on cash`
- Includes an out-of-scope guardrail for unrelated requests such as weather, recipes, creative writing, sports, and general coding tasks
- Uses a hybrid scope guardrail:
  - fast deterministic rejection for obvious unrelated prompts
  - LLM scope classification for ambiguous off-topic prompts
- Returns a safe fallback response instead of crashing if a backend workflow step fails unexpectedly
- Generates artifacts such as:
  - CSVs
  - chart specs
  - markdown memos
  - SQL result artifacts

## System Structure

- `Frontend`
  - [streamlit_app.py](./streamlit_app.py)
- `Backend workflow / orchestrator`
  - [graph/workflow.py](./graph/workflow.py)
- `Tools`
  - [agents/tools.py](./agents/tools.py)
  - [agents/sql_tools.py](./agents/sql_tools.py)
- `Storage + SQL helpers`
  - [storage/query_service.py](./storage/query_service.py)
  - [storage/schema_catalog.py](./storage/schema_catalog.py)
  - [storage/sql_validator.py](./storage/sql_validator.py)
  - [storage/sql_executor.py](./storage/sql_executor.py)
- `Collection pipelines`
  - [pipelines/](./pipelines)

The frontend calls `run_analyst_workflow()`, the workflow decides which tools to use, and those tools call lower-level retrieval, SQL, and artifact helpers.

## Rubric Mapping

### Step 1: Collect

The app collects real, non-trivial company data and retrieves question-relevant evidence before the final answer is written.

- `Runtime external data collection`
  - [pipelines/refresh_company.py](./pipelines/refresh_company.py) `refresh_company_data`
  - [agents/tools.py](./agents/tools.py) `refresh_company_data_tool`
- `SEC filings`
  - [pipelines/sec_client.py](./pipelines/sec_client.py) `fetch_recent_filings`
- `SEC CompanyFacts`
  - [pipelines/companyfacts_client.py](./pipelines/companyfacts_client.py) `fetch_companyfacts`
- `Yahoo Finance market data`
  - [pipelines/market_data.py](./pipelines/market_data.py) `fetch_market_history`
- `NVIDIA press releases`
  - [pipelines/press_releases.py](./pipelines/press_releases.py) `fetch_press_releases`
- `Chunking for qualitative retrieval`
  - [pipelines/text_processing.py](./pipelines/text_processing.py) `process_company_documents`
- `Question-driven retrieval from local structured storage`
  - [agents/tools.py](./agents/tools.py) `retrieve_document_context_tool`
  - [agents/tools.py](./agents/tools.py) `retrieve_financial_metrics_tool`
  - [agents/tools.py](./agents/tools.py) `retrieve_market_data_tool`
  - [storage/query_service.py](./storage/query_service.py) `search_document_chunks`
  - [storage/query_service.py](./storage/query_service.py) `fetch_financial_metrics`
  - [storage/query_service.py](./storage/query_service.py) `fetch_market_data`

How collection is dynamic:

- The planner classifies the question and chooses the evidence path in [graph/workflow.py](./graph/workflow.py) `_planner_node`, `_heuristic_orchestration_plan`, and `_research_plan_from_orchestration`.
- Different questions trigger different retrieval combinations:
  - financial questions -> financial metrics
  - market questions -> market data windows
  - filing / press-release questions -> chunk retrieval
  - metadata / breakdown questions -> SQL path
- If no evidence is available locally, the collector attempts a refresh once inside [graph/workflow.py](./graph/workflow.py) `_collector_node`.

Notes:

- The strongest supported end-to-end path is still `NVDA`.
- Refresh is available from the UI and also as an empty-evidence fallback.
- The current system does not yet auto-refresh based on staleness; it refreshes on explicit user request or when the local evidence bundle is empty.

### Step 2: Explore

The app performs EDA with tool calls over collected data before writing the final answer.

- `EDA workflow node`
  - [graph/workflow.py](./graph/workflow.py) `_eda_node`
- `Financial EDA`
  - [agents/tools.py](./agents/tools.py) `financial_trend_tool`
- `Market EDA`
  - [agents/tools.py](./agents/tools.py) `market_reaction_tool`
- `Qualitative text-theme EDA`
  - [agents/tools.py](./agents/tools.py) `text_theme_tool`
- `Visualization`
  - [agents/tools.py](./agents/tools.py) `chart_tool`
- `Text2SQL EDA for structured metadata / grouped questions`
  - [agents/sql_tools.py](./agents/sql_tools.py) `schema_context_tool`
  - [agents/sql_tools.py](./agents/sql_tools.py) `sql_query_tool`
  - [storage/schema_catalog.py](./storage/schema_catalog.py) `build_schema_context`
  - [storage/sql_validator.py](./storage/sql_validator.py) `validate_select_sql`
  - [storage/sql_executor.py](./storage/sql_executor.py) `execute_readonly_sql`

Examples of EDA behavior:

- `financial_trend_tool` computes period-over-period changes and returns concrete findings.
- `market_reaction_tool` computes total return and volatility over the retrieved market window.
- `text_theme_tool` counts repeated themes and source-specific patterns in filings and press releases.
- `sql_query_tool` generates safe read-only SQL for count / breakdown / distribution / average style questions and returns rows as a finding.
- For structured source-count / metadata questions, the workflow can narrow the run to a SQL-only path in [graph/workflow.py](./graph/workflow.py) `_apply_sql_only_focus`.

The Explore tab in [streamlit_app.py](./streamlit_app.py) shows the EDA findings first and keeps raw payloads behind expanders so the analysis remains auditable without overwhelming the UI.

### Step 3: Hypothesize

The final answer is built from collected evidence and EDA findings, not directly from the question alone.

- `Analyst workflow node`
  - [graph/workflow.py](./graph/workflow.py) `_analyst_node`
- `Final synthesis`
  - [agents/tools.py](./agents/tools.py) `final_answer_builder`

How the hypothesis is grounded:

- The analyst node passes the evidence bundle and analysis bundle into `final_answer_builder`.
- `final_answer_builder` ranks findings, selects key points, gathers source URLs, and writes the final memo artifact.
- For narrow SQL-style questions such as counts, breakdowns, and averages, `final_answer_builder` switches to a direct factual answer mode instead of a longer memo-style response.
- The frontend displays:
  - answer
  - key points
  - sources
  - confidence note

## Core Requirements

### Frontend

- Implemented in [streamlit_app.py](./streamlit_app.py)
- Streamlit UI exposes:
  - ticker selection
  - manual refresh
  - question input
  - explicit `Collect`, `Explore / Analyze`, `Hypothesis`, and `Debug` tabs

### Agent Framework

- Implemented with LangGraph in [graph/workflow.py](./graph/workflow.py)
- Nodes:
  - planner
  - collector
  - eda
  - analyst

### Tool Calling

- Tools live in:
  - [agents/tools.py](./agents/tools.py)
  - [agents/sql_tools.py](./agents/sql_tools.py)
- Workflow nodes invoke tools directly in [graph/workflow.py](./graph/workflow.py)

### Non-Trivial Dataset

- External sources:
  - SEC filings
  - SEC CompanyFacts
  - Yahoo Finance market history
  - NVIDIA press releases
- Local structured storage:
  - SQLite tables for `sources`, `chunks`, `financial_metrics`, `market_data`
- These datasets are large enough that they are not meant to be fully dumped into prompt context.

### Multi-Agent Pattern

- Orchestrated specialist-agent pattern in [graph/workflow.py](./graph/workflow.py)
- Responsibilities are separated across:
  - `Orchestrator / Planner`
  - `Collector`
  - `EDA Agent`
  - `Analyst Agent`
- Distinct prompts live in [prompts/agent_prompts.py](./prompts/agent_prompts.py)
- SQL generation also uses specialist prompts in [prompts/sql_prompts.py](./prompts/sql_prompts.py)

### Deployed

- Public Cloud Run app:
  - `https://company-deepresearch-analyst-git-555207000332.us-central1.run.app`
- Deployment files:
  - [Dockerfile](./Dockerfile)
  - [cloudbuild.yaml](./cloudbuild.yaml)
  - [storage/cloud.py](./storage/cloud.py)

### README

- This file documents:
  - the three-step workflow
  - the core requirements
  - the grab-bag features
  - the current scope and fallback behavior

## Grab Bag Implemented

The project implements more than two elective concepts.

### Second Data Retrieval Method

- `Qualitative retrieval`
  - [storage/query_service.py](./storage/query_service.py) `search_document_chunks`
- `Quantitative retrieval`
  - [storage/query_service.py](./storage/query_service.py) `fetch_financial_metrics`
  - [storage/query_service.py](./storage/query_service.py) `fetch_market_data`
- `Structured SQL retrieval / analysis`
  - [agents/sql_tools.py](./agents/sql_tools.py) `sql_query_tool`

### Code Execution

- Pandas-based deterministic analysis in:
  - [agents/tools.py](./agents/tools.py) `financial_trend_tool`
  - [agents/tools.py](./agents/tools.py) `market_reaction_tool`
  - [agents/tools.py](./agents/tools.py) `text_theme_tool`

### Artifacts

- Artifact writing in [app/artifacts.py](./app/artifacts.py)
- Outputs include:
  - financial trend CSVs
  - market reaction CSVs
  - text-theme CSVs
  - chart spec JSON
  - SQL result JSON
  - final analyst memo markdown

### Structured Output

- Typed schemas in [schemas/models.py](./schemas/models.py)
- Structured workflow bundles:
  - `ResearchPlan`
  - `OrchestrationPlan`
  - `EvidenceBundle`
  - `AnalysisBundle`
  - `FinalAnswer`
  - `SchemaContext`
  - `SQLQueryResult`

### Data Visualization

- Vega-Lite chart generation in [agents/tools.py](./agents/tools.py) `chart_tool`
- Chart rendering in [streamlit_app.py](./streamlit_app.py)

### Iterative Refinement Loop

- Retry path in [graph/workflow.py](./graph/workflow.py)
- If the EDA phase decides evidence is incomplete, the workflow can route back from EDA to Collector once before final synthesis.

## Scope and Fallback Behavior

### Best-Supported Questions

- Financial trend questions
- Market reaction questions
- Filing / press-release growth and risk questions
- Source-count / metadata / breakdown questions through Text2SQL

### Current Fallbacks

- `Planner fallback`
  - If LLM planning fails, the workflow uses heuristic planning in [graph/workflow.py](./graph/workflow.py)
- `Constraint fallback`
  - Explicit negation / exclusion instructions are enforced deterministically in [graph/workflow.py](./graph/workflow.py) `_apply_negation_constraints`, even when the planner falls back to heuristics
- `LLM answer fallback`
  - If the LLM is unavailable, the app falls back to deterministic planning and answer construction through [agents/llm.py](./agents/llm.py) and [agents/tools.py](./agents/tools.py) `final_answer_builder`
- `Refresh fallback`
  - If retrieval returns no evidence, the collector tries `refresh_company_data_tool` once
- `SQL fallback`
  - If LLM SQL generation is unavailable, `sql_query_tool` falls back to a heuristic SQL path
- `Safe workflow fallback`
  - If an unexpected backend exception escapes the workflow, [graph/workflow.py](./graph/workflow.py) `run_analyst_workflow` returns a structured fallback result instead of crashing the app

### Out-of-Scope Handling

The current system **does** have an out-of-scope guardrail.

What it does today:

- The planner checks for clearly unrelated requests in [graph/workflow.py](./graph/workflow.py) `_detect_out_of_scope`.
- If the keyword layer is inconclusive, the planner can run an LLM scope check in [graph/workflow.py](./graph/workflow.py) `_llm_scope_check` before any collection starts.
- If the question is outside the company-analysis domain, the workflow short-circuits before collection and EDA.
- The final answer explains the supported scope and suggests better in-domain question types.
- For broad but still in-domain questions, the planner can surface a clarification suggestion in [graph/workflow.py](./graph/workflow.py) `_detect_clarification_need`.

What it does **not** yet do:

- automatic staleness-based refresh
- fine-grained partial support classification for borderline requests

## Run Locally

1. Create and activate a virtual environment.
2. Install the project:

   ```bash
   pip install -e .
   ```

3. Create a `.env` file with at least:

   ```bash
   OPENAI_API_KEY=...
   OPENAI_MODEL=gpt-4.1-mini
   DEFAULT_TICKER=NVDA
   ```

4. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Tests

- Main workflow tests:
  - [tests/test_workflow.py](./tests/test_workflow.py)
- Text2SQL tests:
  - [tests/test_sql_tools.py](./tests/test_sql_tools.py)

Example:

```bash
.venv\Scripts\python.exe -m unittest tests.test_workflow tests.test_sql_tools
```
