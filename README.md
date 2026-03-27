# Company Diligence Agent

Single-company deep research analyst agent focused on investment-style diligence.

## MVP

- Select one company at a time, starting with NVIDIA (`NVDA`)
- Refresh filings, company facts, press releases, and market data
- Store qualitative chunks in a vector index
- Store quantitative metrics in SQLite
- Answer questions through a Planner -> Researcher -> Analyst workflow

## Initial Setup

1. Create a virtual environment.
2. Install the project:
   ```bash
   pip install -e .
   ```
3. Copy `.env.example` to `.env` and fill in your API key.

## Current Status

Steps 1-2 are scaffolded:
- project structure
- configuration loader
- logging setup
- local storage path helpers

