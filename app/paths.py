from __future__ import annotations

from pathlib import Path

from app.config import get_settings


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def company_slug(ticker: str) -> str:
    return normalize_ticker(ticker).lower()


def company_root(ticker: str) -> Path:
    return get_settings().data_dir / "companies" / company_slug(ticker)


def raw_data_dir(ticker: str) -> Path:
    return company_root(ticker) / "raw"


def processed_data_dir(ticker: str) -> Path:
    return company_root(ticker) / "processed"


def vector_data_dir(ticker: str) -> Path:
    return company_root(ticker) / "vector"


def company_paths(ticker: str) -> dict[str, Path]:
    return {
        "root": company_root(ticker),
        "raw": raw_data_dir(ticker),
        "processed": processed_data_dir(ticker),
        "vector": vector_data_dir(ticker),
    }


def ensure_company_dirs(ticker: str) -> dict[str, Path]:
    paths = company_paths(ticker)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

