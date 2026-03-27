from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from app.logging import get_logger
from app.paths import raw_data_dir
from schemas.models import Company, MarketRecord, Source


logger = get_logger(__name__)


def _market_data_dir(ticker: str) -> Path:
    path = raw_data_dir(ticker) / "market_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_market_history(company: Company, period: str = "5y") -> tuple[Source, list[MarketRecord]]:
    logger.info("Fetching market history for %s using yfinance", company.ticker)
    history = yf.Ticker(company.ticker).history(period=period, interval="1d")
    if history.empty:
        raise ValueError(f"No market history returned for {company.ticker}")

    history = history.reset_index()
    csv_path = _market_data_dir(company.ticker) / f"{company.ticker.lower()}_market_history.csv"
    history.to_csv(csv_path, index=False)

    source = Source(
        company_ticker=company.ticker,
        source_type="market_data",
        title=f"{company.ticker} market history ({period})",
        source_url=f"https://finance.yahoo.com/quote/{company.ticker}/history",
        raw_path=str(csv_path),
        metadata_json={"provider": "yfinance", "period": period},
    )

    records: list[MarketRecord] = []
    for _, row in history.iterrows():
        date_value = pd.to_datetime(row["Date"]).to_pydatetime()
        records.append(
            MarketRecord(
                company_ticker=company.ticker,
                date=date_value,
                open=float(row["Open"]) if pd.notna(row["Open"]) else None,
                high=float(row["High"]) if pd.notna(row["High"]) else None,
                low=float(row["Low"]) if pd.notna(row["Low"]) else None,
                close=float(row["Close"]) if pd.notna(row["Close"]) else None,
                volume=float(row["Volume"]) if pd.notna(row["Volume"]) else None,
                metadata_json={"provider": "yfinance"},
            )
        )

    return source, records
