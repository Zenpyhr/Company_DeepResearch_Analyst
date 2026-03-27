from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings
from app.logging import get_logger


logger = get_logger(__name__)


def get_database_path() -> Path:
    # Central place to read the configured SQLite file location.
    return get_settings().sqlite_db_path


def get_connection() -> sqlite3.Connection:
    # Row factory gives us dict-like access when we read records back later.
    db_path = get_database_path()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


@contextmanager
def connection_scope() -> Iterator[sqlite3.Connection]:
    # Wrap every DB operation in a safe transaction.
    # On success we commit; on failure we roll back automatically.
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    # This creates the structured SQL layer of the system.
    # It is separate from FAISS, which will later hold embeddings for semantic retrieval.
    statements = [
        """
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            company_name TEXT NOT NULL,
            cik TEXT,
            industry TEXT,
            website TEXT,
            last_refreshed_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            source_url TEXT NOT NULL,
            published_at TEXT,
            raw_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            source_id INTEGER,
            source_type TEXT,
            title TEXT,
            source_url TEXT,
            published_at TEXT,
            chunk_text TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            embedding_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES sources(id),
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            fiscal_period TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            unit TEXT,
            source_url TEXT,
            as_of_date TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker),
            UNIQUE(company_ticker, date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            timestamp TEXT,
            source_type TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_subtype TEXT,
            description TEXT NOT NULL,
            evidence_text TEXT,
            source_url TEXT,
            confidence REAL,
            verification_status TEXT NOT NULL DEFAULT 'unverified',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_ticker TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_ticker) REFERENCES companies(ticker)
        );
        """,
        # Indexes keep the most common lookups fast during refresh and question answering.
        "CREATE INDEX IF NOT EXISTS idx_sources_company_type ON sources(company_ticker, source_type);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks(company_ticker);",
        "CREATE INDEX IF NOT EXISTS idx_financial_metrics_company_metric ON financial_metrics(company_ticker, metric_name, fiscal_period);",
        "CREATE INDEX IF NOT EXISTS idx_market_data_company_date ON market_data(company_ticker, date);",
        "CREATE INDEX IF NOT EXISTS idx_events_company_timestamp ON events(company_ticker, timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_refresh_runs_company ON refresh_runs(company_ticker, started_at);",
    ]

    with connection_scope() as connection:
        cursor = connection.cursor()
        for statement in statements:
            cursor.execute(statement)

    logger.info("SQLite database initialized at %s", get_database_path().resolve())
