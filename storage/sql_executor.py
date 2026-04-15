from __future__ import annotations

from time import perf_counter
from typing import Any

from storage.database import connection_scope


def execute_readonly_sql(sql: str) -> tuple[list[dict[str, Any]], float]:
    started = perf_counter()
    with connection_scope() as connection:
        rows = connection.execute(sql).fetchall()
    elapsed_ms = (perf_counter() - started) * 1000.0
    return [dict(row) for row in rows], elapsed_ms

