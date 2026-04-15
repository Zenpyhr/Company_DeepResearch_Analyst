from __future__ import annotations

import re


FORBIDDEN_TOKENS = {"insert", "update", "delete", "drop", "alter", "pragma", "attach", "detach", "replace", "create"}


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def extract_table_names(sql: str) -> list[str]:
    lowered = _normalize_sql(sql).lower()
    table_names: list[str] = []
    for pattern in [r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)"]:
        table_names.extend(re.findall(pattern, lowered))
    return list(dict.fromkeys(table_names))


def validate_select_sql(sql: str, *, allowed_tables: list[str], max_limit: int = 200) -> tuple[bool, str | None]:
    normalized = _normalize_sql(sql)
    lowered = normalized.lower()

    if not lowered.startswith("select "):
        return False, "Only SELECT queries are allowed."
    if ";" in normalized.rstrip(";"):
        return False, "Only a single SQL statement is allowed."
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        return False, "The SQL contains a forbidden statement."

    referenced_tables = extract_table_names(normalized)
    if not referenced_tables:
        return False, "The SQL must reference at least one allowed table."
    if any(table not in allowed_tables for table in referenced_tables):
        return False, "The SQL references a table outside the allowed set."

    limit_match = re.search(r"\blimit\s+(\d+)\b", lowered)
    if not limit_match:
        return False, "The SQL must include a LIMIT clause."
    if int(limit_match.group(1)) > max_limit:
        return False, f"LIMIT must be less than or equal to {max_limit}."

    return True, None

