"""Typed DuckDB helper utilities for parameterized queries and logging."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from kgfoundry_common.errors import RegistryError
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.observability import MetricsProvider, observe_duration

Params = Sequence[object] | Mapping[str, object] | None

DEFAULT_TIMEOUT_S: Final[float] = 5.0
DEFAULT_SLOW_QUERY_THRESHOLD_S: Final[float] = 0.5
DEFAULT_THREADS: Final[int] = 14
MAX_SQL_PREVIEW_CHARS: Final[int] = 160

__all__ = [
    "DEFAULT_SLOW_QUERY_THRESHOLD_S",
    "DEFAULT_TIMEOUT_S",
    "connect",
    "execute",
    "fetch_all",
    "fetch_one",
    "validate_identifier",
]

logger = get_logger(__name__)
metrics = MetricsProvider.default()


def _format_sql(sql: str) -> str:
    compact = " ".join(sql.split())
    if len(compact) <= MAX_SQL_PREVIEW_CHARS:
        return compact
    return f"{compact[:MAX_SQL_PREVIEW_CHARS]}…"


def _truncate_value(value: object) -> object:
    if isinstance(value, str) and len(value) > MAX_SQL_PREVIEW_CHARS:
        return value[:MAX_SQL_PREVIEW_CHARS] + "…"
    return value


def _format_params(params: Params) -> object:
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return {str(key): _truncate_value(value) for key, value in params.items()}
    return [_truncate_value(value) for value in params]


def _ensure_parameterized(sql: str, *, require_parameterized: bool) -> None:
    if not require_parameterized:
        return
    if "?" in sql or ":" in sql or "$" in sql:
        return
    error_message = "DuckDB query must be parameterized"
    raise RegistryError(
        error_message,
        context={"sql_preview": _format_sql(sql)},
    )


def _set_timeout(conn: DuckDBPyConnection, timeout_s: float) -> None:
    if timeout_s <= 0:
        return
    timeout_ms = int(timeout_s * 1000)
    conn.execute(f"PRAGMA statement_timeout='{timeout_ms}ms'")


def connect(
    db_path: str | Path,
    *,
    read_only: bool = False,
    pragmas: Mapping[str, object] | None = None,
) -> DuckDBPyConnection:
    """Create a DuckDB connection with standard pragmas."""
    conn = duckdb.connect(str(db_path), read_only=read_only)
    effective_pragmas: dict[str, object] = {"threads": DEFAULT_THREADS}
    if pragmas:
        effective_pragmas.update({key.lower(): value for key, value in pragmas.items()})
    for key, value in effective_pragmas.items():
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (int, float)):
            literal = str(value)
        else:
            literal = f"'{value}'"
        conn.execute(f"PRAGMA {key}={literal}")
    return conn


def execute(  # noqa: PLR0913
    conn: DuckDBPyConnection,
    sql: str,
    params: Params = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    slow_query_threshold_s: float = DEFAULT_SLOW_QUERY_THRESHOLD_S,
    operation: str = "duckdb.execute",
    require_parameterized: bool | None = None,
) -> DuckDBPyRelation:
    """Execute a DuckDB query with parameter binding, logging, and metrics."""
    require_flag = params is not None if require_parameterized is None else require_parameterized
    _ensure_parameterized(sql, require_parameterized=require_flag)

    sql_preview = _format_sql(sql)
    query_params = _format_params(params)

    with (
        with_fields(
            logger,
            component="registry",
            operation=operation,
            sql_preview=sql_preview,
        ) as log,
        observe_duration(metrics, operation, component="registry") as observer,
    ):
        start = time.perf_counter()
        try:
            _set_timeout(conn, timeout_s)
            relation = conn.execute(sql) if params is None else conn.execute(sql, params)
        except duckdb.Error as exc:
            observer.error()
            log.exception(
                "DuckDB query failed",
                extra={"params": query_params},
            )
            error_message = "DuckDB query failed"
            raise RegistryError(
                error_message,
                cause=exc,
                context={"sql_preview": sql_preview, "params": query_params},
            ) from exc
        else:
            observer.success()
            duration = time.perf_counter() - start
            if duration >= slow_query_threshold_s:
                log.warning(
                    "Slow DuckDB query",
                    extra={"duration_ms": round(duration * 1000, 2), "params": query_params},
                )
            else:
                log.debug(
                    "DuckDB query executed",
                    extra={"duration_ms": round(duration * 1000, 2)},
                )
            return relation


def fetch_all(
    conn: DuckDBPyConnection,
    sql: str,
    params: Params = None,
    **kwargs: object,
) -> list[tuple[Any, ...]]:
    """Execute a query and return all rows as a list of tuples."""
    relation = execute(conn, sql, params, **kwargs)
    return relation.fetchall()


def fetch_one(
    conn: DuckDBPyConnection,
    sql: str,
    params: Params = None,
    **kwargs: object,
) -> tuple[Any, ...] | None:
    """Execute a query and return the first row or None."""
    relation = execute(conn, sql, params, **kwargs)
    return relation.fetchone()


def validate_identifier(
    identifier: str,
    allowed: Iterable[str],
    *,
    label: str = "identifier",
) -> str:
    """Validate that an identifier is within an allowed set."""
    allowed_set = set(allowed)
    if identifier in allowed_set:
        return identifier
    error_message = f"Invalid {label}: {identifier}"
    raise RegistryError(
        error_message,
        context={label: identifier, "allowed": sorted(allowed_set)},
    )
