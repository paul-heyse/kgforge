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
    """Describe  format sql.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    sql : str
        Configure the sql.

    Returns
    -------
    str
        Describe return value.
"""
    compact = " ".join(sql.split())
    if len(compact) <= MAX_SQL_PREVIEW_CHARS:
        return compact
    return f"{compact[:MAX_SQL_PREVIEW_CHARS]}…"


def _truncate_value(value: object) -> object:
    """Describe  truncate value.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    value : object
        Configure the value.

    Returns
    -------
    object
        Describe return value.
"""
    if isinstance(value, str) and len(value) > MAX_SQL_PREVIEW_CHARS:
        return value[:MAX_SQL_PREVIEW_CHARS] + "…"
    return value


def _format_params(params: Params) -> object:
    """Describe  format params.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    params : object | str | object | NoneType
        Configure the params.

    Returns
    -------
    object
        Describe return value.
"""
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return {str(key): _truncate_value(value) for key, value in params.items()}
    return [_truncate_value(value) for value in params]


def _ensure_parameterized(sql: str, *, require_parameterized: bool) -> None:
    """Describe  ensure parameterized.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    sql : str
        Configure the sql.
    require_parameterized : bool
        Indicate whether require parameterized.

    Raises
    ------
    RegistryError
        Raised when error_message.
"""
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
    """Describe  set timeout.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    conn : DuckDBPyConnection
        Configure the conn.
    timeout_s : float
        Configure the timeout s.
"""
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
    """Create a DuckDB connection with standard pragmas.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    db_path : str | Path
        Describe ``db_path``.
    read_only : bool, optional
        Describe ``read_only``.
        Defaults to ``False``.
    pragmas : str | object | NoneType, optional
        Describe ``pragmas``.
        Defaults to ``None``.

    Returns
    -------
    DuckDBPyConnection
        Describe return value.
"""
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
    """Execute a DuckDB query with parameter binding, logging, and metrics.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    conn : DuckDBPyConnection
        Describe ``conn``.
    sql : str
        Describe ``sql``.
    params : object | str | object | NoneType, optional
        Describe ``params``.
        Defaults to ``None``.
    timeout_s : float, optional
        Describe ``timeout_s``.
        Defaults to ``5.0``.
    slow_query_threshold_s : float, optional
        Describe ``slow_query_threshold_s``.
        Defaults to ``0.5``.
    operation : str, optional
        Describe ``operation``.
        Defaults to ``'duckdb.execute'``.
    require_parameterized : bool | NoneType, optional
        Describe ``require_parameterized``.
        Defaults to ``None``.

    Returns
    -------
    DuckDBPyRelation
        Describe return value.
"""
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
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    slow_query_threshold_s: float = DEFAULT_SLOW_QUERY_THRESHOLD_S,
    operation: str = "duckdb.fetch_all",
    require_parameterized: bool | None = None,
) -> list[tuple[Any, ...]]:
    """Execute a query and return all rows as a list of tuples.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    conn : DuckDBPyConnection
        Describe ``conn``.
    sql : str
        Describe ``sql``.
    params : object | str | object | NoneType, optional
        Describe ``params``.
        Defaults to ``None``.
    timeout_s : float, optional
        Describe ``timeout_s``.
        Defaults to ``5.0``.
    slow_query_threshold_s : float, optional
        Describe ``slow_query_threshold_s``.
        Defaults to ``0.5``.
    operation : str, optional
        Describe ``operation``.
        Defaults to ``'duckdb.fetch_all'``.
    require_parameterized : bool | NoneType, optional
        Describe ``require_parameterized``.
        Defaults to ``None``.

    Returns
    -------
    list[tuple[Any, ...]]
        Describe return value.
"""
    relation = execute(
        conn,
        sql,
        params,
        timeout_s=timeout_s,
        slow_query_threshold_s=slow_query_threshold_s,
        operation=operation,
        require_parameterized=require_parameterized,
    )
    return relation.fetchall()


def fetch_one(
    conn: DuckDBPyConnection,
    sql: str,
    params: Params = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    slow_query_threshold_s: float = DEFAULT_SLOW_QUERY_THRESHOLD_S,
    operation: str = "duckdb.fetch_one",
    require_parameterized: bool | None = None,
) -> tuple[Any, ...] | None:
    """Execute a query and return the first row or None.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    conn : DuckDBPyConnection
        Describe ``conn``.
    sql : str
        Describe ``sql``.
    params : object | str | object | NoneType, optional
        Describe ``params``.
        Defaults to ``None``.
    timeout_s : float, optional
        Describe ``timeout_s``.
        Defaults to ``5.0``.
    slow_query_threshold_s : float, optional
        Describe ``slow_query_threshold_s``.
        Defaults to ``0.5``.
    operation : str, optional
        Describe ``operation``.
        Defaults to ``'duckdb.fetch_one'``.
    require_parameterized : bool | NoneType, optional
        Describe ``require_parameterized``.
        Defaults to ``None``.

    Returns
    -------
    tuple[Any, ...] | NoneType
        Describe return value.
"""
    relation = execute(
        conn,
        sql,
        params,
        timeout_s=timeout_s,
        slow_query_threshold_s=slow_query_threshold_s,
        operation=operation,
        require_parameterized=require_parameterized,
    )
    return relation.fetchone()


def validate_identifier(
    identifier: str,
    allowed: Iterable[str],
    *,
    label: str = "identifier",
) -> str:
    """Validate that an identifier is within an allowed set.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    identifier : str
        Describe ``identifier``.
    allowed : str
        Describe ``allowed``.
    label : str, optional
        Describe ``label``.
        Defaults to ``'identifier'``.

    Returns
    -------
    str
        Describe return value.
"""
    allowed_set = set(allowed)
    if identifier in allowed_set:
        return identifier
    error_message = f"Invalid {label}: {identifier}"
    raise RegistryError(
        error_message,
        context={label: identifier, "allowed": sorted(allowed_set)},
    )
