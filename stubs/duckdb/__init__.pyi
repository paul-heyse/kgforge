"""Type stubs for DuckDB.

This module provides type annotations for DuckDB operations
to enable full type checking with parameterized queries and
typed result objects.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

__all__ = ["Connection", "DuckDBConnection", "DuckDBPyConnection", "DuckDBPyRelation"]

class DuckDBPyRelation(Protocol):
    """Protocol for DuckDB relation objects returned by execute()."""

    def fetchall(self) -> list[tuple[object, ...]]:
        """Fetch all rows from the query result.

        Returns
        -------
        list[tuple[object, ...]]
            List of tuples, where each tuple represents a row.
        """
        ...

    def fetchone(self) -> tuple[object, ...] | None:
        """Fetch one row from the query result.

        Returns
        -------
        tuple[object, ...] | None
            Single row tuple or None if no rows remain.
        """
        ...

    def fetchmany(self, size: int = 1) -> list[tuple[object, ...]]:
        """Fetch multiple rows from the query result.

        Parameters
        ----------
        size : int, optional
            Number of rows to fetch. Defaults to 1.

        Returns
        -------
        list[tuple[object, ...]]
            List of row tuples.
        """
        ...

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        """Iterate over query results."""
        ...

class DuckDBPyConnection(Protocol):
    """Protocol for DuckDB connection objects with typed execute methods."""

    def execute(
        self,
        sql: str,
        parameters: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBPyRelation:
        """Execute a SQL query with optional parameters.

        Parameters
        ----------
        sql : str
            SQL query string. Must use parameterized placeholders (? or :name).
        parameters : Sequence[object] | Mapping[str, object] | None, optional
            Query parameters for placeholders. Sequence for positional (?),
            Mapping for named (:name). Defaults to None.

        Returns
        -------
        DuckDBPyRelation
            Query result object with fetchall(), fetchone(), fetchmany() methods.

        Raises
        ------
        RuntimeError
            If SQL contains syntax errors or execution fails.
        ValueError
            If parameter count/types don't match placeholders.

        Examples
        --------
        >>> con = duckdb.connect(":memory:")
        >>> result = con.execute("SELECT ?", [42])
        >>> rows = result.fetchall()
        """
        ...

    def close(self) -> None:
        """Close the connection."""
        ...

class DuckDBConnection(DuckDBPyConnection):
    """Protocol alias for backward compatibility.

    Prefer using DuckDBPyConnection directly.
    """

class Connection:
    """DuckDB connection class stub.

    This class provides type hints for duckdb.connect() return value.
    """

    def __init__(
        self,
        database: str = ":memory:",
        read_only: bool = False,
        config: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize a DuckDB connection.

        Parameters
        ----------
        database : str, optional
            Database path or ":memory:" for in-memory database.
            Defaults to ":memory:".
        read_only : bool, optional
            Whether to open database in read-only mode.
            Defaults to False.
        config : Mapping[str, object] | None, optional
            Configuration options as key-value pairs.
            Defaults to None.
        """

    def execute(
        self,
        sql: str,
        parameters: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBPyRelation:
        """Execute a SQL query with optional parameters.

        Parameters
        ----------
        sql : str
            SQL query string. Must use parameterized placeholders (? or :name).
        parameters : Sequence[object] | Mapping[str, object] | None, optional
            Query parameters for placeholders. Sequence for positional (?),
            Mapping for named (:name). Defaults to None.

        Returns
        -------
        DuckDBPyRelation
            Query result object with fetchall(), fetchone(), fetchmany() methods.

        Raises
        ------
        RuntimeError
            If SQL contains syntax errors or execution fails.
        ValueError
            If parameter count/types don't match placeholders.
        """

    def close(self) -> None:
        """Close the connection."""

class Error(Exception):
    """Base exception for DuckDB errors."""

    pass

def connect(
    database: str = ":memory:",
    read_only: bool = False,
    config: Mapping[str, object] | None = None,
) -> Connection:
    """Create a DuckDB connection.

    Parameters
    ----------
    database : str, optional
        Database path or ":memory:" for in-memory database.
        Defaults to ":memory:".
    read_only : bool, optional
        Whether to open database in read-only mode.
        Defaults to False.
    config : Mapping[str, object] | None, optional
        Configuration options as key-value pairs.
        Defaults to None.

    Returns
    -------
    Connection
        DuckDB connection instance.

    Raises
    ------
    Error
        If connection fails (file not found, permission denied, etc.).
    """
