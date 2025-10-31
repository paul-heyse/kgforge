"""Type stubs for DuckDB.

This module provides type annotations for DuckDB operations
to prevent no-any-unimported errors.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["Connection", "DuckDBConnection"]

class DuckDBConnection(Protocol):
    """Protocol for DuckDB connection objects."""

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a SQL query.

        Parameters
        ----------
        sql : str
            SQL query string.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Query result object with fetchall() method.
        """
        ...

    def close(self) -> None:
        """Close the connection."""
        ...

class Connection:
    """DuckDB connection class stub."""

    def __init__(self, database: str = ":memory:", **kwargs: Any) -> None:
        """Initialize a DuckDB connection.

        Parameters
        ----------
        database : str, optional
            Database path or ":memory:" for in-memory database.
            Defaults to ``":memory:"``.
        **kwargs : Any
            Additional connection parameters.
        """

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a SQL query.

        Parameters
        ----------
        sql : str
            SQL query string.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Query result object with fetchall() method.
        """

    def close(self) -> None:
        """Close the connection."""

def connect(database: str = ":memory:", **kwargs: Any) -> Connection:
    """Create a DuckDB connection.

    Parameters
    ----------
    database : str, optional
        Database path or ":memory:" for in-memory database.
        Defaults to ``":memory:"``.
    **kwargs : Any
        Additional connection parameters.

    Returns
    -------
    Connection
        DuckDB connection instance.
    """
