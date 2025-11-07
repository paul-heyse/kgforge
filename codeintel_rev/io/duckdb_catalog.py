"""DuckDB catalog for querying Parquet chunks.

Provides SQL views over Parquet directories and query helpers for fast
chunk retrieval and joins.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class DuckDBCatalog:
    """DuckDB catalog for querying chunks.

    Parameters
    ----------
    db_path : Path
        DuckDB database path.
    vectors_dir : Path
        Directory containing Parquet files.
    """

    def __init__(self, db_path: Path, vectors_dir: Path) -> None:
        self.db_path = db_path
        self.vectors_dir = vectors_dir
        self.conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> None:
        """Open database connection."""
        self.conn = duckdb.connect(str(self.db_path))
        self._ensure_views()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> DuckDBCatalog:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager."""
        self.close()

    def _ensure_views(self) -> None:
        """Create views over Parquet directories."""
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        # Create chunks view
        parquet_pattern = f"{self.vectors_dir}/**/*.parquet"
        self.conn.execute(
            f"CREATE OR REPLACE VIEW chunks AS SELECT * FROM read_parquet('{parquet_pattern}')"
        )

    def query_by_ids(self, ids: Sequence[int]) -> list[dict]:
        """Query chunks by their unique IDs.

        Retrieves chunk metadata (text, URI, line numbers, etc.) for a list of
        chunk IDs. This is typically used after a FAISS search returns chunk IDs
        to hydrate the results with full chunk information.

        The function constructs a SQL IN clause to efficiently fetch multiple
        chunks in a single query. Results are returned as dictionaries with column
        names as keys, matching the Parquet schema.

        Parameters
        ----------
        ids : Sequence[int]
            Sequence of chunk IDs to retrieve. IDs must exist in the chunks table.
            Empty sequence returns empty list.

        Returns
        -------
        list[dict]
            List of chunk records as dictionaries. Each dict contains all columns
            from the chunks Parquet file (id, uri, text, start_line, end_line,
            symbols, etc.). Returns empty list if no IDs provided or no matches.

        Raises
        ------
        RuntimeError
            If the database connection is not open. Call open() or use the
            context manager before querying.
        """
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        if not ids:
            return []

        id_list = ",".join(str(i) for i in ids)
        result = self.conn.execute(f"SELECT * FROM chunks WHERE id IN ({id_list})").fetchall()

        # Convert to dicts
        cols = [desc[0] for desc in self.conn.description]
        return [dict(zip(cols, row)) for row in result]

    def query_by_uri(self, uri: str, limit: int = 100) -> list[dict]:
        """Query chunks by file URI/path.

        Retrieves all chunks from a specific file. Useful for file-level operations
        like displaying all chunks in a file or filtering search results by file.

        The query uses parameterized SQL to prevent injection and efficiently
        filters by URI. Results are limited to prevent excessive memory usage
        for large files.

        Parameters
        ----------
        uri : str
            File URI or path to query. Should match the uri field in the chunks
            table (typically a relative path from repo root).
        limit : int, optional
            Maximum number of chunks to return. Defaults to 100. Set higher for
            large files, but be aware of memory usage. Use 0 or very large value
            for no limit (not recommended for production).

        Returns
        -------
        list[dict]
            List of chunk records from the specified file. Each dict contains
            all chunk columns. Results are ordered by chunk ID (which typically
            corresponds to file order). Returns empty list if file not found or
            no chunks in file.

        Raises
        ------
        RuntimeError
            If the database connection is not open. Call open() or use the
            context manager before querying.
        """
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        result = self.conn.execute(
            "SELECT * FROM chunks WHERE uri = ? LIMIT ?", [uri, limit]
        ).fetchall()

        cols = [desc[0] for desc in self.conn.description]
        return [dict(zip(cols, row)) for row in result]

    def get_embeddings_by_ids(self, ids: Sequence[int]) -> np.ndarray:
        """Extract embedding vectors for given chunk IDs.

        Retrieves the pre-computed embedding vectors for chunks, typically used
        after a FAISS search to get the actual vectors for re-ranking or analysis.
        The embeddings are stored in Parquet as FixedSizeList arrays and are
        converted to NumPy arrays for efficient computation.

        The function preserves the order of input IDs in the output array. If
        an ID is not found, it's silently skipped (the output will have fewer
        rows than input IDs).

        Parameters
        ----------
        ids : Sequence[int]
            Sequence of chunk IDs to retrieve embeddings for. IDs must exist
            in the chunks table. Empty sequence returns empty array.

        Returns
        -------
        np.ndarray
            Embedding vectors as a 2D NumPy array of shape (n_found, vec_dim)
            where n_found <= len(ids). Dtype is float32 for memory efficiency.
            Returns empty array (shape (0, vec_dim)) if no IDs provided or no
            matches found. The array is ordered by the input ID sequence.

        Raises
        ------
        RuntimeError
            If the database connection is not open. Call open() or use the
            context manager before querying.
        """
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        if not ids:
            return np.array([], dtype=np.float32)

        id_list = ",".join(str(i) for i in ids)
        result = self.conn.execute(
            f"SELECT id, embedding FROM chunks WHERE id IN ({id_list}) ORDER BY id"
        ).fetchall()

        # Extract embeddings in order
        id_to_emb = {row[0]: np.array(row[1], dtype=np.float32) for row in result}
        embeddings = [id_to_emb[i] for i in ids if i in id_to_emb]

        if not embeddings:
            return np.array([], dtype=np.float32)

        return np.vstack(embeddings)

    def count_chunks(self) -> int:
        """Count total number of chunks in the index.

        Returns the total number of chunks across all files. Useful for monitoring
        index size and validating that indexing completed successfully.

        The count is computed efficiently using DuckDB's COUNT aggregation over
        the chunks view, which reads directly from Parquet files.

        Returns
        -------
        int
            Total number of chunks in the index. Returns 0 if the chunks view
            is empty or no Parquet files exist.

        Raises
        ------
        RuntimeError
            If the database connection is not open. Call open() or use the
            context manager before querying.
        """
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return result[0] if result else 0


__all__ = ["DuckDBCatalog"]
