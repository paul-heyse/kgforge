"""Overview of parquet io.

This module bundles parquet io logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, NotRequired, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq

from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.navmap_types import NavMap

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

pd: ModuleType | None
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

__all__ = [
    "ChunkDocTags",
    "ChunkRow",
    "ParquetChunkWriter",
    "ParquetVectorWriter",
    "read_table",
    "read_table_to_dataframe",
    "validate_table_schema",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.parquet_io",
    "synopsis": "Utilities for writing embedding vectors and chunks to Parquet",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "ParquetVectorWriter",
                "ParquetChunkWriter",
                "read_table",
                "read_table_to_dataframe",
                "validate_table_schema",
            ],
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        "ParquetVectorWriter": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "ParquetChunkWriter": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "read_table": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "read_table_to_dataframe": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "validate_table_schema": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}


class ChunkDocTags(TypedDict, total=False):
    """Structured metadata describing a chunk's doctags span."""

    node_id: str
    start: int
    end: int


class ChunkRow(TypedDict, total=False):
    """Row expected by :class:`ParquetChunkWriter`."""

    chunk_id: str
    doc_id: str
    section: str
    start_char: int
    end_char: int
    doctags_span: NotRequired[ChunkDocTags | None]
    text: str
    tokens: int
    created_at: NotRequired[int]


ZSTD_LEVEL = 6
ROW_GROUP_SIZE = 4096


# [nav:anchor ParquetVectorWriter]
class ParquetVectorWriter:
    """Describe ParquetVectorWriter.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    root : str
        Describe ``root``.
    """

    @staticmethod
    def dense_schema(dim: int) -> pa.Schema:
        """Describe dense schema.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        dim : int
            Describe ``dim``.

        Returns
        -------
        pyarrow.lib.schema
            Describe return value.
        """
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("model", pa.dictionary(pa.int32(), pa.string())),
                pa.field("run_id", pa.dictionary(pa.int32(), pa.string())),
                pa.field("dim", pa.int16()),
                pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
                pa.field("l2_norm", pa.float32()),
                pa.field("created_at", pa.timestamp("ms")),
            ]
        )

    def __init__(self, root: str) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        root : str
            Describe ``root``.
        """
        self.root = Path(root)

    def write_dense(
        self,
        model: str,
        run_id: str,
        dim: int,
        records: Iterable[tuple[str, list[float], float]],
        shard: int = 0,
    ) -> str:
        """Describe write dense.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        model : str
            Describe ``model``.
        run_id : str
            Describe ``run_id``.
        dim : int
            Describe ``dim``.
        records : tuple[str, list[float], float]
            Describe ``records``.
        shard : int, optional
            Describe ``shard``.
            Defaults to ``0``.

        Returns
        -------
        str
            Describe return value.
        """
        part_dir = self.root / f"model={model}" / f"run_id={run_id}" / f"shard={shard:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        now = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
        rows = [
            {
                "chunk_id": cid,
                "model": model,
                "run_id": run_id,
                "dim": dim,
                "vector": vec,
                "l2_norm": float(l2),
                "created_at": now,
            }
            for cid, vec, l2 in records
        ]
        table = pa.Table.from_pylist(rows, schema=self.dense_schema(dim))
        pq.write_table(
            table,
            part_dir / "part-00000.parquet",
            compression="ZSTD",
            compression_level=ZSTD_LEVEL,
            data_page_size=ROW_GROUP_SIZE,
        )
        return str(self.root)

    @staticmethod
    def splade_schema() -> pa.Schema:
        """Describe splade schema.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Returns
        -------
        pyarrow.lib.schema
            Describe return value.
        """
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("model", pa.dictionary(pa.int32(), pa.string())),
                pa.field("run_id", pa.dictionary(pa.int32(), pa.string())),
                pa.field("vocab_ids", pa.list_(pa.int32())),
                pa.field("weights", pa.list_(pa.float32())),
                pa.field("nnz", pa.int16()),
                pa.field("created_at", pa.timestamp("ms")),
            ]
        )

    def write_splade(
        self,
        model: str,
        run_id: str,
        records: Iterable[tuple[str, list[int], list[float]]],
        shard: int = 0,
    ) -> str:
        """Describe write splade.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        model : str
            Describe ``model``.
        run_id : str
            Describe ``run_id``.
        records : tuple[str, list[int], list[float]]
            Describe ``records``.
        shard : int, optional
            Describe ``shard``.
            Defaults to ``0``.

        Returns
        -------
        str
            Describe return value.
        """
        part_dir = self.root / f"model={model}" / f"run_id={run_id}" / f"shard={shard:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        now = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
        rows = [
            {
                "chunk_id": cid,
                "model": model,
                "run_id": run_id,
                "vocab_ids": ids,
                "weights": wts,
                "nnz": len(ids),
                "created_at": now,
            }
            for cid, ids, wts in records
        ]
        table = pa.Table.from_pylist(rows, schema=self.splade_schema())
        pq.write_table(
            table,
            part_dir / "part-00000.parquet",
            compression="ZSTD",
            compression_level=ZSTD_LEVEL,
            data_page_size=ROW_GROUP_SIZE,
        )
        return str(self.root)


# [nav:anchor ParquetChunkWriter]
class ParquetChunkWriter:
    """Describe ParquetChunkWriter.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    root : str
        Describe ``root``.
    model : str, optional
        Describe ``model``.
        Defaults to ``'docling_hybrid'``.
    run_id : str, optional
        Describe ``run_id``.
        Defaults to ``'dev'``.
    """

    @staticmethod
    def chunk_schema() -> pa.Schema:
        """Describe chunk schema.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Returns
        -------
        pyarrow.lib.schema
            Describe return value.
        """
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("section", pa.string()),
                pa.field("start_char", pa.int32()),
                pa.field("end_char", pa.int32()),
                pa.field(
                    "doctags_span",
                    pa.struct(
                        [
                            pa.field("node_id", pa.string()),
                            pa.field("start", pa.int32()),
                            pa.field("end", pa.int32()),
                        ]
                    ),
                    nullable=True,
                ),
                pa.field("text", pa.string()),
                pa.field("tokens", pa.int32()),
                pa.field("created_at", pa.timestamp("ms")),
            ]
        )

    def __init__(self, root: str, model: str = "docling_hybrid", run_id: str = "dev") -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        root : str
            Describe ``root``.
        model : str, optional
            Describe ``model``.
            Defaults to ``'docling_hybrid'``.
        run_id : str, optional
            Describe ``run_id``.
            Defaults to ``'dev'``.
        """
        self.root = Path(root) / f"model={model}" / f"run_id={run_id}" / "shard=00000"
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, rows: Iterable[ChunkRow]) -> str:
        """Describe write.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        rows : dict[str, object]
            Describe ``rows``.

        Returns
        -------
        str
            Describe return value.
        """
        timestamp_ms = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
        prepared: list[ChunkRow] = []
        for row in rows:
            materialized: ChunkRow = {**row}
            materialized.setdefault("created_at", timestamp_ms)
            prepared.append(materialized)
        table = pa.Table.from_pylist(prepared, schema=self.chunk_schema())
        pq.write_table(
            table,
            self.root / "part-00000.parquet",
            compression="ZSTD",
            compression_level=ZSTD_LEVEL,
            data_page_size=ROW_GROUP_SIZE,
        )
        return str(self.root.parent.parent)


# [nav:anchor read_table]
def read_table(
    path: str | Path,
    *,
    schema: pa.Schema | None = None,
    validate_schema: bool = True,
) -> pa.Table:
    """Read a Parquet file and return a typed Table.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : str | Path
        Path to Parquet file or directory.
    schema : pyarrow.lib.Schema | NoneType, optional
        Expected schema for validation. If provided and validate_schema=True,
        raises DeserializationError on mismatch.
        Defaults to ``None``.
    validate_schema : bool, optional
        Whether to validate against the provided schema.
        Defaults to ``True``.

    Returns
    -------
    pyarrow.lib.Table
        Typed pyarrow Table with concrete schema.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.
    DeserializationError
        If schema validation fails or the file is corrupted.

    Examples
    --------
    >>> from pathlib import Path
    >>> from kgfoundry_common.parquet_io import read_table
    >>> # Note: requires existing Parquet file
    >>> # table = read_table("data.parquet")
    >>> # assert table.num_rows > 0
    """
    path_obj = Path(path)
    if not path_obj.exists():
        msg = f"Parquet file not found: {path_obj}"
        raise FileNotFoundError(msg)

    try:
        table = pq.read_table(path_obj)
    except Exception as exc:
        msg = f"Failed to read Parquet file {path_obj}: {exc}"
        raise DeserializationError(msg) from exc

    if schema is not None and validate_schema:
        validate_table_schema(table, schema)

    return table


# [nav:anchor read_table_to_dataframe]
def read_table_to_dataframe(
    path: str | Path,
    *,
    schema: pa.Schema | None = None,
    validate_schema: bool = True,
) -> DataFrame:
    """Read a Parquet file and return a typed DataFrame.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : str | Path
        Path to Parquet file or directory.
    schema : pyarrow.lib.Schema | NoneType, optional
        Expected schema for validation. If provided and validate_schema=True,
        raises DeserializationError on mismatch.
        Defaults to ``None``.
    validate_schema : bool, optional
        Whether to validate against the provided schema.
        Defaults to ``True``.

    Returns
    -------
    pandas.core.frame.DataFrame
        Typed pandas DataFrame with schema metadata preserved.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.
    DeserializationError
        If schema validation fails or the file is corrupted.
    ImportError
        If pandas is not installed.

    Examples
    --------
    >>> from pathlib import Path
    >>> from kgfoundry_common.parquet_io import read_table_to_dataframe
    >>> # Note: requires existing Parquet file and pandas
    >>> # df = read_table_to_dataframe("data.parquet")
    >>> # assert len(df) > 0
    """
    if pd is None:
        msg = "pandas is required for DataFrame conversion"
        raise ImportError(msg)

    table = read_table(path, schema=schema, validate_schema=validate_schema)
    return cast(DataFrame, table.to_pandas())


# [nav:anchor validate_table_schema]
def validate_table_schema(table: pa.Table, expected_schema: pa.Schema) -> None:
    """Validate that a table matches an expected schema.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    table : pyarrow.lib.Table
        Table to validate.
    expected_schema : pyarrow.lib.Schema
        Expected schema.

    Raises
    ------
    DeserializationError
        If the table schema does not match the expected schema.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from kgfoundry_common.parquet_io import validate_table_schema
    >>> schema = pa.schema([pa.field("id", pa.string())])
    >>> table = pa.Table.from_pylist([{"id": "test"}], schema=schema)
    >>> validate_table_schema(table, schema)  # No error
    """
    if not table.schema.equals(expected_schema):
        msg = f"Schema mismatch: expected {expected_schema}, got {table.schema}"
        raise DeserializationError(msg)
