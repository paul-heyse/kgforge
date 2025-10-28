"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.parquet_io
"""


from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from kgfoundry_common.navmap_types import NavMap

__all__ = ["ParquetChunkWriter", "ParquetVectorWriter"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.parquet_io",
    "synopsis": "Utilities for writing embedding vectors and chunks to Parquet",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["ParquetVectorWriter", "ParquetChunkWriter"],
        },
    ],
}

ZSTD_LEVEL = 6
ROW_GROUP_SIZE = 4096


# [nav:anchor ParquetVectorWriter]
class ParquetVectorWriter:
    """Represent ParquetVectorWriter.

    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    dense_schema()
        Method description.
    __init__()
        Method description.
    write_dense()
        Method description.
    splade_schema()
        Method description.
    write_splade()
        Method description.
    
    Examples
    --------
    >>> from kgfoundry_common.parquet_io import ParquetVectorWriter
    >>> result = ParquetVectorWriter()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.parquet_io
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    @staticmethod
    def dense_schema(dim: int) -> pa.schema:
        """Return dense schema.

        Parameters
        ----------
        dim : int
            Description for ``dim``.
        
        Returns
        -------
        pa.schema
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import dense_schema
        >>> result = dense_schema(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("model", pa.string()),
                pa.field("run_id", pa.string()),
                pa.field("dim", pa.int16()),
                pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
                pa.field("l2_norm", pa.float32()),
                pa.field("created_at", pa.timestamp("ms")),
            ]
        )

    def __init__(self, root: str) -> None:
        """Return init.

        Parameters
        ----------
        root : str
            Description for ``root``.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import __init__
        >>> __init__(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
        """Return write dense.

        Parameters
        ----------
        model : str
            Description for ``model``.
        run_id : str
            Description for ``run_id``.
        dim : int
            Description for ``dim``.
        records : Iterable[Tuple[str, List[float], float]]
            Description for ``records``.
        shard : int, optional
            Description for ``shard``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import write_dense
        >>> result = write_dense(..., ..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
    def splade_schema() -> pa.schema:
        """Return splade schema.

        Returns
        -------
        pa.schema
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import splade_schema
        >>> result = splade_schema()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("model", pa.string()),
                pa.field("run_id", pa.string()),
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
        """Return write splade.

        Parameters
        ----------
        model : str
            Description for ``model``.
        run_id : str
            Description for ``run_id``.
        records : Iterable[Tuple[str, List[int], List[float]]]
            Description for ``records``.
        shard : int, optional
            Description for ``shard``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import write_splade
        >>> result = write_splade(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
    """Represent ParquetChunkWriter.

    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    chunk_schema()
        Method description.
    __init__()
        Method description.
    write()
        Method description.
    
    Examples
    --------
    >>> from kgfoundry_common.parquet_io import ParquetChunkWriter
    >>> result = ParquetChunkWriter()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.parquet_io
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    @staticmethod
    def chunk_schema() -> pa.schema:
        """Return chunk schema.

        Returns
        -------
        pa.schema
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import chunk_schema
        >>> result = chunk_schema()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
                    ).with_nullable(True),
                ),
                pa.field("text", pa.string()),
                pa.field("tokens", pa.int32()),
                pa.field("created_at", pa.timestamp("ms")),
            ]
        )

    def __init__(self, root: str, model: str = "docling_hybrid", run_id: str = "dev") -> None:
        """Return init.

        Parameters
        ----------
        root : str
            Description for ``root``.
        model : str, optional
            Description for ``model``.
        run_id : str, optional
            Description for ``run_id``.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import __init__
        >>> __init__(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.root = Path(root) / f"model={model}" / f"run_id={run_id}" / "shard=00000"
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, rows: Iterable[dict[str, Any]]) -> str:
        """Return write.

        Parameters
        ----------
        rows : Iterable[dict[str, Any]]
            Description for ``rows``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from kgfoundry_common.parquet_io import write
        >>> result = write(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kgfoundry_common.parquet_io
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        table = pa.Table.from_pylist(list(rows), schema=self.chunk_schema())
        pq.write_table(
            table,
            self.root / "part-00000.parquet",
            compression="ZSTD",
            compression_level=ZSTD_LEVEL,
            data_page_size=ROW_GROUP_SIZE,
        )
        return str(self.root.parent.parent)
