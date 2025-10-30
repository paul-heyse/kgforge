"""Overview of parquet io.

This module bundles parquet io logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
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
    },
}

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
    def dense_schema(dim: int) -> pa.schema:
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
        records : Iterable[tuple[str, list[float], float]]
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
    def splade_schema() -> pa.schema:
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
        records : Iterable[tuple[str, list[int], list[float]]]
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
    def chunk_schema() -> pa.schema:
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

    def write(self, rows: Iterable[dict[str, Any]]) -> str:
        """Describe write.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        rows : Iterable[dict[str, Any]]
            Describe ``rows``.
            

        Returns
        -------
        str
            Describe return value.
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
