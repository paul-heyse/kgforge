"""Module for kgforge_common.parquet_io."""


from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
import datetime as dt

ZSTD_LEVEL = 6
ROW_GROUP_SIZE = 4096

class ParquetVectorWriter:
    """Parquetvectorwriter."""
    @staticmethod
    def dense_schema(dim: int) -> pa.schema:
        """Dense schema.

        Args:
            dim (int): TODO.

        Returns:
            pa.schema: TODO.
        """
        return pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("model", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("dim", pa.int16()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
            pa.field("l2_norm", pa.float32()),
            pa.field("created_at", pa.timestamp("ms")),
        ])

    def __init__(self, root: str):
        """Init.

        Args:
            root (str): TODO.
        """
        self.root = Path(root)

    def write_dense(self, model: str, run_id: str, dim: int,
                    records: Iterable[Tuple[str, List[float], float]], shard: int=0) -> str:
        """Write dense.

        Args:
            model (str): TODO.
            run_id (str): TODO.
            dim (int): TODO.
            records (Iterable[Tuple[str, List[float], float]]): TODO.
            shard (int): TODO.

        Returns:
            str: TODO.
        """
        part_dir = self.root / f"model={model}" / f"run_id={run_id}" / f"shard={shard:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        now = int(dt.datetime.now(dt.timezone.utc).timestamp()*1000)
        rows = [{"chunk_id":cid, "model":model, "run_id":run_id, "dim":dim, "vector":vec, "l2_norm":float(l2), "created_at": now} for cid, vec, l2 in records]
        table = pa.Table.from_pylist(rows, schema=self.dense_schema(dim))
        pq.write_table(table, part_dir / "part-00000.parquet", compression="ZSTD", compression_level=ZSTD_LEVEL, data_page_size=ROW_GROUP_SIZE)
        return str(self.root)

    @staticmethod
    def splade_schema() -> pa.schema:
        """Splade schema.

        Returns:
            pa.schema: TODO.
        """
        return pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("model", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("vocab_ids", pa.list_(pa.int32())),
            pa.field("weights", pa.list_(pa.float32())),
            pa.field("nnz", pa.int16()),
            pa.field("created_at", pa.timestamp("ms")),
        ])

    def write_splade(self, model: str, run_id: str,
                     records: Iterable[Tuple[str, List[int], List[float]]], shard: int=0) -> str:
        """Write splade.

        Args:
            model (str): TODO.
            run_id (str): TODO.
            records (Iterable[Tuple[str, List[int], List[float]]]): TODO.
            shard (int): TODO.

        Returns:
            str: TODO.
        """
        part_dir = self.root / f"model={model}" / f"run_id={run_id}" / f"shard={shard:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        now = int(dt.datetime.now(dt.timezone.utc).timestamp()*1000)
        rows = [{"chunk_id":cid, "model":model, "run_id":run_id, "vocab_ids":ids, "weights":wts, "nnz":len(ids), "created_at":now} for cid, ids, wts in records]
        table = pa.Table.from_pylist(rows, schema=self.splade_schema())
        pq.write_table(table, part_dir / "part-00000.parquet", compression="ZSTD", compression_level=ZSTD_LEVEL, data_page_size=ROW_GROUP_SIZE)
        return str(self.root)

class ParquetChunkWriter:
    """Parquetchunkwriter."""
    @staticmethod
    def chunk_schema() -> pa.schema:
        """Chunk schema.

        Returns:
            pa.schema: TODO.
        """
        return pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("section", pa.string()),
            pa.field("start_char", pa.int32()),
            pa.field("end_char", pa.int32()),
            pa.field("doctags_span", pa.struct([
                pa.field("node_id", pa.string()),
                pa.field("start", pa.int32()),
                pa.field("end", pa.int32()),
            ])),
            pa.field("text", pa.string()),
            pa.field("tokens", pa.int32()),
            pa.field("created_at", pa.timestamp("ms")),
        ])

    def __init__(self, root: str, model: str="docling_hybrid", run_id: str="dev"):
        """Init.

        Args:
            root (str): TODO.
            model (str): TODO.
            run_id (str): TODO.
        """
        self.root = Path(root) / f"model={model}" / f"run_id={run_id}" / "shard=00000"
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, rows: Iterable[Dict[str, Any]]) -> str:
        """Write.

        Args:
            rows (Iterable[Dict[str, Any]]): TODO.

        Returns:
            str: TODO.
        """
        table = pa.Table.from_pylist(list(rows), schema=self.chunk_schema())
        pq.write_table(table, self.root / "part-00000.parquet", compression="ZSTD", compression_level=ZSTD_LEVEL, data_page_size=ROW_GROUP_SIZE)
        return str(self.root.parent.parent)
