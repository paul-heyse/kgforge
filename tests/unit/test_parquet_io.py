from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from kgfoundry.kgfoundry_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter


def test_parquet_dense_write(tmp_path: Path) -> None:
    root = tmp_path / "dense"
    w = ParquetVectorWriter(str(root))
    vec = [0.0] * 2560
    w.write_dense("Qwen3-Embedding-4B", "run123", 2560, [("chunk:1", vec, 1.0)])
    files = list(root.rglob("*.parquet"))
    assert files, "Dense parquet not written"
    t = pq.read_table(files[0])
    assert t.num_rows == 1
    assert t.schema.field("vector").type.value_type == pa.float32()


def test_parquet_splade_write(tmp_path: Path) -> None:
    root = tmp_path / "splade"
    w = ParquetVectorWriter(str(root))
    w.write_splade("SPLADE-v3-distilbert", "run123", [("chunk:1", [1, 2, 3], [0.1, 0.2, 0.3])])
    files = list(root.rglob("*.parquet"))
    assert files, "SPLADE parquet not written"


def test_parquet_chunks_write(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    w = ParquetChunkWriter(str(root), model="docling_hybrid", run_id="runX")
    dataset_root = w.write(
        [
            {
                "chunk_id": "urn:chunk:abc:0-10",
                "doc_id": "urn:doc:abc",
                "section": "Intro",
                "start_char": 0,
                "end_char": 10,
                "doctags_span": {"node_id": "n1", "start": 0, "end": 10},
                "text": "Hello world",
                "tokens": 2,
                "created_at": 0,
            }
        ]
    )
    files = list(Path(dataset_root).rglob("*.parquet"))
    assert files, "Chunk parquet not written"
