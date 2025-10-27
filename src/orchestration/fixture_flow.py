"""Module for orchestration.fixture_flow."""

from __future__ import annotations

from pathlib import Path

from kgforge.kgforge_common.models import Doc
from kgforge.kgforge_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter
from kgforge.registry.helper import DuckDBRegistryHelper
from prefect import flow, task


@task
def t_prepare_dirs(root: str) -> dict[str, bool]:
    """T prepare dirs.

    Args:
        root (str): TODO.

    Returns:
        dict: TODO.
    """
    p = Path(root)
    (p / "parquet" / "dense").mkdir(parents=True, exist_ok=True)
    (p / "parquet" / "sparse").mkdir(parents=True, exist_ok=True)
    (p / "parquet" / "chunks").mkdir(parents=True, exist_ok=True)
    (p / "catalog").mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@task
def t_write_fixture_chunks(chunks_root: str) -> tuple[str, int]:
    """T write fixture chunks.

    Args:
        chunks_root (str): TODO.

    Returns:
        tuple[str, int]: TODO.
    """
    writer = ParquetChunkWriter(chunks_root, model="docling_hybrid", run_id="fixture")
    rows = [
        {
            "chunk_id": "urn:chunk:fixture:0-28",
            "doc_id": "urn:doc:fixture:0001",
            "section": "Intro",
            "start_char": 0,
            "end_char": 28,
            "doctags_span": {"node_id": "n1", "start": 0, "end": 28},
            "text": "Hello fixture text about LLMs",
            "tokens": 5,
            "created_at": 0,
        }
    ]
    dataset_root = writer.write(rows)
    return dataset_root, len(rows)


@task
def t_write_fixture_dense(dense_root: str) -> tuple[str, int]:
    """T write fixture dense.

    Args:
        dense_root (str): TODO.

    Returns:
        tuple[str, int]: TODO.
    """
    w = ParquetVectorWriter(dense_root)
    vec = [0.0] * 2560
    out_root = w.write_dense(
        "Qwen3-Embedding-4B", "fixture", 2560, [("urn:chunk:fixture:0-28", vec, 1.0)], shard=0
    )
    return out_root, 1


@task
def t_write_fixture_splade(sparse_root: str) -> tuple[str, int]:
    """T write fixture splade.

    Args:
        sparse_root (str): TODO.

    Returns:
        tuple[str, int]: TODO.
    """
    w = ParquetVectorWriter(sparse_root)
    out_root = w.write_splade(
        "SPLADE-v3-distilbert",
        "fixture",
        [("urn:chunk:fixture:0-28", [1, 7, 42], [0.3, 0.2, 0.1])],
        shard=0,
    )
    return out_root, 1


@task
def t_register_in_duckdb(
    db_path: str,
    chunks_info: tuple[str, int],
    dense_info: tuple[str, int],
    sparse_info: tuple[str, int],
) -> dict[str, list[str]]:
    """T register in duckdb.

    Args:
        db_path (str): TODO.
        chunks_info: TODO.
        dense_info: TODO.
        sparse_info: TODO.

    Returns:
        dict: TODO.
    """
    reg = DuckDBRegistryHelper(db_path)
    dense_run = reg.new_run("dense_embed", "Qwen3-Embedding-4B", "main", {"dim": 2560})
    sparse_run = reg.new_run("splade_encode", "SPLADE-v3-distilbert", "main", {"topk": 256})
    ds_chunks = reg.begin_dataset("chunks", dense_run)
    reg.commit_dataset(ds_chunks, chunks_info[0], rows=chunks_info[1])
    ds_dense = reg.begin_dataset("dense", dense_run)
    reg.commit_dataset(ds_dense, dense_info[0], rows=dense_info[1])
    ds_sparse = reg.begin_dataset("sparse", sparse_run)
    reg.commit_dataset(ds_sparse, sparse_info[0], rows=sparse_info[1])
    reg.register_documents(
        [
            Doc(
                id="urn:doc:fixture:0001",
                title="Fixture Doc",
                authors=[],
                pdf_uri="/dev/null",
                source="fixture",
                openalex_id=None,
                doi=None,
                arxiv_id=None,
                pmcid=None,
                pub_date=None,
                license="CC0",
                language="en",
                content_hash="",
            )
        ]
    )
    reg.close_run(dense_run, True)
    reg.close_run(sparse_run, True)
    return {"runs": [dense_run, sparse_run]}


@flow(name="kgforge_fixture_pipeline")
def fixture_pipeline(
    root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb"
) -> dict[str, list[str]]:
    """Fixture pipeline.

    Args:
        root (str): TODO.
        db_path (str): TODO.
    """
    t_prepare_dirs(root)
    chunks_info = t_write_fixture_chunks(f"{root}/parquet/chunks")
    dense_info = t_write_fixture_dense(f"{root}/parquet/dense")
    sparse_info = t_write_fixture_splade(f"{root}/parquet/sparse")
    summary = t_register_in_duckdb(db_path, chunks_info, dense_info, sparse_info)
    return summary


if __name__ == "__main__":
    fixture_pipeline()
