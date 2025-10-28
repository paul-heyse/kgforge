"""Fixture Flow utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from kgfoundry.kgfoundry_common.models import Doc
from kgfoundry.kgfoundry_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter
from kgfoundry.registry.helper import DuckDBRegistryHelper
from prefect import flow, task

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "fixture_pipeline",
    "t_prepare_dirs",
    "t_register_in_duckdb",
    "t_write_fixture_chunks",
    "t_write_fixture_dense",
    "t_write_fixture_splade",
]

__navmap__: Final[NavMap] = {
    "title": "orchestration.fixture_flow",
    "synopsis": "Prefect tasks that generate fixture parquet datasets and register them.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "t_prepare_dirs",
                "t_write_fixture_chunks",
                "t_write_fixture_dense",
                "t_write_fixture_splade",
                "t_register_in_duckdb",
                "fixture_pipeline",
            ],
        },
    ],
}


# [nav:anchor t_prepare_dirs]
@task
def t_prepare_dirs(root: str) -> dict[str, bool]:
    """Return t prepare dirs.

    Parameters
    ----------
    root : str
        Description for ``root``.

    Returns
    -------
    Mapping[str, bool]
        Description of return value.
    """
    path = Path(root)
    (path / "parquet" / "dense").mkdir(parents=True, exist_ok=True)
    (path / "parquet" / "sparse").mkdir(parents=True, exist_ok=True)
    (path / "parquet" / "chunks").mkdir(parents=True, exist_ok=True)
    (path / "catalog").mkdir(parents=True, exist_ok=True)
    return {"ok": True}


# [nav:anchor t_write_fixture_chunks]
@task
def t_write_fixture_chunks(chunks_root: str) -> tuple[str, int]:
    """Return t write fixture chunks.

    Parameters
    ----------
    chunks_root : str
        Description for ``chunks_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.
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


# [nav:anchor t_write_fixture_dense]
@task
def t_write_fixture_dense(dense_root: str) -> tuple[str, int]:
    """Return t write fixture dense.

    Parameters
    ----------
    dense_root : str
        Description for ``dense_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.
    """
    writer = ParquetVectorWriter(dense_root)
    vector = [0.0] * 2560
    out_root = writer.write_dense(
        "Qwen3-Embedding-4B", "fixture", 2560, [("urn:chunk:fixture:0-28", vector, 1.0)], shard=0
    )
    return out_root, 1


# [nav:anchor t_write_fixture_splade]
@task
def t_write_fixture_splade(sparse_root: str) -> tuple[str, int]:
    """Return t write fixture splade.

    Parameters
    ----------
    sparse_root : str
        Description for ``sparse_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.
    """
    writer = ParquetVectorWriter(sparse_root)
    out_root = writer.write_splade(
        "SPLADE-v3-distilbert",
        "fixture",
        [("urn:chunk:fixture:0-28", [1, 7, 42], [0.3, 0.2, 0.1])],
        shard=0,
    )
    return out_root, 1


# [nav:anchor t_register_in_duckdb]
@task
def t_register_in_duckdb(
    db_path: str,
    chunks_info: tuple[str, int],
    dense_info: tuple[str, int],
    sparse_info: tuple[str, int],
) -> dict[str, list[str]]:
    """Return t register in duckdb.

    Parameters
    ----------
    db_path : str
        Description for ``db_path``.
    chunks_info : Tuple[str, int]
        Description for ``chunks_info``.
    dense_info : Tuple[str, int]
        Description for ``dense_info``.
    sparse_info : Tuple[str, int]
        Description for ``sparse_info``.

    Returns
    -------
    Mapping[str, List[str]]
        Description of return value.
    """
    registry = DuckDBRegistryHelper(db_path)
    dense_run = registry.new_run("dense_embed", "Qwen3-Embedding-4B", "main", {"dim": 2560})
    sparse_run = registry.new_run("splade_encode", "SPLADE-v3-distilbert", "main", {"topk": 256})

    ds_chunks = registry.begin_dataset("chunks", dense_run)
    registry.commit_dataset(ds_chunks, chunks_info[0], rows=chunks_info[1])

    ds_dense = registry.begin_dataset("dense", dense_run)
    registry.commit_dataset(ds_dense, dense_info[0], rows=dense_info[1])

    ds_sparse = registry.begin_dataset("sparse", sparse_run)
    registry.commit_dataset(ds_sparse, sparse_info[0], rows=sparse_info[1])

    registry.register_documents(
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
    registry.close_run(dense_run, True)
    registry.close_run(sparse_run, True)
    return {"runs": [dense_run, sparse_run]}


# [nav:anchor fixture_pipeline]
@flow(name="kgfoundry_fixture_pipeline")
def fixture_pipeline(
    root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb"
) -> dict[str, list[str]]:
    """Return fixture pipeline.

    Parameters
    ----------
    root : str | None
        Description for ``root``.
    db_path : str | None
        Description for ``db_path``.

    Returns
    -------
    Mapping[str, List[str]]
        Description of return value.
    """
    t_prepare_dirs(root)
    chunks_info = t_write_fixture_chunks(f"{root}/parquet/chunks")
    dense_info = t_write_fixture_dense(f"{root}/parquet/dense")
    sparse_info = t_write_fixture_splade(f"{root}/parquet/sparse")
    return t_register_in_duckdb(db_path, chunks_info, dense_info, sparse_info)


if __name__ == "__main__":
    fixture_pipeline()
