"""Overview of fixture flow.

This module bundles fixture flow logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from prefect import flow, task

from kgfoundry_common.models import Doc
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter
from registry.helper import DuckDBRegistryHelper

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
    "synopsis": "Prefect flows that build local fixture datasets",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@orchestration",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@orchestration",
            "stability": "experimental",
            "since": "0.1.0",
        }
        for name in __all__
    },
}


# [nav:anchor t_prepare_dirs]
def _t_prepare_dirs_impl(root: str) -> dict[str, bool]:
    """Compute t prepare dirs.

    Carry out the t prepare dirs operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    root : str
        Description for ``root``.

    Returns
    -------
    collections.abc.Mapping
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import t_prepare_dirs
    >>> result = t_prepare_dirs(...)
    >>> result  # doctest: +ELLIPSIS
    """
    path = Path(root)
    (path / "parquet" / "dense").mkdir(parents=True, exist_ok=True)
    (path / "parquet" / "sparse").mkdir(parents=True, exist_ok=True)
    (path / "parquet" / "chunks").mkdir(parents=True, exist_ok=True)
    (path / "catalog").mkdir(parents=True, exist_ok=True)
    return {"ok": True}


# [nav:anchor t_prepare_dirs]
t_prepare_dirs = task(_t_prepare_dirs_impl)


# [nav:anchor t_write_fixture_chunks]
def _t_write_fixture_chunks_impl(chunks_root: str) -> tuple[str, int]:
    """Compute t write fixture chunks.

    Carry out the t write fixture chunks operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    chunks_root : str
        Description for ``chunks_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import t_write_fixture_chunks
    >>> result = t_write_fixture_chunks(...)
    >>> result  # doctest: +ELLIPSIS
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


# [nav:anchor t_write_fixture_chunks]
t_write_fixture_chunks = task(_t_write_fixture_chunks_impl)


# [nav:anchor t_write_fixture_dense]
def _t_write_fixture_dense_impl(dense_root: str) -> tuple[str, int]:
    """Compute t write fixture dense.

    Carry out the t write fixture dense operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    dense_root : str
        Description for ``dense_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import t_write_fixture_dense
    >>> result = t_write_fixture_dense(...)
    >>> result  # doctest: +ELLIPSIS
    """
    writer = ParquetVectorWriter(dense_root)
    vector = [0.0] * 2560
    out_root = writer.write_dense(
        "Qwen3-Embedding-4B", "fixture", 2560, [("urn:chunk:fixture:0-28", vector, 1.0)], shard=0
    )
    return out_root, 1


# [nav:anchor t_write_fixture_dense]
t_write_fixture_dense = task(_t_write_fixture_dense_impl)


# [nav:anchor t_write_fixture_splade]
def _t_write_fixture_splade_impl(sparse_root: str) -> tuple[str, int]:
    """Compute t write fixture splade.

    Carry out the t write fixture splade operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    sparse_root : str
        Description for ``sparse_root``.

    Returns
    -------
    Tuple[str, int]
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import t_write_fixture_splade
    >>> result = t_write_fixture_splade(...)
    >>> result  # doctest: +ELLIPSIS
    """
    writer = ParquetVectorWriter(sparse_root)
    out_root = writer.write_splade(
        "SPLADE-v3-distilbert",
        "fixture",
        [("urn:chunk:fixture:0-28", [1, 7, 42], [0.3, 0.2, 0.1])],
        shard=0,
    )
    return out_root, 1


# [nav:anchor t_write_fixture_splade]
t_write_fixture_splade = task(_t_write_fixture_splade_impl)


# [nav:anchor t_register_in_duckdb]
def _t_register_in_duckdb_impl(
    db_path: str,
    chunks_info: tuple[str, int],
    dense_info: tuple[str, int],
    sparse_info: tuple[str, int],
) -> dict[str, list[str]]:
    """Compute t register in duckdb.

    Carry out the t register in duckdb operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
    collections.abc.Mapping
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import t_register_in_duckdb
    >>> result = t_register_in_duckdb(..., ..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
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


# [nav:anchor t_register_in_duckdb]
t_register_in_duckdb = task(_t_register_in_duckdb_impl)


# [nav:anchor fixture_pipeline]
def _fixture_pipeline_impl(
    root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb"
) -> dict[str, list[str]]:
    """Compute fixture pipeline.

    Carry out the fixture pipeline operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    root : str | None
        Optional parameter default ``'/data'``. Description for ``root``.
    db_path : str | None
        Optional parameter default ``'/data/catalog/catalog.duckdb'``. Description for ``db_path``.

    Returns
    -------
    collections.abc.Mapping
        Description of return value.

    Examples
    --------
    >>> from orchestration.fixture_flow import fixture_pipeline
    >>> result = fixture_pipeline()
    >>> result  # doctest: +ELLIPSIS
    """
    t_prepare_dirs(root)
    chunks_info = t_write_fixture_chunks(f"{root}/parquet/chunks")
    dense_info = t_write_fixture_dense(f"{root}/parquet/dense")
    sparse_info = t_write_fixture_splade(f"{root}/parquet/sparse")
    return cast(
        dict[str, list[str]],
        t_register_in_duckdb(db_path, chunks_info, dense_info, sparse_info),
    )


# [nav:anchor fixture_pipeline]
fixture_pipeline = flow(name="kgfoundry_fixture_pipeline")(_fixture_pipeline_impl)


if __name__ == "__main__":
    fixture_pipeline()
