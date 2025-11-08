#!/usr/bin/env python3
"""One-shot indexing: SCIP → chunk → embed → Parquet → FAISS.

This script orchestrates the full indexing pipeline:
1. Parse SCIP index for symbol definitions
2. Chunk files using cAST (SCIP-based)
3. Embed chunks with vLLM
4. Write to Parquet with embeddings
5. Build FAISS index
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codeintel_rev.config.settings import IndexConfig, Settings, VLLMConfig, load_settings
from codeintel_rev.indexing.cast_chunker import Chunk, chunk_file
from codeintel_rev.indexing.scip_reader import (
    SCIPIndex,
    SymbolDef,
    extract_definitions,
    get_top_level_definitions,
    parse_scip_json,
)
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.parquet_store import write_chunks_parquet
from codeintel_rev.io.vllm_client import VLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_PREVIEW_CHARS = 1000
TRAINING_LIMIT = 10_000


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved filesystem paths for the indexing pipeline."""

    repo_root: Path
    scip_index: Path
    vectors_dir: Path
    faiss_index: Path
    duckdb_path: Path


def main() -> None:
    """Run the end-to-end indexing pipeline."""
    settings = load_settings()
    paths = _resolve_paths(settings)

    logger.info("Starting indexing for repo root %s", paths.repo_root)
    scip_index = _load_scip_index(paths)
    grouped_defs = _group_definitions_by_file(scip_index)
    chunks = _chunk_repository(paths, grouped_defs, settings.index.chunk_budget)

    if not chunks:
        logger.error("No chunks were produced; aborting pipeline")
        return

    embeddings = _embed_chunks(chunks, settings.vllm)
    parquet_path = _write_parquet(chunks, embeddings, paths, settings.index.vec_dim)
    _build_faiss_index(embeddings, paths, settings.index)
    catalog_count = _initialize_duckdb(paths)

    logger.info(
        "Indexing pipeline complete; chunks=%s embeddings=%s parquet=%s faiss_index=%s duckdb_rows=%s",
        len(chunks),
        len(embeddings),
        parquet_path,
        paths.faiss_index,
        catalog_count,
    )


def _resolve_paths(settings: Settings) -> PipelinePaths:
    """Resolve and normalize key filesystem paths.

    Parameters
    ----------
    settings : Settings
        Application settings containing path configuration.

    Returns
    -------
    PipelinePaths
        Absolute paths for all filesystem locations used by the pipeline.
    """
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()

    def _resolve(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.expanduser().resolve()
        return (repo_root / path).resolve()

    return PipelinePaths(
        repo_root=repo_root,
        scip_index=_resolve(settings.paths.scip_index),
        vectors_dir=_resolve(settings.paths.vectors_dir),
        faiss_index=_resolve(settings.paths.faiss_index),
        duckdb_path=_resolve(settings.paths.duckdb_path),
    )


def _load_scip_index(paths: PipelinePaths) -> SCIPIndex:
    """Load and parse the SCIP index from disk.

    Parameters
    ----------
    paths : PipelinePaths
        Pipeline paths configuration containing SCIP index location.

    Returns
    -------
    SCIPIndex
        Parsed SCIP index containing all documents and occurrences.

    Raises
    ------
    FileNotFoundError
        If the configured SCIP index file does not exist.
    """
    if not paths.scip_index.exists():
        msg = f"SCIP index not found at {paths.scip_index}"
        raise FileNotFoundError(msg)

    index = parse_scip_json(paths.scip_index)
    logger.info("Parsed %s documents from SCIP index", len(index.documents))
    return index


def _group_definitions_by_file(index: SCIPIndex) -> Mapping[str, list[SymbolDef]]:
    """Group symbol definitions by their relative file path.

    Parameters
    ----------
    index : SCIPIndex
        SCIP index containing symbol definitions.

    Returns
    -------
    Mapping[str, list[SymbolDef]]
        Definitions grouped by file path for downstream chunking.
    """
    grouped: dict[str, list[SymbolDef]] = defaultdict(list)
    for definition in extract_definitions(index):
        grouped[definition.path].append(definition)

    logger.info("Found definitions in %s files", len(grouped))
    return grouped


def _chunk_repository(
    paths: PipelinePaths,
    definitions_by_file: Mapping[str, Sequence[SymbolDef]],
    budget: int,
) -> list[Chunk]:
    """Chunk all files referenced by the SCIP index.

    Parameters
    ----------
    paths : PipelinePaths
        Pipeline paths configuration.
    definitions_by_file : Mapping[str, Sequence[SymbolDef]]
        Symbol definitions grouped by file path.
    budget : int
        Character budget per chunk.

    Returns
    -------
    list[Chunk]
        Collection of generated chunks across the repository.
    """
    chunks: list[Chunk] = []
    for relative_path, defs in definitions_by_file.items():
        full_path = paths.repo_root / relative_path
        if not full_path.exists():
            logger.warning("Skipping missing file %s", full_path)
            continue

        try:
            text = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Unable to read %s: %s", full_path, exc)
            continue

        def_list = list(defs)
        top_level_defs = get_top_level_definitions(def_list)
        if not top_level_defs:
            continue

        file_chunks = chunk_file(full_path, text, top_level_defs, budget=budget)
        chunks.extend(file_chunks)

    logger.info("Chunked %s files into %s chunks", len(definitions_by_file), len(chunks))
    return chunks


def _embed_chunks(chunks: Sequence[Chunk], config: VLLMConfig) -> np.ndarray:
    """Generate embeddings for the supplied chunks using vLLM.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Chunks to embed.
    config : VLLMConfig
        vLLM client configuration.

    Returns
    -------
    np.ndarray
        Embedding matrix aligned with the chunk order.
    """
    client = VLLMClient(config)
    texts = [chunk.text[:EMBED_PREVIEW_CHARS] for chunk in chunks]
    vectors = client.embed_chunks(texts, batch_size=config.batch_size)
    logger.info("Generated %s embeddings", len(vectors))
    return vectors


def _write_parquet(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    paths: PipelinePaths,
    vec_dim: int,
) -> Path:
    """Persist chunk metadata and embeddings to Parquet.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Chunks to persist.
    embeddings : np.ndarray
        Embedding vectors aligned with chunks.
    paths : PipelinePaths
        Pipeline paths configuration.
    vec_dim : int
        Embedding vector dimension.

    Returns
    -------
    Path
        Path to the written Parquet file containing chunk data.
    """
    output_path = paths.vectors_dir / "part-000.parquet"
    write_chunks_parquet(
        output_path=output_path,
        chunks=chunks,
        embeddings=embeddings,
        start_id=0,
        vec_dim=vec_dim,
    )
    logger.info("Wrote Parquet dataset to %s", output_path)
    return output_path


def _build_faiss_index(
    embeddings: np.ndarray,
    paths: PipelinePaths,
    index_config: IndexConfig,
) -> None:
    """Train and persist the FAISS index.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors to index.
    paths : PipelinePaths
        Pipeline paths configuration.
    index_config : IndexConfig
        FAISS index configuration.

    Raises
    ------
    RuntimeError
        If embeddings are empty and the index cannot be trained.
    """
    if embeddings.size == 0:
        msg = "No embeddings available to build FAISS index"
        raise RuntimeError(msg)

    faiss_mgr = FAISSManager(
        index_path=paths.faiss_index,
        vec_dim=index_config.vec_dim,
        nlist=index_config.faiss_nlist,
        use_cuvs=index_config.use_cuvs,
    )
    train_limit = min(len(embeddings), TRAINING_LIMIT)
    training_vectors = embeddings[:train_limit]
    faiss_mgr.build_index(training_vectors)

    ids = np.arange(len(embeddings), dtype=np.int64)
    faiss_mgr.add_vectors(embeddings, ids)
    faiss_mgr.save_cpu_index()
    logger.info("Persisted FAISS index to %s", paths.faiss_index)


def _initialize_duckdb(paths: PipelinePaths) -> int:
    """Create or refresh the DuckDB catalog and return the chunk count.

    Parameters
    ----------
    paths : PipelinePaths
        Pipeline paths configuration.

    Returns
    -------
    int
        Number of chunk records registered in the catalog.
    """
    with DuckDBCatalog(paths.duckdb_path, paths.vectors_dir) as catalog:
        count = catalog.count_chunks()
    logger.info("DuckDB catalog initialized at %s (rows=%s)", paths.duckdb_path, count)
    return count


if __name__ == "__main__":
    main()
