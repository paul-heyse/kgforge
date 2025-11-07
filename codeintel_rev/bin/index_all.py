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
from pathlib import Path

import numpy as np
from codeintel_rev.config.settings import load_settings
from codeintel_rev.indexing.cast_chunker import Chunk, chunk_file
from codeintel_rev.indexing.scip_reader import (
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


def main() -> None:
    """Run full indexing pipeline."""
    settings = load_settings()

    logger.info("Loading configuration...")
    logger.info(f"Repo root: {settings.paths.repo_root}")
    logger.info(f"SCIP index: {settings.paths.scip_index}")
    logger.info(f"Output dir: {settings.paths.vectors_dir}")

    # Parse SCIP index
    logger.info("Parsing SCIP index...")
    scip_path = Path(settings.paths.repo_root) / settings.paths.scip_index
    if not scip_path.exists():
        msg = f"SCIP index not found: {scip_path}"
        raise FileNotFoundError(msg)

    scip_index = parse_scip_json(scip_path)
    logger.info(f"Parsed {len(scip_index.documents)} documents")

    # Group definitions by file
    logger.info("Grouping definitions by file...")
    definitions_by_file: dict[str, list] = {}
    for defn in extract_definitions(scip_index):
        if defn.path not in definitions_by_file:
            definitions_by_file[defn.path] = []
        definitions_by_file[defn.path].append(defn)

    logger.info(f"Processing {len(definitions_by_file)} files")

    # Chunk all files
    logger.info("Chunking files...")
    all_chunks: list[Chunk] = []
    repo_root = Path(settings.paths.repo_root)

    for file_path, defs in definitions_by_file.items():
        full_path = repo_root / file_path
        if not full_path.exists():
            logger.warning("File not found: %s", full_path)
            continue

        try:
            text = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", full_path, e)
            continue

        # Get top-level definitions only
        top_level = get_top_level_definitions(defs)
        if not top_level:
            continue

        # Chunk with cAST
        chunks = chunk_file(
            full_path,
            text,
            top_level,
            budget=settings.index.chunk_budget,
        )
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} chunks")

    if not all_chunks:
        logger.error("No chunks generated!")
        return

    # Embed chunks
    logger.info("Embedding chunks with vLLM...")
    vllm_client = VLLMClient(settings.vllm)

    # Extract preview text for embedding
    chunk_texts = [c.text[:1000] for c in all_chunks]  # First 1000 chars

    embeddings = vllm_client.embed_chunks(
        chunk_texts,
        batch_size=settings.vllm.batch_size,
    )
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Write to Parquet
    logger.info("Writing to Parquet...")
    output_path = Path(settings.paths.vectors_dir) / "part-000.parquet"
    write_chunks_parquet(
        output_path,
        all_chunks,
        embeddings,
        start_id=0,
        vec_dim=settings.index.vec_dim,
    )
    logger.info("Wrote Parquet to %s", output_path)

    # Build FAISS index
    logger.info("Building FAISS index...")
    faiss_mgr = FAISSManager(
        index_path=Path(settings.paths.faiss_index),
        vec_dim=settings.index.vec_dim,
        nlist=settings.index.faiss_nlist,
        use_cuvs=settings.index.use_cuvs,
    )

    # Train and build index
    faiss_mgr.build_index(embeddings[:10000])  # Train on subset

    # Add all vectors
    ids = np.arange(len(all_chunks), dtype=np.int64)
    faiss_mgr.add_vectors(embeddings, ids)

    # Save CPU index
    faiss_mgr.save_cpu_index()
    logger.info(f"Saved FAISS index to {settings.paths.faiss_index}")

    # Initialize DuckDB catalog
    logger.info("Initializing DuckDB catalog...")
    with DuckDBCatalog(
        Path(settings.paths.duckdb_path),
        Path(settings.paths.vectors_dir),
    ) as catalog:
        count = catalog.count_chunks()
        logger.info("DuckDB catalog initialized (%s chunks)", count)

    logger.info("Indexing complete!")
    logger.info(f"  Chunks: {len(all_chunks)}")
    logger.info(f"  Embeddings: {len(embeddings)}")
    logger.info(f"  FAISS index: {settings.paths.faiss_index}")
    logger.info(f"  Parquet: {settings.paths.vectors_dir}")


if __name__ == "__main__":
    main()
