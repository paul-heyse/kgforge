# 0001 — Record architecture decisions

Date: 2025-10-25

Status: accepted

Context: Single-machine, high-performance hybrid search system requiring DocTags, Qwen3 embeddings, SPLADE, FAISS GPU/cuVS, and DuckDB as registry.

Decision: We adopt a hexagonal architecture with strict interfaces and Parquet-only vector persistence.

Consequences: Independent workstreams can ship without coordination friction; testing and reproducibility are improved.
