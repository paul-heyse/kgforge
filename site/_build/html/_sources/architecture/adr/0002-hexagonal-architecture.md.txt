# 0002 – Adopt a Hexagonal Architecture

Date: 2025-10-25

## Status
Accepted

## Context
We are delivering a single-machine, high-performance hybrid search system that stitches together DocTags, Qwen3 embeddings, SPLADE, FAISS GPU/cuVS, and DuckDB as the registry. We must keep interfaces stable while enabling rapid iteration.

## Decision
Adopt a hexagonal architecture with strict inbound/outbound ports. Persist vectors in Parquet to keep storage self-describing and tooling-friendly.

## Consequences
- Independent workstreams can progress without coordination friction.
- Ports/adapters isolate infrastructure churn from core domain logic.
- Testing and reproducibility improve because IO remains deterministic.
