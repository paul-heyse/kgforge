
# kgfoundry — End-to-End Local RAG + Ontology Grounding (Skeleton)

This repository is the **skeleton** for the single-machine architecture you specified:
- Ubuntu 24.04 / RTX 5090 / AMD 9950X / 192 GB RAM
- Python 3.13.9, PyTorch 2.9 (CUDA 13.0)
- vLLM (pre-release supporting CUDA 13)
- DuckDB ≥ 1.4.1
- Docling VLM (Granite-Docling) → DocTags + Docling HybridChunker
- Dense embeddings: Qwen3-Embedding-4B (2560-d)
- Sparse: BM25 + SPLADE-v3 (GPU)
- FAISS (GPU) with cuVS backend
- Neo4j local graph
- Everything registered in DuckDB; all embeddings in Parquet

> **Note**: This is a scaffold. Many functions are stubs and marked TODO.
> It is structured for high cohesion & clean interfaces so multiple teams can implement independently.

## Onboarding (uv-first)

```bash
# one-time setup
uv python pin 3.13.9          # writes .python-version
uv venv                       # creates .venv/
uv sync                       # installs from uv.lock / pyproject
```

For GPU support (FAISS GPU, cuVS, PyTorch, vLLM, etc.):
```bash
uv sync --extra gpu            # installs GPU dependencies
```

- `direnv` automatically activates `.venv` and replays `uv sync` with any extras you've synced.
  GPU installs drop a marker at `.venv/.uv-extras/gpu`; see `.envrc`.
- Pre-commit hooks install automatically the first time direnv loads the project.
- VS Code terminals default to `.venv`; see `.vscode/settings.json`.
- Secrets live in `.env`; add overrides in `.envrc.local` (gitignored).

## Directory layout

- `src/kgfoundry_common`: contracts (Pydantic), IDs, config utils, exceptions, logging.
- `src/download`: PyAlex-based harvester + OA PDF downloader with fallbacks.
- `src/docling`: VLM client + HybridChunker wrapper.
- `src/embeddings_dense`: Qwen3-Embedding-4B client (OpenAI-style vLLM).
- `src/embeddings_sparse`: SPLADE-v3 GPU encoder + BM25 (Pyserini) skeletons.
- `src/vectorstore_faiss`: FAISS GPU/cuVS adapter.
- `src/ontology`: loaders for OWL/OBO/SKOS + normalizer.
- `src/linking`: linker pipelines + calibration placeholders.
- `src/kg_builder`: Neo4j adapter.
- `src/search_api`: FastAPI app with /search, /graph/concepts, /healthz.
- `src/orchestration`: Prefect 2.x flows (idempotent, local-only).
- `src/observability`: Prometheus metrics + OpenTelemetry tracing priming.
- `registry/migrations`: DuckDB DDL; incremental migrations.
- `config`: example YAML + Nginx and Systemd templates.
- `tests`: unit and e2e skeletons.

## Python & Tooling

- `pyproject.toml`: Python 3.13.9; dependencies pinned minimally; heavy libs listed but commented or optional extras.
- **Installation**: Base install includes core dependencies. GPU features (FAISS GPU, cuVS, PyTorch, vLLM) require `uv sync --extra gpu`.
- **Extras**: Set `KGFOUNDRY_UV_EXTRAS=gpu` (comma-separated) or remove `.venv/.uv-extras/gpu`
  / export `KGFOUNDRY_DISABLE_GPU_EXTRA=1` to opt-out after a CUDA install.
- **Namespace consolidation**: All public APIs are available under the `kgfoundry.*` namespace (e.g., `from kgfoundry import vectorstore_faiss`). The namespace proxy automatically resolves submodules on first access.
- Pre-commit with ruff/black/mypy; strict typing by default.
- **Running tests**: Ensure the repository is importable before invoking pytest. Either install the package in editable mode (`uv pip install -e .`) or prefix commands with `PYTHONPATH=src` (for example, `PYTHONPATH=src uv run pytest -q`).
- Package README generation: `python tools/gen_readmes.py` (see
  [Automated Documentation](README-AUTOMATED-DOCUMENTATION.md#readme-generation-toolsgen_readmespy)
  for full workflow details).
- Gallery examples live in `examples/`; follow
  [`docs/how-to/contributing-gallery-examples.md`](docs/how-to/contributing-gallery-examples.md)
  and run `python tools/validate_gallery.py --strict` before building docs.

---

**This skeleton intentionally does not execute heavy GPU code** — the goal is to provide interfaces, contracts,
and minimal shims so teams can implement their parts with confidence.


<!-- merged from kgfoundry_skeleton (1).zip -->
# kgfoundry (skeleton)
Single-machine architecture for ontology-grounded hybrid search (dense + sparse + KG) with local registries and indices.
**Target host:** Ubuntu 24.04 · RTX 5090 (CUDA 13.0) · AMD 9950X (16 cores) · 192 GB RAM
**Runtime:** Python 3.13.9, PyTorch 2.9 (CUDA 13), vLLM pre-release (CUDA 13), DuckDB ≥ 1.4.1
This repository is a **skeleton** generated on 2025-10-25. It contains packages, interfaces, configs, and stubs to accelerate
independent development of each workstream.

<!-- merged from kgfoundry_skeleton (2).zip -->
This repository is the **skeleton** for the single-machine architecture:
- Ubuntu 24.04, RTX 5090, AMD 9950X (16c), 192 GB RAM
- Python 3.13.9, PyTorch 2.9 (CUDA 13)
- vLLM prerelease (CUDA 13): Granite-Docling VLM + Qwen3-Embedding-4B (2560-d)
- Docling HybridChunker for chunking
- Dense: Qwen3-Embedding-4B (2560)
- FAISS GPU (cuVS) for vector indexing
- Neo4j local KG
- DuckDB ≥ 1.4.1 registry
- **All embeddings stored as Parquet** (no JSONL)
## Quickstart
make bootstrap   # venv, dev deps, DuckDB migrations
make run         # starts FastAPI app (skeleton) on :8080
make e2e         # runs skeleton end-to-end tests
## Mock OA discovery (OpenAlex / Unpaywall) & fixture pipeline
Run local mock servers (one shell):
python -m tests.mock_servers.run_all
# OpenAlex mock:  http://localhost:8998/works
# Unpaywall mock: http://localhost:8997/v2/{doi}
# PDF host:       http://localhost:8999/pdf/...
In another shell, use the mock-aware harvester:
kgf harvest-mock --topic "test" --max-works 2 --out-dir /data/pdfs --db-path /data/catalog/catalog.duckdb
Generate tiny fixture Parquet datasets (chunks, dense 2560-d, SPLADE) and register them:
kgf fixture --root /data --db-path /data/catalog/catalog.duckdb
You can then iterate on individual components using these fixed contracts.

<!-- merged from kgfoundry_skeleton (4).zip -->
# kgfoundry (skeleton with fusion & KG mock)
This minimal skeleton exposes:
- **FastAPI** search service with **BM25 + SPLADE + FAISS** fusion (RRF) and a tiny **KG mock** boost
- **CLI** commands to build BM25 (pure) and FAISS indexes for fixture datasets
## Run the API
uvicorn kgfoundry.search_api.app:app --port 8080 --reload
Set environment variables for local data if you have them:
export KGF_FIXTURE_ROOT=/data
export KGF_FIXTURE_DB=/data/catalog/catalog.duckdb
export KGF_FAISS_CPU_INDEX=/data/faiss/fixture.index
export SEARCH_API_KEYS=testkey
## Build indexes
# BM25 (pure Python pickle)
kgf index-bm25 --db-path /data/catalog/catalog.duckdb --out-dir /data/lucene/bm25
# FAISS CPU index
kgf index-faiss --db-path /data/catalog/catalog.duckdb --factory "OPQ64,IVF8192,PQ64" --cpu-index-path /data/faiss/fixture.index
## Health & search
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/search -H "Authorization: Bearer testkey" -H "Content-Type: application/json" -d '{"query":"LLM alignment","k":5}'
