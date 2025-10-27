# Implementation-Grade Architecture Overview

Below is the **fully elaborated, implementation‑grade architecture** for your end‑to‑end, single‑machine system. It updates **every topic** from the original plan and adds all **new constraints** you set:

* **OS/Host**: Ubuntu 24.04 (single box, no external servers).
* **GPU**: NVIDIA RTX 5090 (CUDA **13.0** toolchain).
* **CPU/RAM**: AMD 9950X (16 cores), 192 GB RAM.
* **Python**: **3.13**.
* **PyTorch**: **2.9** (CUDA 13 build).
* **vLLM**: latest **pre‑release** that supports CUDA 13.
* **DuckDB**: **≥ 1.4.1** (interpreting your “1.41” as **1.4.1**).
* **Dense embeddings**: **Qwen3‑Embedding‑4B** at **2560‑dimensional** output.
* **Sparse embeddings**: **BM25** + **SPLADE‑v3** (GPU).
* **PDF→DocTags & chunking**: **Docling VLM (Granite‑Docling)** + **Docling HybridChunker**.
* **All embeddings in Parquet** (no JSONL at rest).
* **Vector indexing & ops**: **FAISS GPU** with **cuVS** enabled.
* **Model serving**: a **single logical endpoint** fronting **two local vLLM processes** (Granite‑Docling VLM + Qwen3‑Embedding‑4B), routed by **Nginx** (necessary because one vLLM process ≙ one base model).
* **Registry**: all artifacts (PDFs, DocTags, chunks, embeddings, indices, ontologies, links) registered in local **DuckDB**.

---

## 0) Big picture (end‑to‑end, single‑box)

```
Topic → (PyAlex) Harvest & OA PDF Download (with fallbacks)
     → Docling VLM (Granite‑Docling) → DocTags
     → Docling HybridChunker → Chunk Parquet
     → Dense (Qwen3‑Embedding‑4B, 2560‑d via vLLM) → Parquet → FAISS (GPU, cuVS)
     → Sparse (SPLADE‑v3, GPU → Parquet → Lucene impact) + BM25 (Lucene)
     → Ontology ingest → Concept catalog + embeddings (dense 2560‑d + SPLADE)
     → Chunk–Concept linker → Assertions (Parquet) → KG (Neo4j local)
     → Hybrid search API (FAISS + BM25 + SPLADE + KG‑aware rerank)
     → Everything registered in DuckDB (≥1.4.1)
```

**Architectural style**: **Ports & Adapters (Hexagonal)** + **Domain contracts** (Pydantic v2) + **Plugin registry** (entry points) + **Immutable, content‑addressed artifacts**. All heavy compute runs **locally** (no remote compute/storage).

---

## 1) Design principles (concretized)

* **Interface‑first**: every subsystem exposes an ABC (Abstract Base Class). Impl classes are pluggable via entry‑points.
* **Encapsulation & cohesion**: one public façade per package; internals hidden. Each module owns a single concern.
* **Idempotency**: outputs keyed by **content hashes**; re‑runs never duplicate artifacts.
* **Determinism**: global `seed=42`, fixed training samples for FAISS, fixed chunking parameters.
* **All embeddings in Parquet**: columnar, compressed with **ZSTD=6**, **row_group=4096**; **no JSONL** persisted.
* **Observability from day one**: OpenTelemetry traces; structured logs with artifact IDs; Prometheus counters/histograms.
* **Local‑only**: no remote indices or vector DBs; FAISS, Pyserini, Neo4j, DuckDB all on the box.
* **Security (local)**: vLLM bound to localhost; Nginx fronts a single endpoint; API‑key auth for the search API.

---

## 2) Repository layout (monorepo; independently ownable packages)

```
/src
  /kgfoundry_common            # contracts, IDs, hashing, config, utils
  /download                  # PyAlex harvester + OA PDF downloader + fallbacks
  /docling                   # VLM convert-to-DocTags + HybridChunker wrappers
  /embeddings_dense          # vLLM client, Qwen3-Embedding-4B (2560-d)
  /embeddings_sparse         # SPLADE-v3 GPU encoder (Parquet) + BM25/SPLADE indices (Pyserini)
  /vectorstore_faiss         # FAISS GPU/cuVS index build/search adapters
  /ontology                  # OWL/OBO/SKOS loaders, normalization, concept embeddings
  /linking                   # candidate gen, scoring, calibration, assertions
  /kg_builder                # Neo4j adapter; nodes/edges upsert
  /search_api                # FastAPI; hybrid retrieval + KG-aware rerank; OpenAPI
  /orchestration             # Prefect flows & CLI commands (local)
  /registry                  # DuckDB schema, migrations, dataset registration
  /observability             # OTEL + Prometheus exporters
/tests
/config                      # YAML config(s), ngnix.conf, systemd units
/scripts                     # bootstrap & build scripts (CUDA 13, FAISS+cuVS, vLLM)
```

**Packaging**: `pyproject.toml` (Poetry or uv). **Entry‑points** register providers:

```toml
[project.entry-points.kgfoundry.plugins]
dense.qwen3 = "embeddings_dense.qwen3:Qwen3Embedder"
sparse.splade_v3 = "embeddings_sparse.splade:SPLADEv3Encoder"
sparse.bm25 = "embeddings_sparse.bm25:BM25Index"
docling.vlm = "docling.vlm:GraniteDoclingVLM"
chunker.docling_hybrid = "docling.hybrid:HybridChunker"
vector.faiss_gpu = "vectorstore_faiss.gpu:FaissGpuIndex"
graph.neo4j = "kg_builder.neo4j:Neo4jStore"
ontology.loader = "ontology.loader:OntologyLoader"
```

---

## 3) Core data contracts (Pydantic v2; content‑addressed IDs)

> Keep models lean. Payloads (large text, vectors) live in Parquet files; rows refer to them via IDs.

```python
# kgfoundry_common/models.py (Python 3.13, Pydantic v2)
from pydantic import BaseModel, Field, AwareDatetime
from typing import Optional, List, Dict, Literal
from dataclasses import dataclass

Id = str  # URN-like opaque IDs, stable and content-addressed

class Doc(BaseModel):
    id: Id                           # urn:doc:sha256:<16b>
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmcid: Optional[str] = None
    title: str
    authors: List[str] = []
    pub_date: Optional[str] = None   # ISO8601
    license: Optional[str] = None
    language: Optional[str] = "en"
    pdf_uri: str                     # local path (e.g., /data/pdfs/<id>.pdf)
    source: str                      # 'openalex', 'arxiv', 'pmc', ...
    content_hash: str                # sha256 of canonical text (after DocTags->text)
    created_at: AwareDatetime

class DoctagsAsset(BaseModel):
    doc_id: Id
    doctags_uri: str                 # /data/doctags/<doc_id>.dt.json.zst
    pages: int
    vlm_model: str                   # granite-docling-258M
    vlm_revision: str                # 'untied' if used
    avg_logprob: Optional[float] = None
    created_at: AwareDatetime

class Chunk(BaseModel):
    id: Id                           # urn:chunk:<doc_hash>:<start>-<end>
    doc_id: Id
    section: Optional[str]
    start_char: int
    end_char: int
    tokens: int
    doctags_span: Dict[str, int]     # {node_id, start, end}
    created_at: AwareDatetime
    dataset_id: str                  # backpointer to Parquet dataset

class DenseVectorMeta(BaseModel):
    chunk_id: Id
    model: str                       # 'Qwen3-Embedding-4B'
    run_id: str
    dim: int                         # 2560
    l2_norm: float
    created_at: AwareDatetime

class SparseVectorMeta(BaseModel):
    chunk_id: Id
    model: str                       # 'SPLADE-v3-distilbert'
    run_id: str
    nnz: int
    created_at: AwareDatetime

class Concept(BaseModel):
    id: Id                           # urn:concept:<ontology>:<curie>
    ontology: str
    pref_label: str
    alt_labels: List[str] = []
    definition: Optional[str] = None
    parents: List[Id] = []
    meta: Dict[str, str] = {}

class LinkAssertion(BaseModel):
    id: Id                           # urn:assert:<chunk_id>:<concept_id>@<run_id>
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal['link','reject','uncertain']
    evidence_span: Optional[str] = None
    features: Dict[str, float] = {}  # dense_sim, sparse_sim, lexical_overlap, depth_bonus
    run_id: str
    created_at: AwareDatetime
```

**ID scheme (deterministic)**

* `doc_id = urn:doc:sha256:<first16 bytes of sha256 canonical_text, base32>`
* `chunk_id = urn:chunk:<doc_hash>:<start>-<end>`
* `dense vec id = urn:vec:<chunk_id>:qwen3@<run_id>`
* `sparse id = urn:sparse:<chunk_id>:splade_v3@<run_id>`
* `concept id = urn:concept:<ontology>:<curie>`
* `assertion id = urn:assert:<chunk_id>:<concept_id>@<run_id>`

---

## 4) Storage layers & adapters (ports)

**VectorStore (FAISS GPU + cuVS)**

```python
class VectorStore(Protocol):
    def train(self, train_vectors: "np.ndarray", **params) -> None: ...
    def add(self, keys: list[str], vectors: "np.ndarray") -> None: ...
    def search(self, query: "np.ndarray", k: int) -> list[tuple[str, float]]: ...
    def save(self, index_uri: str, idmap_uri: str) -> None: ...
    def load(self, index_uri: str, idmap_uri: str) -> None: ...
```

**SparseIndex**

```python
class SparseIndex(Protocol):
    def build(self, docs_iterable: "Iterable[SparseDoc]") -> None: ...
    def search(self, query: str, k: int, fields: dict|None=None) -> list[tuple[str, float]]: ...
    def stats(self) -> dict: ...
```

Implementations: **BM25Index** (Pyserini) and **SpladeImpactIndex** (Pyserini).

**GraphStore**

```python
class GraphStore(Protocol):
    def upsert_nodes(self, docs: list[Doc], concepts: list[Concept], chunks: list[Chunk]) -> None: ...
    def upsert_mentions(self, assertions: list[LinkAssertion]) -> None: ...
    def neighbors(self, concept_id: str, depth:int=1) -> list[str]: ...
    def linked_concepts(self, chunk_id: str) -> list[str]: ...
```

**Registry (DuckDB ≥1.4.1)**

* Provides **DDL/migrations**, **safe registration** (two‑phase commit: write Parquet → atomically register).
* Exposes **views** across Parquet datasets (union_by_name).
* Threading: default **PRAGMA threads=14**.

---

## 5) Orchestration (Prefect 2.x; all local)

**Flows** (idempotent, content‑hash keyed):

1. `harvest_and_download(topic, years, max_works)`
2. `convert_to_doctags(doc_ids[])`
3. `chunk_with_docling(doc_ids[])`
4. `embed_dense_qwen3(chunk_dataset_id)`
5. `encode_splade_v3(chunk_dataset_id)`
6. `build_bm25_index(chunk_dataset_id)`
7. `build_faiss_index(dense_run_id)`
8. `ingest_ontologies(ontology_specs[])`
9. `embed_concepts(ontology_id)`
10. `link_chunks_to_concepts(chunk_dataset_id, ontology_id)`
11. `upsert_kg(link_run_id)`
12. `serve_search_api()` (starts uvicorn if not already)

**Events** recorded in `registry.pipeline_events`: `DocumentIngested`, `DoctagsReady`, `ChunksCreated`, `DenseEmbedded`, `SpladeEncoded`, `BM25Built`, `FAISSBuilt`, `OntologyLoaded`, `ConceptEmbeddingsReady`, `LinkerRun`, `KGUpdated`.

**Concurrency defaults**:

* Downloader 8 parallel fetches;
* VLM 2 docs in flight;
* Dense/SPLADE batchers tuned to ~80% VRAM;
* FAISS build: 2 shards concurrently;
* All others CPU‑bound threads = 14.

---

## 6) Harvesters & PDF download (PyAlex first; resilient fallbacks)

**Workflow**

* **Search**: PyAlex (OpenAlex) by `topic` across title/abstract/fulltext; filter OA flags; optionally by year range.
* Extract candidate OA locations: `best_oa_location`, `primary_location`, `locations[]`, DOI, arXiv id, pmcid.
* **Download resolution** (in order):

  1. OpenAlex `best_oa_location.pdf_url`.
  2. Any `locations[].pdf_url` with `is_oa=true` favoring `publishedVersion` or `acceptedVersion`.
  3. **Unpaywall** by DOI (`url_for_pdf`) — requires configured contact email and rate limits.
  4. **Source‑specific**:

     * arXiv → `https://arxiv.org/pdf/<id>.pdf`
     * PubMed Central → `https://www.ncbi.nlm.nih.gov/pmc/articles/<pmcid>/pdf`
* **License guard**: accept only OA‑compatible licenses; persist license string on `Doc`.
* **Download**: 8 concurrency; 60s timeout; 3 retries w/ exp backoff; `User-Agent` includes contact/email.
* **Storage**: `/data/pdfs/<doc_id>.pdf` where `doc_id` derived later from canonical text (post‑DocTags). Temporarily name by OpenAlex ID then rename after canonicalization.
* **Registration**: each successful PDF produces a `documents` row (openalex id, doi, license, pdf_uri, source).

---

## 7) PDF → **DocTags** (Docling VLM Granite‑Docling)

* **Serving**: vLLM pre‑release (CUDA 13) process **#1** on localhost:8001.
* **Router**: Nginx exposes `/vlm/* → 8001`.
* **Defaults**: DPI=220, page_batch=8, bf16 if supported, fallback fp16.
* **Quality gate**: if avg token logprob < 0.5 on a page → fallback OCR for that page; provenance notes retained.
* **Timeouts**: 120s per 100 pages; max 2000 pages.
* **Outputs**: `/data/doctags/<doc_id>.dt.json.zst`.
* **Registration**: DuckDB `doctags` with pages, model, revision (`untied` if applicable), avg_logprob.

> We intentionally present one **logical** model endpoint via Nginx, but use **two** local vLLM processes (one model each) due to vLLM’s one‑model‑per‑process design.

---

## 8) Chunking (Docling **HybridChunker**)

* **Parameters**: `target_tokens=400`, `overlap=80`, `min_tokens=120`, `max_tokens=480`.
* Structure‑aware segmentation first; spillover via sliding windows.
* Captures `doctags_span` and `start_char/end_char` for explainable highlights.
* **Parquet dataset**: `parquet/chunks/model=docling_hybrid/run=<run_id>/part-*.parquet`
  **Schema**:

  ```
  chunk_id: string
  doc_id: string
  section: string
  start_char: int32
  end_char: int32
  doctags_span: struct<node_id:string,start:int32,end:int32>
  text: string
  tokens: int32
  created_at: timestamp
  ```
* Register resulting dataset in `registry.datasets` (kind='chunks') and each row in `chunks`.

---

## 9) Dense embeddings (Qwen3‑Embedding‑4B @ **2560‑dim**, vLLM)

* **Serving**: vLLM pre‑release process **#2** on localhost:8002; router maps `/v1/embeddings → 8002`.
* **Embedding length**: **2560** (model supports 32–2560; set explicitly).
* **Batching**: automatic; target **≤ 80%** of available VRAM.
* **Normalization**: L2; store norm (float32).
* **Parquet dataset**: `parquet/dense/model=Qwen3-Embedding-4B/run=<run_id>/part-*.parquet`
  **Schema**:

  ```
  chunk_id: string
  model: string              -- 'Qwen3-Embedding-4B'
  run_id: string
  dim: int16                 -- 2560
  vector: list<float>        -- length==2560 (float32 on disk)
  l2_norm: float
  created_at: timestamp
  ```
* **Compression**: ZSTD=6; row_group=4096; dictionary encoding on categorical cols.

---

## 10) Sparse embeddings & indices (SPLADE‑v3 **GPU** + BM25)

* **SPLADE‑v3 encoder**: `naver/splade-v3-distilbert`

  * CUDA (Torch 2.9, cu130), AMP fp16, `max_seq_len=512`, `topK=256` nnz per chunk.
  * Tokenizer: distilbert WordPiece, `vocab_size=30522`.
  * **Parquet dataset**: `parquet/sparse/model=SPLADE-v3-distilbert/run=<run_id>/part-*.parquet`
    **Schema**:

    ```
    chunk_id: string
    model: string
    run_id: string
    vocab_ids: list<int32>    -- sorted, unique
    weights:   list<float>    -- same length as vocab_ids
    nnz: int16
    created_at: timestamp
    ```
* **Indexing**: **Lucene impact index** (Pyserini).

  * **No JSONL at rest**: a streaming adapter reads Parquet rows and writes directly to Lucene via Pyserini builders (any temporary files are ephemeral and excluded from registry).
* **BM25**: Pyserini (Lucene). Params: **k1=0.9**, **b=0.4**; field boosts `title^2.0, section^1.2, body^1.0`.

---

## 11) FAISS (GPU with **cuVS**), CUDA 13

* **Install**: Prefer **GPU binary** (wheel) built for CUDA 13 **with cuVS enabled**. If unavailable, build from source with `-DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_CUVS=ON`.
* **Index factory (default)**: `OPQ64,IVF8192,PQ64` (good for **d=2560**; OPQ pre‑rotation at 64; PQ m=64).
* **Training**: sample **10M** vectors or all (if fewer), seed=42.
* **Search**: `nprobe=64`. Codes 8‑bit per subvector.
* **Memory**: PQ codes ~**64 B/vector** (+ID/overhead ⇒ ~**80–100 B/vector** typical).
* **Persistence**: `.faiss` index + `.ids` idmap; shards ≤10M vectors each.
* **Registration**: rows in DuckDB `faiss_indexes` (logical index id groups shards).

---

## 12) Ontology ingestion & concept encoders

* **Formats**: OWL/RDF (TTL/RDF‑XML), OBO, SKOS.
* **Normalization**: lowercase, NFC, strip most punctuation, lemmatize EN, stopword‑smart; build surface forms from `pref_label`, `alt_labels`, `synonyms`, and `definition`.
* **Dense concept embeddings**: Qwen3‑Embedding‑4B with **dim=2560** (concat text: `pref_label | definition | 5 best synonyms`).
* **Sparse concept embeddings**: SPLADE‑v3 with **topK=128**.
* **Parquet datasets** mirroring chunk schemas; registered in `concept_embeddings`.

---

## 13) Linker (chunk→concept), calibrated

**Candidate generation**

* SPLADE concept index **@100** using chunk text;
* Lexical dictionary match **@50** (max phrase len=6 tokens, case‑folded).

**Feature scoring**

* `dense_sim` = cosine(qwen_chunk, qwen_concept)
* `sparse_sim` = normalized SPLADE (min–max per query)
* `lexical_overlap` = matched_chars / chunk_chars (clamp [0, 0.2])
* `depth_bonus` = +0.02 × levels from root (cap +0.10)

**Fusion & decision**

* `score = 0.55*dense + 0.35*sparse + 0.10*lexical + depth_bonus`
* Thresholds: **link ≥ 0.62**, **reject ≤ 0.35**, else uncertain.
* **Calibration**: isotonic regression stored per linker run; output `LinkAssertion` Parquet.

---

## 14) Knowledge Graph (Neo4j, local)

**Nodes**

* `(:Doc {doc_id, title, year, license, source, source_id})`
* `(:Chunk {chunk_id, section, start_char, end_char})`
* `(:Concept {concept_id, ontology, pref_label})`

**Relationships**

* `(:Doc)-[:HAS_CHUNK]->(:Chunk)`
* `(:Chunk)-[:MENTIONS {score, run_id, evidence_span, created_at}]->(:Concept)`
* `(:Concept)-[:IS_A]->(:Concept)` and `[:RELATED_TO]` (from ontology).

**Constraints & indexes**

* Uniqueness on `doc_id`, `chunk_id`, `concept_id`; B‑tree index on `MENTIONS.score` and `MENTIONS.run_id`.

---

## 15) Hybrid search API (FastAPI, local)

**Endpoints**

* `POST /search`
  Body: `{query: str, k?: int=10, filters?: {...}, explain?: bool=false}`
  Returns top chunks & doc rollups with dense/sparse scores, KG boosts, spans, linked concepts.
* `POST /graph/concepts` (browse ontology)
* `GET /healthz`

**Algorithm (fixed)**

1. Query embedding via Qwen3 (2560‑d) + parse lexical concept mentions.
2. Retrieve **dense@200** (FAISS) + **sparse@200** (BM25 and SPLADE).
3. Fuse via **RRF(k=60)**.
4. KG boosts: +0.08 for direct concept match; +0.04 for one‑hop neighbor.
5. **MMR** (λ=0.7) at doc level.
6. Return top‑k.

---

## 16) Configuration (YAML; single‑box defaults)

```yaml
system:
  os: ubuntu-24.04
  threads: 14
  seed: 42
  parquet_root: /data/parquet
  artifacts_root: /data/artifacts
  duckdb_path: /data/catalog/catalog.duckdb

runtime:
  python: "3.13"
  cuda: "13.0"
  torch: "2.9"
  duckdb_min: "1.4.1"
  vllm_channel: "pre-release-cuda13"

network:
  nginx_port: 80
  vlm_port: 8001
  emb_port: 8002
  api_port: 8080

harvest:
  provider: pyalex
  per_page: 200
  max_works: 20000
  years: ">=2018"
  filters:
    is_oa: true
    has_oa_published_version: true
  fallbacks:
    unpaywall: true
    arxiv: true
    pmc: true
  concurrency: 8
  timeout_sec: 60
  retries: 3

doc_conversion:
  vlm_model: ibm-granite/granite-docling-258M
  vlm_revision: untied
  endpoint: http://localhost/vlm/
  dpi: 220
  page_batch: 8
  ocr_fallback: true
  max_pages: 2000
  timeout_sec: 120

chunking:
  engine: docling_hybrid
  target_tokens: 400
  overlap_tokens: 80
  min_tokens: 120
  max_tokens: 480

dense_embedding:
  model: Qwen/Qwen3-Embedding-4B
  endpoint: http://localhost/v1/embeddings
  output_dim: 2560
  parquet_out: ${system.parquet_root}/dense/model=Qwen3-Embedding-4B/run=${run_id}

sparse_embedding:
  splade:
    model: naver/splade-v3-distilbert
    device: cuda
    amp: fp16
    max_seq_len: 512
    topk: 256
    parquet_out: ${system.parquet_root}/sparse/model=SPLADE-v3-distilbert/run=${run_id}
  bm25:
    k1: 0.9
    b: 0.4
    field_boosts: { title: 2.0, section: 1.2, body: 1.0 }
    index_dir: /data/lucene/bm25

faiss:
  index_factory: OPQ64,IVF8192,PQ64
  nprobe: 64
  train_samples: 10000000
  shards:
    max_vectors_per_shard: 10000000
  gpu: true
  cuvs: true
  output_dir: /data/faiss/qwen3_ivfpq

ontology:
  inputs:
    - { ontology_id: mesh, format: obo, uri: /data/ontologies/mesh.obo }
    - { ontology_id: go,   format: obo, uri: /data/ontologies/go.obo }
  concept_embed:
    dense_model: Qwen3-Embedding-4B
    dense_dim: 2560
    splade_model: SPLADE-v3-distilbert
    splade_topk: 128

linker:
  candidates: { splade_topk: 100, lexicon_topk: 50 }
  fusion_weights: { dense: 0.55, sparse: 0.35, lexical: 0.10, depth_bonus_per_level: 0.02, depth_cap: 0.10 }
  thresholds: { high: 0.62, low: 0.35 }
  calibration: isotonic

graph:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password_env: NEO4J_PASSWORD

search:
  k: 10
  dense_candidates: 200
  sparse_candidates: 200
  rrf_k: 60
  mmr_lambda: 0.7
  kg_boosts: { direct: 0.08, one_hop: 0.04 }
```

---

## 17) DuckDB (≥ 1.4.1) schema & views (full fidelity)

**Tables (key ones)**

```sql
CREATE TABLE model_registry (
  model_id TEXT,
  repo TEXT,
  revision TEXT,
  tokenizer TEXT,
  embedding_dim INT,      -- 2560 for Qwen3 embeddings
  vocab_size INT,         -- 30522 for SPLADE v3 tokenizer
  framework TEXT,         -- 'vllm'|'hf'
  framework_version TEXT,
  build_info JSON,        -- FAISS flags, CUDA, cuVS info
  PRIMARY KEY (model_id, revision)
);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  purpose TEXT,           -- 'dense_embed'|'splade_encode'|'bm25_build'|'faiss_build'|...
  model_id TEXT,
  revision TEXT,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  config JSON
);

CREATE TABLE documents (
  doc_id TEXT PRIMARY KEY,
  openalex_id TEXT, doi TEXT, arxiv_id TEXT, pmcid TEXT,
  title TEXT, authors JSON, pub_date TIMESTAMP,
  license TEXT, language TEXT,
  pdf_uri TEXT, source TEXT,
  content_hash TEXT,         -- canonical text hash (after DocTags→text)
  created_at TIMESTAMP
);

CREATE TABLE doctags (
  doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
  doctags_uri TEXT, pages INT,
  vlm_model TEXT, vlm_revision TEXT,
  avg_logprob DOUBLE,
  created_at TIMESTAMP
);

CREATE TABLE datasets (
  dataset_id TEXT PRIMARY KEY,
  kind TEXT,                 -- 'chunks'|'dense'|'sparse'|'concepts'
  parquet_root TEXT,
  run_id TEXT REFERENCES runs(run_id),
  created_at TIMESTAMP
);

CREATE TABLE chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT REFERENCES documents(doc_id),
  section TEXT, start_char INT, end_char INT,
  doctags_span JSON, tokens INT,
  dataset_id TEXT REFERENCES datasets(dataset_id),
  created_at TIMESTAMP
);

CREATE TABLE dense_runs (
  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
  model TEXT, dim INT,       -- 2560
  parquet_root TEXT,
  created_at TIMESTAMP
);

CREATE TABLE sparse_runs (
  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
  model TEXT, vocab_size INT,
  parquet_root TEXT,
  created_at TIMESTAMP,
  backend TEXT               -- 'lucene-impact'|'lucene-bm25'
);

CREATE TABLE faiss_indexes (
  logical_index_id TEXT,     -- groups shards into a single logical index
  run_id TEXT REFERENCES dense_runs(run_id),
  shard_id INT,
  index_type TEXT, nlist INT, m INT, opq INT, nprobe INT,
  gpu BOOLEAN, cuvs BOOLEAN,
  index_uri TEXT, idmap_uri TEXT,
  created_at TIMESTAMP,
  PRIMARY KEY (logical_index_id, shard_id)
);

CREATE TABLE ontologies (
  ontology_id TEXT PRIMARY KEY,
  format TEXT, src_uri TEXT,
  loaded_at TIMESTAMP, concept_count INT
);

CREATE TABLE concept_embeddings (
  ontology_id TEXT REFERENCES ontologies(ontology_id),
  model TEXT, dim INT,
  parquet_root TEXT,
  created_at TIMESTAMP,
  PRIMARY KEY (ontology_id, model)
);

CREATE TABLE link_assertions (
  id TEXT PRIMARY KEY,
  chunk_id TEXT REFERENCES chunks(chunk_id),
  concept_id TEXT,
  score DOUBLE, decision TEXT,
  evidence_span TEXT,
  features JSON,
  run_id TEXT REFERENCES runs(run_id),
  created_at TIMESTAMP
);

CREATE TABLE pipeline_events (
  event_id TEXT PRIMARY KEY,
  event_name TEXT, subject_id TEXT, payload JSON,
  created_at TIMESTAMP
);

-- Helpful indexes
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_link_chunk ON link_assertions(chunk_id);
CREATE INDEX idx_link_concept ON link_assertions(concept_id);
```

**Views** (Parquet unification)

```sql
PRAGMA threads=14;

CREATE OR REPLACE VIEW dense_vectors_view AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM dense_runs), union_by_name=true);

CREATE OR REPLACE VIEW splade_vectors_view AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM sparse_runs WHERE backend='lucene-impact'), union_by_name=true);

CREATE OR REPLACE VIEW chunk_texts AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM datasets WHERE kind='chunks'), union_by_name=true);
```

---

## 18) vLLM & Nginx (single logical endpoint)

**Nginx** (local):

```nginx
server {
  listen 80;
  server_name localhost;

  location /vlm/ {              # Docling VLM
    proxy_pass http://127.0.0.1:8001/;
  }
  location /v1/embeddings {     # Qwen3 embeddings (OpenAI-compatible)
    proxy_pass http://127.0.0.1:8002/v1/embeddings;
  }
  location /healthz {
    return 200 'ok\n';
  }
}
```

**vLLM processes (systemd units suggested)**

* **Granite‑Docling**: `vllm serve ibm-granite/granite-docling-258M --revision untied --host 127.0.0.1 --port 8001 --dtype bfloat16`
* **Qwen3‑Embedding**: `vllm serve Qwen/Qwen3-Embedding-4B --host 127.0.0.1 --port 8002 --dtype float16 --max-num-seqs 1024 --trust-remote-code`

*(One model per vLLM process; router gives you a single host/endpoint.)*

---

## 19) Implementation skeletons (selected)

### 19.1 Downloader (PyAlex + fallbacks)

```python
# download/harvester.py
class OpenAccessHarvester:
    def __init__(self, cfg, http, unpaywall_client):
        ...

    def search_openalex(self, topic: str, years: str, max_works: int) -> list[dict]:
        # paginate PyAlex queries; collect candidate OA PDF URLs + metadata
        ...

    def resolve_pdf(self, work: dict) -> str | None:
        # try best_oa_location.pdf_url → locations[].pdf_url → unpaywall → arxiv/pmc
        ...

    def download_pdf(self, url: str, dest_path: str) -> bool:
        # 60s timeout, 3 retries, 8 concurrent workers
        ...

    def run(self, topic: str, years: str, max_works: int) -> list[Doc]:
        # returns registered documents (DuckDB insert)
        ...
```

### 19.2 DocTags conversion

```python
# docling/vlm.py
class GraniteDoclingVLM:
    def __init__(self, endpoint: str, dpi: int, page_batch: int, timeout: int, ocr_fallback: bool):
        ...

    def to_doctags(self, pdf_path: str) -> dict:
        # call vLLM endpoint; assemble DocTags JSON; OCR fallback per page if needed
        ...

    def persist(self, doc_id: str, doctags: dict) -> str:
        # write to /data/doctags/<doc_id>.dt.json.zst, return URI
```

### 19.3 Chunking

```python
# docling/hybrid.py
class HybridChunker:
    def __init__(self, target: int=400, overlap: int=80, min_tokens: int=120, max_tokens: int=480):
        ...
    def chunk(self, doctags_uri: str) -> list[Chunk]:
        # parse, section-aware splitting, sliding windows, record spans/offsets
        ...
```

### 19.4 Dense embeddings (Qwen3 2560‑d via vLLM)

```python
# embeddings_dense/qwen3.py
class Qwen3Embedder:
    name = "Qwen3-Embedding-4B"
    dim = 2560
    def __init__(self, endpoint: str):
        ...
    def embed_texts(self, texts: list[str]) -> "np.ndarray":
        # call /v1/embeddings; enforce dim=2560; L2-normalize
        ...
```

### 19.5 SPLADE‑v3 encoder (GPU)

```python
# embeddings_sparse/splade.py
class SPLADEv3Encoder:
    def __init__(self, model_id: str, device="cuda", topk=256, max_seq_len=512):
        ...
    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        # torch.no_grad + autocast; prune to topK
        ...
```

### 19.6 FAISS index (GPU + cuVS)

```python
# vectorstore_faiss/gpu.py
class FaissGpuIndex:
    def __init__(self, factory: str, nprobe: int, gpu: bool=True, cuvs: bool=True):
        ...
    def train(self, train_vectors: "np.ndarray") -> None:
        ...
    def add(self, keys: list[str], vectors: "np.ndarray") -> None:
        ...
    def search(self, query: "np.ndarray", k: int) -> list[tuple[str, float]]:
        ...
    def save(self, index_uri: str, idmap_uri: str) -> None:
        ...
```

### 19.7 Hybrid retrieval & rerank (search_api)

```python
# search_api/service.py
def hybrid_search(query: str, k: int) -> list[dict]:
    q_vec = dense_embedder.embed_texts([query])[0]
    dense_hits = faiss.search(q_vec, k=200)
    sparse_hits = bm25.search(query, k=200) + splade.search(query, k=200)
    fused = rrf_fuse(dense_hits, sparse_hits, k=60)
    boosted = apply_kg_boosts(fused, query)
    results = mmr_deduplicate(boosted, lambda_=0.7)
    return explain(results, k)
```

---

## 20) Quality gates, evaluation & CI

* **Retrieval**: nDCG@10 ≥ 0.50 (baseline), Recall@1K ≥ 0.85.
* **Linker**: F1 ≥ 0.70 on a labeled set; **ECE ≤ 0.08** after isotonic calibration.
* **Latency**: p95 search **< 300 ms** (top‑10); p50 chunk→embed throughput tracked.
* **CI**: Unit tests (mypy, ruff), golden tests, nightly 1k‑doc mini‑pipeline; fail on regressions.

---

## 21) Security, licensing, governance (local)

* **License guard** at harvest; store license string in `documents.license`.
* vLLM binds to **localhost** only; Nginx is the only front door; **API‑key** auth on search API.
* **Secrets** (if any) from environment; never persisted.
* **Provenance**: every artifact row includes `{run_id, model_id, revision, created_at}`.

---

## 22) Capacity & sizing with **2560‑dim**

* **Dense Parquet**: ≈ **10 KB/chunk** pre‑compression (2560 × 4 B).
* **IVF‑PQ** (PQ64) codes: ≈ **64 B/vector** (+ID/overhead ~80–100 B).
* **SPLADE postings**: ~**2 KB/chunk** before Lucene compression (256 nnz).
* Scale by chunk count (e.g., 5M chunks → ~50 GB dense Parquet uncompressed; index is small enough for GPU).

---

## 23) Operational runbooks

* **CUDA 13 + Torch 2.9**: install CUDA 13, confirm `nvcc --version`, install cu130 wheels.
* **vLLM pre‑release**: install channel with CUDA‑13 support; start two processes; verify health on `:8001`, `:8002`; Nginx routes.
* **FAISS GPU + cuVS**: install GPU binary (preferred); if needed, build from source with CMake flags enabling cuVS.
* **Directories**:

  * PDFs: `/data/pdfs`
  * DocTags: `/data/doctags`
  * Parquet: `/data/parquet`
  * Lucene: `/data/lucene/{bm25,splade}`
  * FAISS: `/data/faiss`
  * DuckDB: `/data/catalog/catalog.duckdb`
* **Resource limits**: `ulimit -n 65535`; I/O scheduler `mq-deadline`; set `PRAGMA threads=14` in DuckDB.

---

## 24) Workstreams (implementation plans & deliverables)

1. **Downloader & Harvester (PyAlex + fallbacks)**

   * PyAlex client; OA filters; DOI/ID extract; Unpaywall integration; robust downloader; license guard; DuckDB registration.
   * Deliverables: module, config, tests (mock HTTP), metrics (`pdf_download_success_total`, `oa_resolution_latency`).

2. **Docling VLM & DocTags**

   * vLLM integration; page batching; OCR fallback; DocTags writer; quality metrics (avg logprob/page).
   * Deliverables: module, Nginx+vLLM configs, DDL updates, golden DocTags samples.

3. **Chunking (Docling Hybrid)**

   * Hybrid configuration; token accounting; offsets/spans; Parquet writer; golden chunk fixtures.

4. **Dense embeddings (Qwen3 2560‑d)**

   * vLLM client; batching & normalization; Parquet writer; memory instrumentation; retries.

5. **SPLADE‑v3 GPU encoder + Pyserini indices**

   * Torch 2.9 encoder; Parquet encoder; streaming into Lucene (impact); BM25 build; query APIs.

6. **FAISS GPU/cuVS**

   * Binary installation or source build; index builder (OPQ64,IVF8192,PQ64); training sampler; shard manager; search adapter.

7. **Ontology & concept embeddings**

   * Loaders (OWL/OBO/SKOS); normalization; concept Parquet; Qwen3(2560) + SPLADE encoders.

8. **Linker**

   * Candidate gen; fusion; calibration (isotonic); assertions Parquet; explainability; ablations.

9. **KG Builder (Neo4j)**

   * Node/edge upserts; constraints; indexes; maintenance; graph query helpers.

10. **Search API**

    * FastAPI; hybrid retrieval; RRF; KG boosts; MMR; explanations; OpenAPI schema; auth.

11. **Registry (DuckDB)**

    * Migrations; registration helpers; views across Parquet; integrity checks; provenance guards.

12. **Orchestration & Observability**

    * Prefect flows; retries; dashboards; tracing; alerting; local startup scripts.

---

## 25) Coding standards & best‑in‑class Python practices

* **Static typing** everywhere (`mypy --strict`).
* **Pydantic v2** for all externalized contracts; validate at boundaries only.
* **Dataclasses/NamedTuples** for internal‑only small records on hot paths.
* **Pure functions** for transformations; side effects behind adapters.
* **Dependency inversion**: constructors accept interfaces, not concretes.
* **Factories** resolve plugins from entry‑points; no `if/elif` on provider names in business logic.
* **Error handling**: wrap external calls in `Result[T, E]`‑like helpers (or `try/except` mapping to domain errors).
* **Logging**: structured (`json`); include `run_id`, `doc_id`, `chunk_id`.
* **Testing**: unit + contract tests per port; “fake” in‑memory providers; golden tests for DocTags/chunks; end‑to‑end smoke.
* **Performance**: vector ops on GPU; streaming I/O; Parquet row groups sized for cache locality; avoid pandas on hot paths.
* **Docs**: every public class/method has docstrings; README per package; ADRs for significant choices.

---

# Addendum - additional architecture information

## PART A — GAP ANALYSIS (WHAT’S UNSPECIFIED) → RESOLUTIONS IN ADDENDUM

| Area                                      | Gap / ambiguity                                                                                                                     | Why it matters                                | Final specification (summary)                                                                                                                                                                             |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Canonical text & ID hashing**           | Exact *canonicalization* pipeline from DocTags to text (line breaks, whitespace, Unicode norms) not fixed; ID hashing step unclear. | Reproducibility; stable `doc_id`, `chunk_id`. | Define canonicalizer (NFC → collapse whitespace → normalize bullets/ligatures → Unix line endings). Hash **canonical text** SHA‑256, use first 16 bytes (base32) for URNs. (§B1)                          |
| **Tokenizer for chunk sizes**             | Which tokenizer governs `target_tokens` not pinned.                                                                                 | Chunk consistency, embedding costs.           | Use **Qwen3‑Embedding tokenizer** via HF `AutoTokenizer` for token counts. (§B2)                                                                                                                          |
| **DocTags fidelity**                      | Which DocTags nodes to include/exclude (headers/footers/refs), and OCR fallback merge rules unclear.                                | Content quality; linkability back to pages.   | Include body, headings, tables, captions, figure text; exclude references by default. OCR per‑page fallback with provenance flag; page order preserved. (§B3)                                             |
| **Downloader resolution order & de‑dupe** | Ties between multiple OA locations; DOI/arXiv/PMCID overlaps; file dedup not specified.                                             | Avoid duplicate processing and wasted space.  | Stable priority list (OpenAlex best_oa → other locations → Unpaywall → source‑specific), MIME check; compute `pdf_sha256`; deduplicate by hash; symlink duplicate DoIs. (§B4)                             |
| **PyAlex topic query spec**               | Which OpenAlex fields are searched; year filters and pagination; retries/timeouts not fixed.                                        | Recall and reliability.                       | Search `title, abstract_inverted_index, fulltext` with AND semantics; filter by `from_year..to_year`; `per_page=200`; backoff retries 3×; http timeout 30 s; polite `User-Agent` and mailto. (§B4)        |
| **Qwen3 embeddings 2560‑d**               | Request parameter to enforce 2560 output; failure behavior if unsupported.                                                          | Index shape; FAISS train.                     | Request `dimension=2560`; assert response dim; if unsupported (model mismatch), **fail fast** and record incident. (§B5)                                                                                  |
| **SPLADE‑v3 GPU encoder**                 | Pre‑processing (lowercasing, punctuation), truncation, batching, AMP, and top‑K pruning details not pinned.                         | Index size and quality; repeatability.        | Lowercase; strip control chars; keep punctuation; truncation at 512 WP tokens; AMP fp16; batch size auto‑tuned; top‑K=256 with stable tiebreaker (token id). (§B6)                                        |
| **BM25 analyzer**                         | Analyzer pipeline (tokenizer, stopwords, stemming, casefold) not fixed.                                                             | Matching behavior and score comparability.    | Lucene StandardTokenizer; English minimal stemming; lowercase; English stoplist; synonyms off by default (can be added per domain). (§B6)                                                                 |
| **Parquet schemas & partitioning**        | Columns, dtypes, compression, row group size, directory partitioning not fully fixed.                                               | IO performance; schema drift.                 | Fixed schemas listed; **ZSTD=6**, `row_group=4096`; partition by `model`, `run_id`, `shard`; filename `part-<nnnnn>.parquet`. (§B7)                                                                       |
| **DuckDB catalog**                        | Migrations, integrity checks, FK behaviors, thread settings unspecified.                                                            | Registry reliability.                         | Versioned migrations (`registry/migrations/*.sql`), FK constraints, indexes, `PRAGMA threads=14`, two‑phase registration. (§B8)                                                                           |
| **FAISS details (GPU+cuVS)**              | Factory choice confirmed, but training sample selection, OPQ params, memory budgeting, shard policy not fully pinned.               | Performance & determinism.                    | `OPQ64,IVF8192,PQ64`, train on 10M random (seed 42); `nprobe=64`; ≤10M vectors/shard; GPU add/search; save `.faiss` + `.ids`. (§B9)                                                                       |
| **vLLM topology**                         | Routing is known, but request/response schemas, timeouts, retries, health checks not fully specified.                               | Client stability.                             | OpenAI‑compatible `/v1/embeddings`; per‑request timeout 30 s; 3× retries; health endpoints `/v1/models` & `/health`; router (Nginx) config pinned; backpressure policy documented. (§B10)                 |
| **Ontology ingestion**                    | Supported predicates for labels/synonyms/defs, CURIE mapping, deprecations, and merges not fully specified.                         | Correct concept catalog and IDs.              | Accept SKOS (`prefLabel`,`altLabel`), OBO (`hasExactSynonym`,`def`), RDFS labels; support `deprecated` and `replaced_by`; CURIE mapping table; normalize text; unique `urn:concept:<ont>:<curie>`. (§B11) |
| **Linker calibration**                    | Dev set size, fold policy, calibration storage, and runtime application not explicit.                                               | Score interpretability.                       | Dev set 2,000 chunk‑concept pairs labeled; 5‑fold isotonic regression; parameters stored per `linker_run_id`; applied at runtime; ECE reported. (§B12)                                                    |
| **Hybrid fusion math**                    | RRF `k`, candidate pool sizes, KG boost exact math not completely fixed.                                                            | Deterministic results.                        | Dense@200 + Sparse@200; **RRF(k=60)**; KG boosts: direct +0.08, 1‑hop +0.04; MMR λ=0.7 at doc level. (§B13)                                                                                               |
| **API design**                            | Only endpoint sketch; errors, pagination, filters, sorting, rate limits, auth, and OpenAPI not fully specified.                     | Independent backend/frontend work.            | Full **OpenAPI** spec, error schema, pagination, filters (year, source, license), API‑key auth, per‑key rate limits (local token bucket), JSON response formats. (§B14)                                   |
| **End‑to‑end testing**                    | What E2E scenarios and fixtures to use; success criteria per stage unclear.                                                         | Integration confidence.                       | Provide seed corpora (10 docs), synthetic ontologies, golden outputs; E2E pipeline test that asserts all DDLs, artifact counts, index sizes, retrieval metrics over thresholds. (§B15)                    |
| **Observability**                         | Log schema, metric names, labels, exemplar traces, dashboards unspecified.                                                          | Ops readiness.                                | Define **metrics** (names, types), **logs** (JSON schema), **traces** (span names & attributes); provide Grafana panels JSON. (§B16)                                                                      |
| **Failure handling**                      | Error taxonomy, retry matrices, poison‑pill protocol, quarantine dirs not fixed.                                                    | Robustness under real corpora.                | Standard error classes; retry/backoff tables; quarantine locations; incident log table and CLI. (§B17)                                                                                                    |
| **Security & licensing**                  | Key storage, license enforcement points, audit fields not formalized.                                                               | Compliance and reproducibility.               | `.env` for secrets; license filters in downloader; persist license string and OA source; audit table records provenance; API keys in env; localhost binding. (§B18)                                       |
| **Project structure & code quality**      | Pre‑commit, linters, typing level, docstrings, ADRs, contribution workflow not locked.                                              | Team velocity & consistency.                  | Pre‑commit (ruff, black, mypy‑strict), conventional commits, ADR template, mkdocs site, contribution guide, codeowners. (§B19)                                                                            |
| **Local dev & bootstrap**                 | No exact bootstrap steps and Make targets.                                                                                          | Fast onboarding.                              | Provide `scripts/bootstrap.sh`, `Makefile` targets, systemd units for vLLM + Nginx, folder layout creation. (§B20)                                                                                        |
| **Performance SLAs**                      | Hard SLAs and perf targets partially stated.                                                                                        | Planning & regression gates.                  | SLAs/SLOs formalized per stage; CI gates with thresholds; perf budgets by stage. (§B21)                                                                                                                   |

---

## PART B — EXHAUSTIVE SPECIFICATIONS (RESOLUTIONS)

### B1. Canonical text, hashing & IDs

* **Canonicalizer** (DocTags → text):

  1. Decode UTF‑8; apply **Unicode NFC**.
  2. Replace Windows/old Mac line breaks with `\n`.
  3. Collapse runs of whitespace to a single space **except** keep single `\n` line breaks between block nodes (paragraph/heading/list/table caption).
  4. Normalize bullets (`•`, `◦`, `–`) to `-`.
  5. Strip page headers/footers using DocTags region types; **exclude References** section by default (configurable).
  6. Retain figure/table captions, equations (inline LaTeX left as‑is), and footnotes (inlined at end of page).
* **doc_id**: `urn:doc:sha256:<first16B base32>` over **canonical text** (not PDF bytes).
* **chunk_id**: `urn:chunk:<doc_hash>:<start>-<end>` using **character offsets** in canonical text (inclusive start, exclusive end).
* Hashes computed with SHA‑256; extraction reproducible.

### B2. Tokenizer & chunking quantization

* **Tokenizer of record**: HuggingFace tokenizer for **Qwen/Qwen3‑Embedding‑4B**.
* **Chunker parameters (fixed defaults)**: `target=400`, `overlap=80`, `min=120`, `max=480` **Qwen tokens**; stride=`target-overlap=320`.
* If a section is shorter than 120 tokens and can be merged with its successor without exceeding 480, merge; otherwise leave standalone.
* Sliding spill‑windows never cross major section boundaries (title/abstract/intro/methods/results/discussion).

### B3. DocTags selection & OCR fallback

* **Include**: title, authors, abstract, section headings, paragraphs, lists, tables, figures’ captions, equations.
* **Exclude** by default: acknowledgements, references; can enable via `config.doc_conversion.include_references=true`.
* **OCR**: page‑level fallback using Tesseract when Docling VLM avg logprob < 0.5; OCR text anchored into DocTags page node with `provenance="ocr"`.
* **Provenance** fields in DoctagsAsset: `{avg_logprob, ocr_pages:[int]}`.

### B4. Harvester & downloader (PyAlex first, with fallbacks)

* **PyAlex search**:

  * Query compiled as AND of: `topic` (applied to `title`, `abstract_inverted_index`, `fulltext`), optional `year >= from_year`, `is_oa=true OR has_oa_published_version=true OR has_oa_accepted_or_published_version=true`.
  * Pagination: `per_page=200`, resume with `cursor`, cap at `max_works`.
  * HTTP timeout=30 s, retries=3 (exponential 1.0/2.0/4.0 s), `User-Agent: kgfoundry/1.0 (+contact@example.com)`.
* **Resolution priority** (first success wins):

  1. `best_oa_location.pdf_url` (OpenAlex).
  2. Any `locations[].pdf_url` with `is_oa=true` (prefer `publishedVersion` > `acceptedVersion`).
  3. **Unpaywall** by DOI → `url_for_pdf`.
  4. Source‑specific fallback: arXiv (`/pdf/<id>.pdf`), PMC article PDF.
* **MIME + size checks**: Require `application/pdf`; max size 100 MB (configurable).
* **De‑dup**: compute `pdf_sha256` of bytes; if duplicate of an existing doc, **link** new metadata to existing file and record alias (`openalex_id`, `doi`, etc.).
* **Registration**: Insert into `documents` with all source IDs, `pdf_uri`, license, OA source, and timestamp.

### B5. Dense embeddings (Qwen3 2560‑d)

* **API**: OpenAI‑compatible `/v1/embeddings` with body:

  ```json
  {"model":"Qwen/Qwen3-Embedding-4B","input":[ "...chunk text..." ],"dimensions":2560}
  ```
* **Response validation**: assert vector length 2560; otherwise **fail run** with incident record.
* **Batching**: dynamic batch size uses VRAM probe; keep **≤80%** VRAM.
* **Normalization**: client L2‑normalizes vector (float32) and stores original L2.
* **Errors**: HTTP 429/5xx → retries (3×); timeouts → retries; after 3 failures, quarantine batch.

### B6. Sparse (SPLADE‑v3 GPU) & BM25 details

* **SPLADE‑v3 encoder**:

  * Pre‑proc: lowercase; strip control chars (`\x00`..`\x1f`), keep punctuation; whitespace collapse single space; no stopword removal.
  * Tokenization: DistilBERT WordPiece; `max_seq_len=512` (truncate tail).
  * AMP: `torch.autocast(device_type="cuda", dtype=torch.float16)`.
  * Batch size: auto‑tuned per GPU memory; cap at 2048 tokens/batch.
  * Top‑K pruning: select top 256 weights by value; tie‑break by **token id asc**; **sort ascending** token ids for indexer.
* **BM25 (Pyserini/Lucene)**:

  * Analyzer: StandardTokenizer → Lowercase → EnglishMinimalStemFilter → EnglishStopFilter.
  * Params: `k1=0.9`, `b=0.4`.
  * Fields: `title` (boost 2.0), `section` (1.2), `body` (1.0). Title from DocTags title node; section from chunk.section; body from chunk text.

### B7. Parquet datasets (ALL embeddings and chunks)

* **Common options**: `compression="ZSTD"`, level=6; `row_group_size=4096`; `use_dictionary=True` for categorical columns (`model`, `run_id`, `doc_id`, `section`).
* **Dense**:

  ```
  chunk_id: string, model: string, run_id: string, dim: int16 (=2560),
  vector: list<float>, l2_norm: float, created_at: timestamp
  ```

  Partitions: `model=Qwen3-Embedding-4B/run_id=<run>/shard=<nnnn>`.
* **SPLADE**:

  ```
  chunk_id: string, model: string, run_id: string,
  vocab_ids: list<int32>, weights: list<float>, nnz: int16, created_at: timestamp
  ```

  Partitions: `model=SPLADE-v3-distilbert/run_id=<run>/shard=<nnnn>`.
* **Chunks**:

  ```
  chunk_id, doc_id, section, start_char:int32, end_char:int32,
  doctags_span: struct<node_id:string,start:int32,end:int32>,
  text:string, tokens:int32, created_at:timestamp
  ```

  Partition: `model=docling_hybrid/run_id=<run>/shard=<nnnn>`.

### B8. DuckDB registry (≥ 1.4.1): migrations & guards

* **Migrations**: SQL files in `/registry/migrations`, applied by `registry apply-migrations`. Each migration increments `schema_version` table.
* **Two‑phase registration**: write Parquet to temp dir → validate row counts & schema → `BEGIN` → insert into `datasets`/`runs` → `COMMIT` → rename dir into place; on failure `ROLLBACK` and delete temp.
* **Threading**: `PRAGMA threads=14`; all read_parquet with `union_by_name=true`.
* **Integrity**: FKs and unique constraints as defined.
* **Views**: `dense_vectors_view`, `splade_vectors_view`, `chunk_texts` pre‑created.

### B9. FAISS GPU + cuVS (CUDA 13)

* **Factory**: `OPQ64,IVF8192,PQ64` (for d=2560).
* **Train**: Random sample of **min(10M, 100% of corpus)** dense vectors; seed=42; normalize before train.
* **Add**: On GPU; batch of N where memory ≤ 80% VRAM; record adds per shard in DuckDB.
* **Search**: `nprobe=64`; return distances as **IP or L2** consistent with training (we use **IP** on normalized vectors → cosine).
* **Persist**: `.faiss` index + `.ids` idmap per shard; `faiss_indexes` table records `logical_index_id`, `shard_id`, file paths, config.
* **Merge**: logical index = set of shards; search fans out to shards, merges top‑K.

### B10. vLLM servers & router

* **Granite‑Docling (VLM)**: vLLM process #1 on `127.0.0.1:8001`.
* **Qwen3 embeddings**: vLLM process #2 on `127.0.0.1:8002`.
* **Router**: Nginx on `:80` maps `/vlm/*` → 8001 and `/v1/embeddings` → 8002.
* **Health checks**: `/healthz` on router; `/v1/models` on each vLLM.
* **Client policy**: per‑request timeout 30 s, retries ×3 (429/5xx), exponential backoff 1/2/4 s; circuit‑breaker opens after 10 consecutive failures (per‑service) for 30 s.

### B11. Ontology ingestion & catalog

* **Formats**: OBO, OWL/RDF (TTL, RDF/XML), SKOS (RDF).
* **Predicates**:

  * Labels: `rdfs:label`, `skos:prefLabel`.
  * Synonyms: `oboInOwl:hasExactSynonym`, `skos:altLabel`.
  * Definitions: `IAO:0000115`, `skos:definition`.
  * Hierarchy: `rdfs:subClassOf`, `skos:broader`, `BFO:part_of` (normalized to `IS_A` or `PART_OF`).
  * Deprecation: `owl:deprecated` true; replacements: `iao:0100001` or custom `replaced_by`.
* **CURIEs**: Provide `prefix → base IRI` map per ontology; compute `urn:concept:<ont>:<CURIE>`.
* **Normalization**: lowercased, NFC, punctuation trimmed (except hyphen and slash), English lemmatization.
* **Concept embeddings**:

  * Dense (Qwen3 2560‑d): text = `pref_label | definition | top5_synonyms`; same Parquet schema as chunks (swap `chunk_id → concept_id`).
  * Sparse (SPLADE): text same as above; topK=128.

### B12. Linker calibration & decision policy

* **Labeling**: 2,000 chunk‑concept candidate pairs labeled by domain curators.
* **Splits**: 5‑fold CV; train isotonic regression to map fused score → calibrated [0,1].
* **Thresholds**: `τ_high=0.62`, `τ_low=0.35` applied **after calibration**.
* **Outputs**: save calibration params in DuckDB as JSON attached to `run_id` (linker run).
* **Metrics**: precision/recall/F1; **Expected Calibration Error** (ECE).

### B13. Hybrid retrieval fusion (fixed math)

* **Retrieval**: `dense@200` (FAISS), `sparse@200` (BM25 and SPLADE each produce lists; we interleave by highest score then take top 200 combined sparse).
* **RRF**: `RRF_k = 60`. Score = `Σ 1/(k + rank_i)` over participating rankers (dense, bm25, splade).
* **KG boost**: +0.08 if any linked concept of chunk ∈ query concepts; +0.04 if within one ontology hop; max boost +0.12.
* **MMR**: doc‑level, `λ=0.7`; similarity measured by cosine of mean chunk vectors per doc.

### B14. Search API — **OpenAPI 3.1** (final)

* **Auth**: Bearer API key header `Authorization: Bearer <token>` (tokens read from env `SEARCH_API_KEYS`, comma‑separated).
* **Rate limits**: token‑bucket: 120 req/min per key; 1,000 req/day per key (in‑memory counters, process‑local).
* **Endpoints**:

  * `POST /search`

    * Body:

      ```json
      {
        "query": "string",
        "k": 10,
        "filters": {"year_from": 2018, "year_to": 2025, "source": ["openalex","arxiv"], "license":["CC-BY","CC0"]},
        "explain": false
      }
      ```
    * 200 Response (per result):

      ```json
      {
        "doc_id": "urn:doc:...",
        "chunk_id": "urn:chunk:...",
        "title": "string",
        "section": "Methods",
        "score": 2.381,
        "signals": {"dense": 0.73, "sparse": 0.61, "rrf": 0.025, "kg_boost": 0.08},
        "spans": {"start_char": 120, "end_char": 420},
        "concepts": [{"concept_id":"urn:concept:mesh:D012345","label":"...","match":"direct"}]
      }
      ```
    * Errors: `400` (bad input), `401` (auth), `429` (rate limit), `500` (server).
  * `POST /graph/concepts`

    * Body: `{"q":"string","limit":50}` → returns matching concept IDs + labels + paths to root.
  * `GET /healthz` → `{status:"ok", components:{faiss:"ok", bm25:"ok", vllm_embeddings:"ok", neo4j:"ok"}}`.

### B15. End‑to‑end & stage testing

* **Seed fixtures**:

  * 10 OA PDFs (mix arXiv/PMC/journal) on disk for offline tests.
  * Tiny ontology (50 concepts, 2 levels) + MeSH subset (for integration).
  * Golden outputs: a) DocTags JSON; b) chunk Parquet; c) sample dense/sparse Parquets; d) FAISS index shards; e) SPLADE/BM25 Lucene dirs.
* **E2E test** (`pytest -m e2e`):

  1. Harvest & download (mock pyalex/unpaywall): expect N PDFs.
  2. DocTags conversion: expect pages>0, DocTags file exists, avg_logprob recorded.
  3. Chunking: expect chunks>0; token counts within bounds; offsets monotonic.
  4. Dense embed: expect vectors dim=2560; Parquet rows==chunks.
  5. SPLADE encode & BM25 index: Lucene dirs exist; postings >0.
  6. FAISS build: index files exist; search for “topic” returns >0 results.
  7. Ontology ingest + concept embeddings: counts >0; CURIEs valid.
  8. Linker: assertions produced; at least X% linked.
  9. KG upsert: node/edge counts as expected.
  10. Search API: `/search` returns 200 and k results; latency p95 < 300 ms on fixtures.
* **Stage unit tests**: contract tests per ABC; property‑based tests (idempotency; stable IDs); golden tests for DocTags/chunks; benchmarking tests for encoder throughput.

### B16. Observability & diagnostics

* **Logs**: JSON lines with fields: `ts, level, module, event, run_id, doc_id, chunk_id, duration_ms, err_code, message`.
* **Metrics** (Prometheus):

  * Counters: `pdf_download_success_total`, `pdf_download_failure_total{reason}`, `chunks_total`, `dense_vectors_total`, `splade_vectors_total`, `faiss_queries_total`, `bm25_queries_total`, `splade_queries_total`.
  * Histograms: `download_latency_ms`, `doctags_latency_ms`, `chunking_latency_ms`, `embed_dense_latency_ms`, `splade_encode_latency_ms`, `faiss_search_latency_ms`, `search_total_latency_ms`.
  * Gauges: `faiss_index_size_vectors`, `lucene_docs_count`, `duckdb_open_files`.
* **Tracing** (OpenTelemetry):

  * Spans: `harvest.search`, `download.pdf`, `docling.to_doctags`, `chunking.hybrid`, `embed.qwen3`, `encode.splade`, `index.faiss.add`, `index.lucene.build`, `linker.run`, `kg.upsert`, `api.search`.
  * Attributes: `{run_id, doc_id, shard_id, model, dim, nprobe, nnz}`.
* **Dashboards**: provide JSON exports for:

  * **Ingest Health** (download rates/errors).
  * **Pipeline Throughput** (chunks/min, embeddings/min).
  * **Search SLOs** (p50/p95/p99 latency, error rate).
  * **Index Footprint** (FAISS vectors, Lucene docs).

### B17. Failure handling & resilience

* **Error taxonomy**:

  * `DownloadError`, `UnsupportedMIMEError`, `DoclingError`, `OCRTimeout`, `ChunkingError`, `EmbeddingError`, `SpladeOOM`, `FaissOOM`, `IndexBuildError`, `OntologyParseError`, `LinkerCalibrationError`, `Neo4jError`.
* **Retry/backoff** table:

  * External HTTP (download, vLLM): retries=3, backoff 1/2/4 s.
  * GPU OOM: reduce batch by 50%, retry up to 2 times; if still fails → quarantine.
  * FAISS add error: split shard; retry once per split.
* **Quarantine**:

  * PDFs: `/data/quarantine/pdfs/…`
  * Parquet batches: `/data/quarantine/parquet/…`
  * Incidents table: `registry.incidents(event, subject_id, error_class, message, created_at)`.
* **Poison pill**: if a `doc_id` fails at stage X twice, mark `documents.status='poison'` and exclude from subsequent runs until manual override.

### B18. Security, licensing, config secrets

* **Local‑only** network binding for vLLM and Neo4j.
* **API keys** for Search API read from env `SEARCH_API_KEYS` (comma‑separated).
* **Licenses** stored verbatim from source; `documents.license` required for downstream processing; refuse embedding/indexing if missing or not OA.
* **.env** template for secrets and contact email; never commit real secrets.

### B19. Code quality & governance

* **Pre‑commit**: `ruff`, `black`, `mypy --strict`, `pyupgrade`, `interrogate` (docstring coverage).
* **Type discipline**: `from __future__ import annotations`, Protocol/ABC for interfaces, `Final` constants.
* **Docs**: `mkdocs` site with API docs (pdoc or mkdocstrings), how‑to guides, and ADRs (`/docs/adr/0001-...md`).
* **Conventional commits** and **semantic versioning** per package.
* **Codeowners** by directory.

### B20. Bootstrap & DevEx

* **Makefile** targets:

  * `make bootstrap` → create venv, install deps, install CUDA‑13 Torch wheels, build/install FAISS GPU/cuVS (or fetch wheel), install vLLM prerelease, create dirs under `/data/*`, apply migrations, install systemd units for vLLM and Nginx.
  * `make run` → start router, start API, ensure vLLM services up.
  * `make e2e` → run the full fixture pipeline & API checks.
  * `make clean` → remove indexes and temp outputs.
* **Systemd** units: `vllm-vlm.service`, `vllm-embed.service`, `nginx.service`, `search-api.service`.
* **Directory layout**:

  ```
  /data/pdfs
  /data/doctags
  /data/parquet/{chunks,dense,sparse,concepts}
  /data/lucene/{bm25,splade}
  /data/faiss
  /data/catalog/catalog.duckdb
  /data/quarantine/{pdfs,parquet}
  ```

### B21. SLAs/SLOs & performance budgets

* **Downloader**: ≥ 95% of accessible OA PDFs succeed within 60 s; retries included.
* **DocTags**: p95 ≤ 6 s/10 pages (Granite‑Docling, RTX 5090).
* **Chunking**: ≥ 5,000 chunks/min CPU.
* **Dense embedding**: ≥ 8,000 chunks/min (2560‑d) on RTX 5090 under 80% VRAM.
* **SPLADE encoding**: ≥ 5,000 chunks/min (AMP).
* **FAISS search**: p95 ≤ 40 ms for 200‑NN; build time ≤ 2 h per 10M vectors shard.
* **Search API**: p95 end‑to‑end ≤ 300 ms for `k=10`, concurrency 64.
* **Linker run**: ≥ 1M chunk‑concept candidate scorings/hour; ECE ≤ 0.08; F1 ≥ 0.70 on dev set.
* **CI gates**: fail if nDCG@10 drops > 0.02 from last green or p95 latency ↑ > 20%.

---

## PART C — INTERFACES & MODULE CONTRACTS (READY FOR CODING)

Below are **Python interface signatures** (type‑checked; exceptions documented). All implementations must adhere to these contracts.

### C1. Registry (DuckDB) — `registry/api.py`

```python
class Registry(Protocol):
    def begin_dataset(self, kind: str, run_id: str) -> str: ...
    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None: ...
    def rollback_dataset(self, dataset_id: str) -> None: ...

    def insert_run(self, purpose: str, model_id: str | None, revision: str | None, config: dict) -> str: ...
    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None: ...

    def register_documents(self, docs: list[Doc]) -> None: ...
    def register_doctags(self, assets: list[DoctagsAsset]) -> None: ...
    def register_chunks(self, dataset_id: str, chunk_rows: int) -> None: ...
    def register_dense_run(self, run_id: str, model: str, dim: int, parquet_root: str) -> None: ...
    def register_sparse_run(self, run_id: str, model: str, vocab_size: int, parquet_root: str, backend: str) -> None: ...
    def register_faiss_shard(self, logical_index_id: str, run_id: str, shard_id: int, cfg: dict, index_uri: str, idmap_uri: str) -> None: ...

    def emit_event(self, event_name: str, subject_id: str, payload: dict) -> None: ...
    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None: ...
```

### C2. Downloader — `download/harvester.py`

```python
class Harvester(Protocol):
    def search(self, topic: str, years: str, max_works: int) -> list[dict]: ...
    def resolve_pdf(self, work: dict) -> str | None: ...
    def download_pdf(self, url: str, target_path: str) -> str: ...  # returns final path
    def run(self, topic: str, years: str, max_works: int) -> list[Doc]: ...
```

**Exceptions**: `DownloadError`, `UnsupportedMIMEError`.

### C3. DocTags & chunking — `docling/vlm.py`, `docling/hybrid.py`

```python
class DocTagsConverter(Protocol):
    def to_doctags(self, pdf_path: str) -> dict: ...
    def persist(self, doc_id: str, doctags: dict) -> str: ...

class Chunker(Protocol):
    def chunk(self, doctags_uri: str) -> list[Chunk]: ...
```

**Exceptions**: `DoclingError`, `OCRTimeout`, `ChunkingError`.

### C4. Embeddings — `embeddings_dense/qwen3.py`, `embeddings_sparse/splade.py`, `embeddings_sparse/bm25.py`

```python
class DenseEmbedder(Protocol):
    name: str
    dim: int
    def embed_texts(self, texts: list[str]) -> "np.ndarray": ...

class SparseEncoder(Protocol):
    name: str
    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]: ...

class SparseIndex(Protocol):
    def build(self, docs_iterable: "Iterable[tuple[str, dict]]") -> None: ...
    def search(self, query: str, k: int, fields: dict | None = None) -> list[tuple[str, float]]: ...
```

**Exceptions**: `EmbeddingError`, `SpladeOOM`, `IndexBuildError`.

### C5. Vector store — `vectorstore_faiss/gpu.py`

```python
class VectorStore(Protocol):
    def train(self, train_vectors: "np.ndarray", *, factory: str, seed: int = 42) -> None: ...
    def add(self, keys: list[str], vectors: "np.ndarray") -> None: ...
    def search(self, query: "np.ndarray", k: int) -> list[tuple[str, float]]: ...
    def save(self, index_uri: str, idmap_uri: str) -> None: ...
    def load(self, index_uri: str, idmap_uri: str) -> None: ...
```

**Exceptions**: `FaissOOM`, `IndexBuildError`.

### C6. Ontology & catalog — `ontology/loader.py`

```python
class OntologyCatalog(Protocol):
    def load(self, inputs: list[dict]) -> list[Concept]: ...
    def neighbors(self, concept_id: str, depth: int = 1) -> list[str]: ...
    def get(self, concept_id: str) -> Concept | None: ...
```

### C7. Linker — `linking/linker.py`

```python
class Linker(Protocol):
    def candidate_concepts(self, chunk_text: str, k_sparse: int, k_lex: int) -> list[str]: ...
    def score(self, chunk_vec: "np.ndarray", concept_vecs: dict[str, "np.ndarray"], features: dict) -> float: ...
    def calibrate(self, devset: list[tuple[float, int]]) -> dict: ...
    def link_chunk(self, chunk: Chunk) -> list[LinkAssertion]: ...
```

### C8. KG builder — `kg_builder/neo4j.py`

```python
class GraphStore(Protocol):
    def upsert_docs(self, docs: list[Doc]) -> None: ...
    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...
    def upsert_concepts(self, concepts: list[Concept]) -> None: ...
    def upsert_mentions(self, assertions: list[LinkAssertion]) -> None: ...
    def linked_concepts(self, chunk_id: str) -> list[str]: ...
```

### C9. Search API — `search_api/app.py`

* Bootstraps FAISS handle(s), BM25, SPLADE, OntologyCatalog, GraphStore, DenseEmbedder; exposes FastAPI app with routers:

  * `/search`, `/graph/concepts`, `/healthz`.
* All responses validated with `pydantic` models.

---

## PART D — ORCHESTRATION (PREFECT 2.x)

### D1. Flow graph & idempotency keys

```
harvest_and_download → convert_to_doctags → chunk_with_docling → \
embed_dense_qwen3 ┐                                        \
encode_splade_v3  ├── build_bm25_index ┐                    \
                   └── build_faiss_index  → ingest_ontologies → embed_concepts \
                                                   → link_chunks_to_concepts → kg_upsert
```

* **Idempotency keys**:

  * Harvest: `(topic, years, work_id)`
  * DocTags: `pdf_sha256`
  * Chunking: `(doc_id, chunker_version)`
  * Dense: `(chunk_dataset_id, model=Qwen3, dim=2560)`
  * SPLADE: `(chunk_dataset_id, model=SPLADE-v3, topk=256)`
  * FAISS: `(dense_run_id, factory, nlist, m, nprobe)`
  * Ontology: `(ontology_id, src_uri, loader_version)`
  * Linker: `(chunk_dataset_id, ontology_id, linker_version)`

### D2. Concurrency & retries

* Download: 8 workers, retry matrix as in §B17.
* VLM/embeddings/SPLADE: GPU queue depth=2; batch auto‑tune; 3 retries on transient errors.
* Index builds: FAISS shards in parallel up to 2; Lucene index single‑writer with merges at end.

### D3. CLI commands (invoke via Typer)

* `kgf harvest --topic "LLM alignment" --years 2018..2025 --max 20000`
* `kgf doctags --input /data/pdfs --out /data/doctags`
* `kgf chunk --doctags /data/doctags --out /data/parquet/chunks`
* `kgf embed-dense --chunks <dataset_id>`
* `kgf encode-splade --chunks <dataset_id>`
* `kgf index-bm25 --chunks <dataset_id>`
* `kgf index-faiss --dense-run <run_id>`
* `kgf ingest-ontology --id mesh --uri /data/ontologies/mesh.obo`
* `kgf embed-concepts --ontology mesh`
* `kgf link --chunks <dataset_id> --ontology mesh`
* `kgf kg-upsert --link-run <run_id>`
* `kgf api --port 8080`

---

## PART E — END‑TO‑END ACCEPTANCE CRITERIA (PER WORKSTREAM)

**Downloader**

* Given topic and year range, at least N documents discovered; ≥95% OA downloads succeed; duplicates deduped; license persisted.

**DocTags**

* For each PDF, DocTags exists; pages count > 0; avg_logprob recorded; OCR fallback pages tracked.

**Chunking**

* For each `doc_id`, sum of chunk spans covers ≥90% of canonical text; token counts in [min,max]; offsets monotonic.

**Dense embeddings**

* Row count == chunk rows; all vectors dim=2560; Parquet schema validated; L2 norms avg in [0.95,1.05] post‑norm.

**SPLADE/BM25**

* SPLADE nnz distribution mean ≈ 220–260; Lucene impact index has postings; BM25 index doc count == chunk count.

**FAISS**

* Trained index exists; top‑K search returns results; p95 latency ≤ 40 ms for K=200 on fixture set.

**Ontology & concepts**

* Loaded concepts > 0; no duplicate CURIEs; concept embeddings Parquet rows == concepts; ontology edges inserted.

**Linker**

* Produces assertions; on fixtures F1 ≥ 0.70; ECE ≤ 0.08; calibration saved and applied.

**KG**

* Node uniqueness constraints enforced; edge counts match assertions; query `linked_concepts(chunk)` returns expected IDs.

**Search API**

* `/healthz` OK; `/search` returns k results with explanations; p95 ≤ 300 ms on fixtures.

---

## PART F — SAMPLE OPENAPI (EXCERPT)

```yaml
openapi: 3.1.0
info: {title: kgfoundry Search API, version: 1.0.0}
paths:
  /search:
    post:
      security: [{ApiKeyAuth: []}]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [query]
              properties:
                query: {type: string, minLength: 1}
                k: {type: integer, minimum: 1, maximum: 100, default: 10}
                filters:
                  type: object
                  properties:
                    year_from: {type: integer}
                    year_to: {type: integer}
                    source: {type: array, items: {type: string}}
                    license: {type: array, items: {type: string}}
                explain: {type: boolean, default: false}
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      $ref: '#/components/schemas/Result'
        '400': {$ref: '#/components/responses/BadRequest'}
        '401': {$ref: '#/components/responses/Unauthorized'}
        '429': {$ref: '#/components/responses/TooManyRequests'}
        '500': {$ref: '#/components/responses/ServerError'}
components:
  securitySchemes:
    ApiKeyAuth:
      type: http
      scheme: bearer
  schemas:
    Result:
      type: object
      required: [doc_id, chunk_id, title, section, score]
      properties:
        doc_id: {type: string}
        chunk_id: {type: string}
        title: {type: string}
        section: {type: string}
        score: {type: number}
        signals:
          type: object
          properties:
            dense: {type: number}
            sparse: {type: number}
            rrf: {type: number}
            kg_boost: {type: number}
        spans:
          type: object
          properties:
            start_char: {type: integer}
            end_char: {type: integer}
        concepts:
          type: array
          items:
            type: object
            properties:
              concept_id: {type: string}
              label: {type: string}
              match: {type: string, enum: [direct,nearby]}
```

---

## PART G — IMPLEMENTATION CHECKLISTS (BY WORKSTREAM)

1. **Download & Harvest**

   * [ ] PyAlex client; config; retries; polite UA.
   * [ ] Fallback Unpaywall client; DOI resolver.
   * [ ] PDF validator; MIME/size checks; hash; dedup; storage path; registry insert.
   * [ ] Metrics/logs/tests; CLI command; E2E hooks.

2. **DocTags Conversion**

   * [ ] vLLM client; page batches; OCR fallback; canonicalizer; provenance.
   * [ ] Persist `.dt.json.zst`; register DoctagsAsset; golden tests.

3. **Hybrid Chunker**

   * [ ] Qwen tokenizer; section splitting; sliding windows; spans & offsets.
   * [ ] Parquet writer; ID scheme; tests for bounds & coverage.

4. **Dense Embedder (Qwen3 2560)**

   * [ ] OpenAI API client; dimension param; batching; normalization; retries.
   * [ ] Parquet writer; registry run; throughput metric; tests.

5. **SPLADE‑v3 & BM25**

   * [ ] Torch encoder; AMP; top‑K; Parquet writer; impact index streaming.
   * [ ] BM25 analyzer config; indexer; search adapter; tests.

6. **FAISS GPU/cuVS**

   * [ ] Loader; training sampler; add & search; shard manager; save/load.
   * [ ] Query adapter (top‑K); memory gauges; tests.

7. **Ontology & Concepts**

   * [ ] Parsers (OBO/OWL/SKOS); normalization; CURIEs; edges.
   * [ ] Dense & SPLADE embeddings for concepts; Parquet & registry.

8. **Linker**

   * [ ] Candidate gen (SPLADE+lexicon); feature extractor; fusion; calibration; thresholds.
   * [ ] Assertions Parquet; metrics; tests.

9. **KG Builder (Neo4j)**

   * [ ] Node/edge upserts; uniqueness constraints; indexes.
   * [ ] Linked concept lookups; tests.

10. **Search API**

    * [ ] Startup wiring; RRF; KG boosts; MMR; filters; auth; rate limits.
    * [ ] OpenAPI; integration tests; latency regression tests.

11. **Registry & Orchestration**

    * [ ] Migrations; two‑phase dataset registration; events & incidents.
    * [ ] Prefect flows; CLI; idempotency keys; dashboards.

12. **Ops**

    * [ ] systemd units; Nginx config; health checks; log rotation; backup scripts (Parquet+DuckDB).

---

## PART H — EXAMPLE CONFIG (FINAL; READY TO COMMIT)

```yaml
system:
  os: ubuntu-24.04
  threads: 14
  seed: 42
  parquet_root: /data/parquet
  artifacts_root: /data/artifacts
  duckdb_path: /data/catalog/catalog.duckdb

runtime:
  python: "3.13"
  cuda: "13.0"
  torch: "2.9"
  duckdb_min: "1.4.1"
  vllm_channel: "pre-release-cuda13"

network:
  nginx_port: 80
  vlm_port: 8001
  emb_port: 8002
  api_port: 8080

harvest:
  provider: pyalex
  per_page: 200
  max_works: 20000
  years: ">=2018"
  filters:
    is_oa: true
    has_oa_published_version: true
  fallbacks:
    unpaywall: true
    arxiv: true
    pmc: true
  concurrency: 8
  timeout_sec: 60
  retries: 3

doc_conversion:
  vlm_model: ibm-granite/granite-docling-258M
  vlm_revision: untied
  endpoint: http://localhost/vlm/
  dpi: 220
  page_batch: 8
  ocr_fallback: true
  max_pages: 2000
  timeout_sec: 120

chunking:
  engine: docling_hybrid
  target_tokens: 400
  overlap_tokens: 80
  min_tokens: 120
  max_tokens: 480

dense_embedding:
  model: Qwen/Qwen3-Embedding-4B
  endpoint: http://localhost/v1/embeddings
  output_dim: 2560
  parquet_out: ${system.parquet_root}/dense/model=Qwen3-Embedding-4B/run=${run_id}

sparse_embedding:
  splade:
    model: naver/splade-v3-distilbert
    device: cuda
    amp: fp16
    max_seq_len: 512
    topk: 256
    parquet_out: ${system.parquet_root}/sparse/model=SPLADE-v3-distilbert/run=${run_id}
  bm25:
    k1: 0.9
    b: 0.4
    field_boosts: { title: 2.0, section: 1.2, body: 1.0 }
    index_dir: /data/lucene/bm25

faiss:
  index_factory: OPQ64,IVF8192,PQ64
  nprobe: 64
  train_samples: 10000000
  shards:
    max_vectors_per_shard: 10000000
  gpu: true
  cuvs: true
  output_dir: /data/faiss/qwen3_ivfpq

ontology:
  inputs:
    - { ontology_id: mesh, format: obo, uri: /data/ontologies/mesh.obo }
    - { ontology_id: go,   format: obo, uri: /data/ontologies/go.obo }
  concept_embed:
    dense_model: Qwen3-Embedding-4B
    dense_dim: 2560
    splade_model: SPLADE-v3-distilbert
    splade_topk: 128

linker:
  candidates: { splade_topk: 100, lexicon_topk: 50 }
  fusion_weights: { dense: 0.55, sparse: 0.35, lexical: 0.10, depth_bonus_per_level: 0.02, depth_cap: 0.10 }
  thresholds: { high: 0.62, low: 0.35 }
  calibration: isotonic

graph:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password_env: NEO4J_PASSWORD

search:
  k: 10
  dense_candidates: 200
  sparse_candidates: 200
  rrf_k: 60
  mmr_lambda: 0.7
  kg_boosts: { direct: 0.08, one_hop: 0.04 }
```

---

## PART I — WHAT HAPPENS NEXT

With these **resolved specifications**:

* Each workstream has **clear interfaces**, **error types**, **Parquet schemas**, **DuckDB DDL**, **router/API contracts**, **orchestration flows**, **observability**, **SLAs**, and **acceptance tests**.
* Teams can now draft **detailed implementation plans** and begin coding, confident that components will integrate cleanly on the single‑box target environment.

If you want, I can generate **skeleton packages** (directories, ABCs, Pydantic models, Typer CLI, Prefect flows, Makefile, pre‑commit, and sample tests) to serve as the starting point for implementation.
