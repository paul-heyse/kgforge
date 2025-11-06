# Interface Catalog

This catalog is generated from `_nav.json` sidecars and the shared interface registry.

| Interface | Type | Module | Owner | Stability | Spec | Description | Problem Details |
| --------- | ---- | ------ | ----- | -------- | ---- | ----------- | ---------------- |
| orchestration-cli | cli | [orchestration](../modules/orchestration.md) | @orchestration | beta | [CLI Spec](../api/openapi-cli.md) | Primary Typer application for orchestration flows and indexing commands. | schema/examples/problem_details/tool-execution-error.json |
| search-http | http | [search_api](../modules/search_api.md) | @search-api | experimental | [HTTP API](../api/index.md) | FastAPI application exposing search operations via the public HTTP API. | schema/examples/problem_details/search-missing-index.json, schema/examples/problem_details/search-gpu-unavailable.json |

## orchestration-cli

* **Type:** cli
* **Module:** orchestration
* **Owner:** @orchestration
* **Stability:** beta
* **Description:** Primary Typer application for orchestration flows and indexing commands.

### Operations

- [`cli.index_bm25`](../api/openapi-cli.md) — Build BM25 index from JSON/Parquet chunks.
    - Module docs: [orchestration.cli](../modules/orchestration.cli.md)
    - Source: [orchestration.cli](https://github.com/example/kgfoundry/blob/main/src/orchestration/cli.py)
  - Tags: orchestration, index_bm25
  - Handler: `orchestration.cli:index_bm25`
  - Problem Details: schema/examples/problem_details/tool-execution-error.json
  - Code Samples:
    * (bash) `kgf index-bm25 data/chunks.parquet --backend lucene --index-dir ./_indices/bm25`
- [`cli.index_faiss`](../api/openapi-cli.md) — Build FAISS index from dense vectors.
    - Module docs: [orchestration.cli](../modules/orchestration.cli.md)
    - Source: [orchestration.cli](https://github.com/example/kgfoundry/blob/main/src/orchestration/cli.py)
  - Tags: orchestration, index_faiss
  - Handler: `orchestration.cli:index_faiss`
  - Env: KGF_FAISS_RESOURCES
  - Problem Details: schema/examples/problem_details/tool-execution-error.json, schema/examples/problem_details/faiss-index-build-timeout.json
  - Code Samples:
    * (bash) `kgf index-faiss artifacts/vectors.json --factory 'OPQ64,IVF8192,PQ64' --metric ip`
- [`cli.api`](../api/openapi-cli.md) — Launch FastAPI search service.
    - Module docs: [orchestration.cli](../modules/orchestration.cli.md)
    - Source: [orchestration.cli](https://github.com/example/kgfoundry/blob/main/src/orchestration/cli.py)
  - Tags: orchestration, api
  - Handler: `orchestration.cli:api`
  - Env: KGF_SEARCH_CONFIG
  - Problem Details: schema/examples/problem_details/public-api-invalid-config.json
  - Code Samples:
    * (bash) `kgf api --port 8080`
- [`cli.e2e`](../api/openapi-cli.md) — Execute end-to-end orchestration demo.
    - Module docs: [orchestration.cli](../modules/orchestration.cli.md)
    - Source: [orchestration.cli](https://github.com/example/kgfoundry/blob/main/src/orchestration/cli.py)
  - Tags: orchestration, e2e
  - Handler: `orchestration.cli:e2e`
  - Env: KGF_PROFILE
  - Problem Details: schema/examples/problem_details/public-api-invalid-config.json
  - Code Samples:
    * (bash) `kgf e2e`

## search-http

* **Type:** http
* **Module:** search_api
* **Owner:** @search-api
* **Stability:** experimental
* **Description:** FastAPI application exposing search operations via the public HTTP API.

