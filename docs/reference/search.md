# Agent Catalog Search API Reference

This document describes the agent catalog search API, including typed configuration helpers, request/response formats, and error handling.

## Overview

The agent catalog search API provides hybrid lexical-vector search across the kgfoundry codebase. It combines:
- **Lexical (BM25)** search for keyword and term matching
- **Vector (semantic)** search for meaning-based queries

Results are ranked by a configurable `alpha` parameter that tunes the mixing of these signals.

## Core Types

### SearchOptions

Configuration for search queries. Use the helper factories (below) to construct valid instances.

**Schema:** [`search_options.v1.json`](../schemas/search-options.v1.json)

**Fields:**
- `alpha` (float, default 0.6): Mixing parameter ∈ [0.0, 1.0]. 0.0 = pure lexical, 1.0 = pure vector.
- `candidate_pool` (int, default 100): Pre-filtering candidate pool size for re-ranking. Must be ≥ k (final result count).
- `batch_size` (int, default 32): Embedding model batch size for encoding.
- `embedding_model` (str | null, default null): Name or path of embedding model (e.g., "all-MiniLM-L6-v2").
- `facets` (dict | null): Optional result filters. See Facet Filtering below.

### SearchDocument

Intermediate representation for a catalog symbol, produced by the `make_search_document` helper.

**Schema:** [`search_document.v1.json`](../schemas/search-document.v1.json)

**Fields:**
- `symbol_id` (str): Fully qualified symbol ID (e.g., "py:kgfoundry.agent_catalog.search.find_similar")
- `package` (str): Top-level package name
- `module` (str): Fully qualified module path
- `qname` (str): Qualified name within the module
- `kind` (str): Symbol kind (class, function, module, method, property, attribute, type, protocol)
- `stability` (str | null): API stability level (stable, experimental, deprecated)
- `deprecated` (bool): Whether symbol is deprecated
- `summary` (str | null): One-line docstring summary
- `docstring` (str | null): Full docstring
- `text` (str): Normalized combined text for lexical indexing
- `tokens` (dict): Lexical token counts for indexing
- `anchor_start` (int | null): Starting line number (1-indexed)
- `anchor_end` (int | null): Ending line number (1-indexed)
- `row` (int, default -1): Row/database identifier

### SearchRequest

Parameters for a search query.

**Schema:** [`search_request.v1.json`](../schemas/search-request.v1.json)

**Fields:**
- `query` (str, required): Search query string
- `k` (int, default 10): Number of results to return
- `options` (SearchOptions): Configuration for the search

### SearchResult

A single result from a search query.

**Schema:** [`search_response.json`](../schemas/search-response.json) / `VectorSearchResult`

**Fields:**
- `symbol_id` (str): Symbol identifier
- `score` (float): Combined relevance score [0.0, 1.0]
- `lexical_score` (float): BM25 lexical score
- `vector_score` (float): Vector similarity score
- `package`, `module`, `qname`, `kind` (str): Symbol metadata
- `stability` (str | null), `deprecated` (bool): API metadata
- `summary` (str | null): Symbol summary
- `anchor` (dict): Source anchor metadata (start_line, end_line)
- `metadata` (dict): Additional metadata

## Helper Factories

### `build_default_search_options`

Construct `SearchOptions` with canonical defaults.

```python
from kgfoundry.agent_catalog.search import build_default_search_options

# Use all defaults
opts = build_default_search_options()
# => SearchOptions(alpha=0.6, candidate_pool=100, batch_size=32, ...)

# Override specific parameters
opts = build_default_search_options(alpha=0.3, candidate_pool=200)
# => SearchOptions(alpha=0.3, candidate_pool=200, batch_size=32, ...)
```

**Validation:**
- `alpha` must be in [0.0, 1.0]
- `candidate_pool` must be non-negative
- Raises `AgentCatalogSearchError` with RFC 9457 Problem Details on invalid input

### `build_faceted_search_options`

Construct `SearchOptions` with facet filters.

```python
from kgfoundry.agent_catalog.search import build_faceted_search_options

# Filter by package and kind
opts = build_faceted_search_options(facets={
    "package": "kgfoundry",
    "kind": "class"
})
# => SearchOptions(facets={"package": "kgfoundry", "kind": "class"}, ...)

# Combine with other parameters
opts = build_faceted_search_options(
    facets={"stability": "stable"},
    alpha=0.7,
    candidate_pool=150
)
```

**Validation:**
- Facet keys must be in allow-list: `package`, `module`, `kind`, `stability`
- Raises `AgentCatalogSearchError` if unknown facet keys provided

**Supported Facets:**
- `package` (str): Package name (e.g., "kgfoundry")
- `module` (str): Module path (e.g., "kgfoundry.agent_catalog.search")
- `kind` (str): Symbol kind (class, function, module, method, property, attribute, type, protocol)
- `stability` (str): API stability (stable, experimental, deprecated)

### `build_embedding_aware_search_options`

Construct `SearchOptions` for vector search with custom embedding model.

```python
from kgfoundry.agent_catalog.search import build_embedding_aware_search_options

# Provide embedding model and loader
def load_embedding_model(name: str) -> object:
    # Custom loader logic
    return model

opts = build_embedding_aware_search_options(
    embedding_model="all-MiniLM-L6-v2",
    model_loader=load_embedding_model,
    alpha=0.8  # Emphasize vector search
)

# With facets
opts = build_embedding_aware_search_options(
    embedding_model="bge-base-en-v1.5",
    model_loader=load_embedding_model,
    facets={"kind": "function"},
    candidate_pool=250
)
```

**Validation:**
- `embedding_model` and `model_loader` are required
- Facet keys validated against allow-list if provided
- Raises `AgentCatalogSearchError` if dependencies missing

### `make_search_document`

Construct a `SearchDocument` with normalized text and token generation.

```python
from kgfoundry.agent_catalog.search import make_search_document

doc = make_search_document(
    symbol_id="py:kgfoundry.agent_catalog.search.find_similar",
    package="kgfoundry",
    module="kgfoundry.agent_catalog.search",
    qname="find_similar",
    kind="function",
    summary="Find semantically similar symbols",
    docstring="Uses hybrid lexical-vector search to locate symbols..."
)

# With anchor and row information
doc = make_search_document(
    symbol_id="py:kgfoundry.agent_catalog.client.SearchClient",
    package="kgfoundry",
    module="kgfoundry.agent_catalog.client",
    qname="SearchClient",
    kind="class",
    stability="experimental",
    deprecated=False,
    anchor_start=42,
    anchor_end=89,
    row=0
)
```

**Behavior:**
- Normalizes whitespace in text fields
- Generates lexical tokens from combined text
- Validates all required fields present
- Returns immutable `SearchDocument` instance

## Facet Filtering

Filter results to specific symbol categories using the `facets` parameter:

```python
from kgfoundry.agent_catalog.search import search_catalog, build_faceted_search_options

# Filter to classes in the kgfoundry package
facets = {"package": "kgfoundry", "kind": "class"}
opts = build_faceted_search_options(facets=facets)
results = search_catalog("vector search", k=10, options=opts)

# Filter to stable functions
facets = {"stability": "stable", "kind": "function"}
opts = build_faceted_search_options(facets=facets)
results = search_catalog("find index", k=20, options=opts)
```

## Error Handling

Search operations return structured errors following [RFC 9457 Problem Details](https://tools.ietf.org/html/rfc9457).

**Example error response:**

```json
{
  "type": "https://kgfoundry.dev/errors/catalog-search-error",
  "title": "Invalid Search Options",
  "status": 400,
  "detail": "alpha must be in [0.0, 1.0], got 1.5",
  "instance": "/catalog/search"
}
```

**Common errors:**
- `alpha` out of range [0.0, 1.0]
- `candidate_pool` negative
- Unknown facet keys
- Missing embedding model or loader for vector search

All validation errors are raised as `AgentCatalogSearchError` exceptions, which carry Problem Details metadata for HTTP responses.

## Usage Examples

### Basic Search

```python
from kgfoundry.agent_catalog.client import SearchClient

client = SearchClient()
results = client.search("vector search", k=5)
for result in results:
    print(f"{result.symbol_id}: {result.summary}")
```

### Faceted Search

```python
from kgfoundry.agent_catalog.client import SearchClient
from kgfoundry.agent_catalog.search import build_faceted_search_options

client = SearchClient()
opts = build_faceted_search_options(facets={
    "package": "search_api",
    "kind": "class"
})
results = client.search("index", k=10, options=opts)
```

### Custom Configuration

```python
from kgfoundry.agent_catalog.search import build_embedding_aware_search_options, search_catalog

def my_model_loader(name: str):
    # Custom embedding model loading logic
    return load_model(name)

opts = build_embedding_aware_search_options(
    embedding_model="all-MiniLM-L6-v2",
    model_loader=my_model_loader,
    alpha=0.7,  # Emphasize vector search
    candidate_pool=150
)

results = search_catalog("hybrid search", k=10, options=opts)
```

## Configuration Parameters

### Alpha (Mixing Parameter)

Tunes the balance between lexical and vector search:
- `0.0`: Pure lexical (BM25) search - best for exact keyword matches
- `0.3`: Lexical-dominant - good balance for broad queries
- `0.5`: Balanced - default for general-purpose search
- `0.7`: Vector-dominant - better for semantic/conceptual queries
- `1.0`: Pure vector search - best for semantic similarity

### Candidate Pool

Pre-filtering pool before re-ranking. Larger pools increase accuracy at the cost of latency. Should be at least 5-10× k for best results.

- Default: 100
- Recommended: 50-200 for k=5-20 results
- Min: 0 (no limit)

### Batch Size

Embedding model batch size. Larger batches improve throughput but increase memory usage.

- Default: 32
- Typical range: 16-128

## See Also

- [JSON Schemas](../schemas/)
- [Problem Details (RFC 9457)](https://tools.ietf.org/html/rfc9457)
- API Reference: `kgfoundry.agent_catalog.search`
- CLI: `kgfoundry catalog search`
