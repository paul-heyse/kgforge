## Why

The CodeIntel MCP server currently spawns new subprocesses for every Git operation and text search, creates fresh HTTP connections for each embedding batch, and uses synchronous I/O that blocks the event loop under load. These pragmatic choices were appropriate for initial implementation but create performance bottlenecks and operational brittleness as usage scales:

1. **Subprocess Overhead**: `git blame` and `git log` spawn new processes (~50-100ms overhead per call), parse text output with fragile regex, and fail silently on encoding issues. For repositories with thousands of files, this multiplies latency and resource consumption.

2. **HTTP Connection Churn**: `VLLMClient.embed_batch` creates a new `httpx.Client` for every batch, forfeiting TCP connection reuse and HTTP/1.1 keep-alive benefits. Embedding 10K chunks means 10K connection handshakes (~5-10ms overhead each).

3. **Blocking I/O**: Adapters like `list_paths`, `blame_range`, and `file_history` run synchronously in FastAPI's threadpool. Under concurrent load, thread exhaustion causes request queuing and latency spikes.

4. **Fixed FAISS Parameters**: The index uses `IVF-PQ` with hardcoded `nlist=8192`, optimized for large codebases but suboptimal for small repositories (<10K vectors). Small codebases pay quadratic training cost without recall benefits.

5. **No Incremental Indexing**: Re-running `index_all.py` regenerates the entire index (hours for large repos), discarding the previous index. New commits require full rebuild rather than incremental updates.

Fixing these issues is required to:
- **Reduce Latency**: Cut adapter response times by 50-80% via connection pooling and library-based Git operations.
- **Improve Reliability**: Replace fragile subprocess text parsing with typed Python APIs (GitPython, pygit2).
- **Enable Concurrency**: Convert blocking I/O to async patterns, allowing 10-100x more concurrent requests.
- **Adapt to Scale**: Choose FAISS index parameters dynamically based on corpus size (flat for small, IVF-PQ for large).
- **Support Incremental Updates**: Add new chunks to index without full rebuild, reducing indexing time from hours to seconds.

## What Changes

This proposal retrofits the CodeIntel MCP server with **production-grade performance optimizations** across five dimensions: Git operations, HTTP client management, async I/O, adaptive indexing, and incremental updates.

### Core Changes

**1. Git Operations → GitPython**
- **ADDED**: `codeintel_rev/io/git_client.py` wrapping GitPython `Repo` with typed APIs
- **MODIFIED**: `blame_range` and `file_history` adapters to use `GitClient` instead of subprocess
- **REMOVED**: Text parsing logic from `history.py` (replaced by GitPython structured returns)
- **BENEFIT**: 50-80ms latency reduction per Git operation, locale-independent, unit-testable

**2. HTTP Client Pooling → Persistent httpx.Client**
- **MODIFIED**: `VLLMClient` to initialize `httpx.Client` in `__init__`, reuse across batches
- **ADDED**: `VLLMClient.close()` method for resource cleanup (called in `ApplicationContext` shutdown)
- **ADDED**: Async variant `VLLMClient.embed_batch_async` for concurrent embedding generation
- **BENEFIT**: 5-10ms latency reduction per batch, HTTP/1.1 keep-alive, connection pooling

**3. Async I/O Conversion → asyncio.to_thread**
- **MODIFIED**: `list_paths`, `blame_range`, `file_history` converted to `async def` with `asyncio.to_thread` wrappers
- **ADDED**: `AsyncGitClient` variant for async Git operations (uses `asyncio.to_thread` internally)
- **BENEFIT**: 10-100x concurrency improvement, no thread exhaustion under load

**4. Adaptive FAISS Indexing → Dynamic Parameter Selection**
- **MODIFIED**: `FAISSManager.build_index` to choose index type based on corpus size:
  - `< 5K vectors`: `IndexFlatIP` (exact search, fast training)
  - `5K-50K vectors`: `IVFFlat` with dynamic `nlist = sqrt(n_vectors)`
  - `> 50K vectors`: `IVF-PQ` with `nlist = max(sqrt(n_vectors), 1024)`
- **ADDED**: `FAISSManager.estimate_memory_usage()` for capacity planning
- **BENEFIT**: 10-100x faster training for small repos, better recall for medium repos

**5. Incremental Index Updates → Add-Only Flat Index**
- **ADDED**: `FAISSManager.update_index(new_vectors, new_ids)` for incremental updates
- **ADDED**: Dual-index architecture: primary IVF-PQ + secondary Flat for new chunks
- **ADDED**: `FAISSManager.merge_indexes()` to periodically rebuild primary index
- **MODIFIED**: `index_all.py` to support `--incremental` mode
- **BENEFIT**: Indexing time reduced from hours to seconds for new commits

### Testing & Documentation

- **ADDED**: Performance benchmarks (`tests/codeintel_rev/benchmarks/test_git_performance.py`, `test_vllm_performance.py`, `test_async_adapters.py`)
- **ADDED**: Load tests simulating 100 concurrent requests (`tests/codeintel_rev/load/test_concurrent_search.py`)
- **MODIFIED**: Existing adapter tests to verify async behavior (use `pytest-asyncio`)
- **ADDED**: Migration guide for deployments with heavy indexing workloads
- **MODIFIED**: README with performance tuning recommendations

## Impact

### Specs
- **MODIFIED**: `codeintel-git-operations` capability spec to include GitPython requirements
- **MODIFIED**: `codeintel-embedding` capability spec to document async variant and client lifecycle
- **NEW**: `codeintel-performance` capability spec defining latency SLOs and concurrency targets

### Code
- **Core**: `git_client.py` (new), `vllm_client.py` (modified), `faiss_manager.py` (modified)
- **Adapters**: `history.py`, `semantic.py`, `files.py` (all converted to async)
- **Indexing**: `index_all.py` (add `--incremental` flag), `FAISSManager` (adaptive + incremental)
- **Context**: `ApplicationContext` (add VLLMClient cleanup in lifespan shutdown)
- **Tests**: 8 new benchmark/load test files

### Data Contracts
- **VLLMClient**: No breaking changes (async variant is additive)
- **GitClient**: New typed API (replaces subprocess, backward compatible via adapter signature)
- **FAISSManager**: No API changes (parameter selection is internal)

### Performance Targets
- **Git Operations**: <50ms p95 latency (vs 100-200ms baseline)
- **Embedding Batches**: <30ms overhead per batch (vs 40-50ms baseline)
- **Concurrent Requests**: Support 100+ concurrent semantic searches (vs ~20 baseline)
- **FAISS Training**: <10 seconds for 10K vectors (vs 60+ seconds baseline)
- **Incremental Indexing**: <60 seconds to add 1K new chunks (vs full rebuild)

### Rollout / Dependencies
- **Phase 1 & 2 Complete**: Requires `ApplicationContext` and scope management infrastructure
- **Backward Compatible**: All API signatures preserved (async is internal or additive)
- **New Dependencies**: `GitPython>=3.1.43` (pure Python, no C dependencies)
- **Optional**: `pygit2` as alternative to GitPython (faster but requires libgit2)
- **Incremental Deployment**: Performance optimizations can be deployed per-adapter (gradual rollout)

## Success Criteria

1. **Latency Reduction**: Git operations <50ms p95, embedding batches <30ms overhead (measured via Prometheus)
2. **Concurrency**: Server handles 100 concurrent semantic search requests without thread exhaustion
3. **Adaptive Indexing**: FAISS training completes in <10s for 10K vectors, <60s for 100K vectors
4. **Incremental Updates**: Adding 1K new chunks takes <60s (vs hours for full rebuild)
5. **Zero Regressions**: All existing tests pass, no degradation in search quality (recall, precision)
6. **Zero Errors**: Pyright/pyrefly/ruff clean, 100% backward compatibility

