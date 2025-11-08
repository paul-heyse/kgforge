# Capability: codeintel-performance

**Status**: Draft  
**Version**: 1.0.0  
**Owner**: CodeIntel Team  
**Last Updated**: 2025-11-08

## Purpose

Define performance requirements and optimization contracts for the CodeIntel MCP server, covering Git operations, HTTP connection management, async I/O, adaptive FAISS indexing, and incremental index updates.

## Scope

This capability covers:
- Git operations latency and reliability (GitPython integration)
- HTTP connection pooling and lifecycle management
- Async I/O patterns for concurrent request handling
- Adaptive FAISS index selection based on corpus size
- Incremental index updates without full rebuilds

Out of scope (future capabilities):
- Distributed indexing across multiple nodes
- Real-time index updates (streaming)
- Advanced caching strategies (Redis, memcached)

## Requirements

### FR-PERF-001: Git Operations Latency

**Priority**: MUST  
**Added**: 2025-11-08

Git operations (blame, history) MUST complete within performance targets.

**Acceptance Criteria**:
- `git blame` (10 lines): p95 latency < 50ms
- `git log` (50 commits): p95 latency < 40ms
- Operations use typed Python API (GitPython) instead of subprocess
- Unicode/locale handling automatic (no encoding errors)

**Verification**:
- Performance benchmark: `test_git_performance.py`
- Integration test with real Git repo

---

### FR-PERF-002: HTTP Connection Pooling

**Priority**: MUST  
**Added**: 2025-11-08

VLLMClient MUST reuse HTTP connections across embedding batches.

**Acceptance Criteria**:
- Single `httpx.Client` instance created in `__init__`
- Client reused for all `embed_batch` calls
- Connection limits configured (`max_connections=100`, `max_keepalive_connections=20`)
- `close()` method available for cleanup
- Embedding overhead < 30ms p95

**Verification**:
- Unit test verifying client reuse
- Performance benchmark comparing connection overhead

---

### FR-PERF-003: Async I/O Support

**Priority**: MUST  
**Added**: 2025-11-08

Heavy I/O operations MUST support async execution to prevent event loop blocking.

**Acceptance Criteria**:
- `list_paths`, `blame_range`, `file_history` converted to `async def`
- Blocking operations wrapped in `asyncio.to_thread`
- Server handles 100+ concurrent requests without thread exhaustion
- p95 latency unchanged or improved under load

**Verification**:
- Load test with 100 concurrent requests
- Benchmark comparing sync vs async under load

---

### FR-PERF-004: Adaptive FAISS Indexing

**Priority**: MUST  
**Added**: 2025-11-08

FAISS index type MUST be selected based on corpus size for optimal performance.

**Acceptance Criteria**:
- Small corpus (<5K vectors): `IndexFlatIP` (exact search, no training)
- Medium corpus (5K-50K): `IVFFlat` with dynamic `nlist = sqrt(n)`
- Large corpus (>50K): `IVF-PQ` with dynamic `nlist`
- Training time targets:
  - Small: <10s
  - Medium: <60s
  - Large: <180s (similar to fixed IVF-PQ)
- Search recall unchanged (>95%)

**Verification**:
- Unit tests for each corpus size
- Performance benchmarks measuring training time

---

### FR-PERF-005: Incremental Index Updates

**Priority**: SHOULD  
**Added**: 2025-11-08

System SHOULD support adding new chunks without full index rebuild.

**Acceptance Criteria**:
- Dual-index architecture: primary IVF-PQ + secondary flat
- `update_index(new_vectors, new_ids)` adds to secondary index
- Search queries both indexes, merges results by score
- Adding 1K chunks: <60s (vs hours for full rebuild)
- `merge_indexes()` available for periodic rebuild
- Dual-index search overhead: <10ms additional latency

**Verification**:
- Integration test for incremental workflow
- Performance benchmark for update speed

---

### FR-PERF-006: Resource Cleanup

**Priority**: MUST  
**Added**: 2025-11-08

Long-lived HTTP clients and Git repositories MUST be cleaned up during shutdown.

**Acceptance Criteria**:
- `VLLMClient.close()` closes HTTP clients
- `ApplicationContext` calls `vllm_client.close()` in lifespan shutdown
- No resource leaks (measured via health checks)

**Verification**:
- Unit test verifying cleanup
- Health check monitoring open connections

---

### FR-PERF-007: GitPython Error Handling

**Priority**: MUST  
**Added**: 2025-11-08

Git operations MUST provide typed exceptions for error conditions.

**Acceptance Criteria**:
- File not found: raises `FileNotFoundError`
- Git command error: raises `GitCommandError` with details
- Unicode/encoding errors handled automatically
- Error messages structured (no raw subprocess output)

**Verification**:
- Unit tests for error conditions
- Integration tests with edge cases (non-ASCII filenames, etc.)

---

### FR-PERF-008: Async Compatibility

**Priority**: MUST  
**Added**: 2025-11-08

All async adapters MUST work with FastMCP 2.3.2+.

**Acceptance Criteria**:
- Async tool functions callable from MCP server
- No event loop errors under concurrent load
- `pytest-asyncio` used for async tests

**Verification**:
- Integration test calling async tools via MCP
- Load test with concurrent async calls

---

### FR-PERF-009: Memory Efficiency

**Priority**: SHOULD  
**Added**: 2025-11-08

Adaptive indexing SHOULD reduce memory usage for small/medium corpora.

**Acceptance Criteria**:
- Small corpus: Flat index uses ~50% less memory than IVF-PQ
- Medium corpus: IVFFlat uses ~30% less memory than IVF-PQ
- `estimate_memory_usage()` method provides accurate estimates (±20%)

**Verification**:
- Unit test comparing memory usage
- Load test measuring actual memory consumption

---

### FR-PERF-010: Incremental Index Persistence

**Priority**: SHOULD  
**Added**: 2025-11-08

Secondary index SHOULD be persisted to disk for crash recovery.

**Acceptance Criteria**:
- `save_secondary_index()` saves secondary index separately
- `load_secondary_index()` loads secondary on startup
- Server restart preserves incremental updates

**Verification**:
- Integration test for save/load workflow
- Crash recovery test (kill server, restart, verify secondary loaded)

---

## Performance Targets

### Latency Targets (p95)

| Operation | Baseline | Target | Improvement |
|-----------|----------|--------|-------------|
| `git blame` (10 lines) | 200ms | <50ms | 75% |
| `git log` (50 commits) | 150ms | <40ms | 73% |
| `embed_batch` overhead | 45ms | <30ms | 33% |
| `semantic_search` | 350ms | <150ms | 57% |
| `list_paths` (1000 files) | 120ms | <50ms | 58% |

### Concurrency Targets

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Concurrent requests | 20-25 | 100+ | 5x |
| Thread pool usage | 100% | <50% | 2x efficiency |

### Indexing Targets

| Corpus Size | Old Training Time | New Training Time | Improvement |
|-------------|-------------------|-------------------|-------------|
| 5K vectors | 20s | <10s | 2x |
| 50K vectors | 180s | <60s | 3x |
| 100K vectors | 300s | <180s | 1.7x |

---

## Data Contracts

### GitClient API

**GitBlameEntry** (TypedDict):
```python
{
    "line": int,           # Line number (1-indexed)
    "commit": str,         # Short SHA (8 chars)
    "author": str,         # Author name
    "date": str,           # ISO 8601 timestamp
    "message": str,        # Commit summary
}
```

**Commit History Entry**:
```python
{
    "sha": str,            # Short SHA (8 chars)
    "full_sha": str,       # Full SHA (40 chars)
    "author": str,         # Author name
    "email": str,          # Author email
    "date": str,           # ISO 8601 timestamp
    "message": str,        # Commit summary
}
```

---

### VLLMClient Lifecycle

**Initialization**:
```python
client = VLLMClient(config)
# HTTP client created, ready for batches
```

**Usage**:
```python
vectors = client.embed_batch(texts)
# Reuses connection pool
```

**Cleanup**:
```python
client.close()
# Closes HTTP connections
```

**Async Variant**:
```python
vectors = await client.embed_batch_async(texts)
# Uses async HTTP client
```

---

### FAISSManager Incremental API

**Incremental Update**:
```python
manager.update_index(new_vectors, new_ids)
# Adds to secondary flat index
```

**Dual-Index Search**:
```python
distances, ids = manager.search(query, k=50)
# Queries primary + secondary, merges results
```

**Periodic Merge**:
```python
manager.merge_indexes()
# Rebuilds primary with secondary vectors
# Clears secondary
```

---

## Observability

### Metrics

**Gauge**: `codeintel_vllm_http_connections_open`  
Description: Number of open HTTP connections to vLLM service.  
Labels: None  
Target: <100

**Histogram**: `codeintel_git_operation_duration_seconds`  
Description: Git operation latency.  
Labels: `operation` (blame, history)  
Target: p95 <0.05s

**Histogram**: `codeintel_faiss_training_duration_seconds`  
Description: FAISS index training time.  
Labels: `index_type` (flat, ivf_flat, ivf_pq)  
Target: <180s for large corpus

**Counter**: `codeintel_incremental_updates_total`  
Description: Total number of incremental index updates.  
Labels: None

**Gauge**: `codeintel_secondary_index_size`  
Description: Number of vectors in secondary index.  
Labels: None  
Target: <10K (merge if exceeded)

---

### Logs

**Event**: Git operation  
Level: DEBUG  
Message: `"Git operation completed"`  
Fields: `operation` (blame/history), `path`, `duration_ms`

**Event**: HTTP client cleanup  
Level: INFO  
Message: `"Closed VLLMClient HTTP connections"`  
Fields: `open_connections` (before close)

**Event**: FAISS index selection  
Level: INFO  
Message: `"Selected FAISS index type"`  
Fields: `index_type`, `n_vectors`, `nlist`, `estimated_memory_gb`

**Event**: Incremental update  
Level: INFO  
Message: `"Added vectors to secondary index"`  
Fields: `n_vectors`, `total_secondary_size`

**Event**: Index merge  
Level: INFO  
Message: `"Merged secondary index into primary"`  
Fields: `n_merged_vectors`, `merge_duration_s`

---

## Dependencies

- **GitPython**: >=3.1.43 (pure Python, no C dependencies)
- **httpx**: Already dependency (connection pooling built-in)
- **pytest-asyncio**: For async test support
- **Git binary**: Must be available on deployment targets

---

## Migration & Rollout

### Backward Compatibility

**Breaking Changes**: None (all API signatures preserved)

**Additive Changes**:
- `VLLMClient.close()` method (should be called but not required)
- `VLLMClient.embed_batch_async()` method (optional)
- Async adapter signatures (internal, backward compatible)

### Rollout Plan

**Stage 1**: Deploy Git optimization (Week 1)  
**Stage 2**: Deploy HTTP pooling + async I/O (Week 2)  
**Stage 3**: Deploy adaptive indexing (Week 2-3)  
**Stage 4**: Deploy incremental updates (Week 3)

### Client Migration

**No Migration Required**: All changes are internal optimizations.

**Optional**: Use `--incremental` flag for faster indexing of new commits.

---

## Testing Strategy

### Performance Benchmarks

- **Git operations**: Compare subprocess vs GitPython latency
- **HTTP pooling**: Measure connection overhead reduction
- **Async I/O**: Load test with 100 concurrent requests
- **Adaptive indexing**: Training time for small/medium/large corpora
- **Incremental updates**: Time to add 1K new chunks

### Integration Tests

- **Git integration**: Real repo with Unicode filenames, large history
- **Async adapters**: Concurrent MCP tool calls
- **Dual-index search**: Verify result correctness and merge logic
- **Incremental workflow**: End-to-end indexing, update, merge cycle

### Load Tests

- **Concurrent search**: 100 simultaneous semantic_search calls
- **Sustained load**: 1000 requests over 60 seconds
- **Memory usage**: Monitor during heavy indexing

---

## Glossary

**GitPython**: Pure Python library for Git operations (alternative to subprocess).  
**Connection Pooling**: Reusing HTTP connections across requests (HTTP/1.1 keep-alive).  
**asyncio.to_thread**: Run blocking operations in threadpool to prevent event loop blocking.  
**Adaptive Indexing**: Choosing FAISS index type based on corpus size.  
**Dual-Index**: Architecture with primary IVF-PQ + secondary flat for incremental updates.  
**Incremental Update**: Adding new chunks without full index rebuild.

