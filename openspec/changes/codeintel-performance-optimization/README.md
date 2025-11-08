# CodeIntel Performance Optimization (Phase 3)

**Status**: Ready for Implementation  
**Phase**: 3 of 4 (Performance & Scalability)  
**Est. Duration**: 3 weeks (75 hours)  
**Owner**: CodeIntel Team

## Overview

This change proposal implements **production-grade performance optimizations** across five critical dimensions, reducing latency by 50-80%, enabling 5x more concurrent requests, and cutting indexing time from hours to seconds for incremental updates.

### Problem Statement

The current implementation uses pragmatic but performance-limited approaches: subprocess Git calls (50-100ms overhead), fresh HTTP connections per batch (5-10ms overhead), blocking I/O (thread exhaustion under load), fixed FAISS parameters (suboptimal for small repos), and full index rebuilds (hours for new commits).

### Solution Summary

Replace subprocess calls with GitPython (typed APIs, 75% latency reduction), implement HTTP connection pooling (33% overhead reduction), convert to async I/O (5x concurrency), add adaptive FAISS indexing (10-100x faster training for small/medium repos), and enable incremental updates (seconds vs hours).

### Key Benefits

1. **Latency Reduction**: Git ops <50ms p95 (vs 200ms), embedding batches <30ms overhead (vs 45ms)
2. **Concurrency**: 100+ concurrent requests (vs 20-25), no thread exhaustion
3. **Adaptive Performance**: Training time scales with corpus size (10s for 5K vectors, 60s for 50K)
4. **Incremental Indexing**: Add 1K chunks in <60s (vs full rebuild taking hours)
5. **Resource Efficiency**: Connection pooling, memory optimizations, proper cleanup

## Document Structure

```
codeintel-performance-optimization/
├── README.md                       # This file: overview and navigation
├── proposal.md                     # Why, what changes, impact, success criteria
├── design.md                       # Detailed design with code examples
├── tasks.md                        # 38 implementation tasks (75 hours over 3 weeks)
├── specs/
│   └── codeintel-performance/
│       └── spec.md                 # Performance requirements and contracts
└── implementation/
    ├── git_client.py               # GitPython wrapper (reference implementation)
    └── README.md                   # Implementation notes (TBD)
```

## Quick Links

- **Start Here**: [proposal.md](./proposal.md) for high-level context
- **Deep Dive**: [design.md](./design.md) for architectural decisions and code examples
- **Implementation Plan**: [tasks.md](./tasks.md) for week-by-week breakdown (38 tasks)
- **Requirements**: [specs/codeintel-performance/spec.md](./specs/codeintel-performance/spec.md)
- **Code**: [implementation/](./implementation/) for reference implementations

## Key Optimizations

### 1. Git Operations → GitPython

**Problem**: Subprocess overhead (50-100ms), fragile text parsing, locale issues

**Solution**: Typed Python API using GitPython

**Benefits**:
- **75% latency reduction**: blame <50ms p95 (vs 200ms), log <40ms p95 (vs 150ms)
- **Structured data**: Typed dictionaries, no regex
- **Unicode-safe**: Automatic encoding handling
- **Testable**: Mock `git.Repo` instead of subprocess

**Example**:
```python
# Before (subprocess)
stdout = run_subprocess(["git", "blame", "-L", "10,20", "file.py"])
entries = parse_blame_output(stdout)  # Fragile regex parsing

# After (GitPython)
client = GitClient(repo_path=Path("."))
entries = client.blame_range("file.py", 10, 20)  # Typed return
```

---

### 2. HTTP Connection Pooling

**Problem**: Fresh httpx.Client per batch (5-10ms overhead), no keep-alive

**Solution**: Persistent client with connection pool

**Benefits**:
- **33% overhead reduction**: <30ms p95 (vs 45ms)
- **HTTP/1.1 keep-alive**: Reuse TCP connections
- **Connection limits**: Prevent server overload

**Example**:
```python
# Before (per-batch client)
def embed_batch(self, texts):
    with httpx.Client(timeout=...) as client:
        response = client.post(...)

# After (persistent client)
def __init__(self, config):
    self._client = httpx.Client(
        timeout=config.timeout_s,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )

def embed_batch(self, texts):
    response = self._client.post(...)  # Reuses connection

def close(self):
    self._client.close()  # Cleanup in lifespan shutdown
```

---

### 3. Async I/O Conversion

**Problem**: Blocking I/O ties up threads, exhaustion under load (20-25 concurrent max)

**Solution**: Convert adapters to `async def`, use `asyncio.to_thread`

**Benefits**:
- **5x concurrency**: 100+ concurrent requests
- **No thread exhaustion**: Event loop handles thousands
- **Lower latency under load**: No request queuing

**Example**:
```python
# Before (synchronous)
def list_paths(context, path, globs, max_results):
    # ... blocking I/O ...
    for root, dirs, files in os.walk(search_root):  # Blocks thread
        # ...
    return results

# After (asynchronous)
async def list_paths(context, path, globs, max_results):
    return await asyncio.to_thread(
        _list_paths_sync,  # Runs in threadpool
        context, path, globs, max_results
    )
```

---

### 4. Adaptive FAISS Indexing

**Problem**: Fixed IVF-PQ with nlist=8192 suboptimal for small/medium repos

**Solution**: Choose index type based on corpus size

**Benefits**:
- **Small (<5K)**: Flat index, 10x faster training (10s vs 20s)
- **Medium (5K-50K)**: IVFFlat, 3x faster (60s vs 180s)
- **Large (>50K)**: IVF-PQ, similar performance
- **Better recall**: Appropriate parameters for each size

**Example**:
```python
def build_index(self, vectors: np.ndarray):
    n_vectors = len(vectors)
    
    if n_vectors < 5000:
        # Small: Flat index (exact search, no training)
        cpu_index = faiss.IndexFlatIP(self.vec_dim)
        logger.info(f"Using IndexFlatIP for {n_vectors} vectors")
    
    elif n_vectors < 50000:
        # Medium: IVFFlat with dynamic nlist
        nlist = min(int(np.sqrt(n_vectors)), n_vectors // 39)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.vec_dim, nlist, ...)
        cpu_index.train(vectors)
        logger.info(f"Using IVFFlat with nlist={nlist}")
    
    else:
        # Large: IVF-PQ with dynamic nlist
        nlist = max(int(np.sqrt(n_vectors)), 1024)
        index_string = f"OPQ64,IVF{nlist},PQ64"
        cpu_index = faiss.index_factory(self.vec_dim, index_string, ...)
        cpu_index.train(vectors)
        logger.info(f"Using IVF-PQ with nlist={nlist}")
```

---

### 5. Incremental Index Updates

**Problem**: Full rebuild takes hours for new commits, blocks searches

**Solution**: Dual-index architecture (primary IVF-PQ + secondary Flat)

**Benefits**:
- **Seconds vs hours**: Add 1K chunks in <60s
- **No downtime**: Both indexes queryable during updates
- **Exact search**: Secondary index is flat (100% recall)

**Architecture**:
```
┌────────────────────────────────────────────────────────┐
│              FAISSManager (Dual-Index)                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Primary Index (IVF-PQ, trained)                 │ │
│  │  - 1M existing chunks                            │ │
│  │  - Persisted to disk                             │ │
│  │  - Updated via periodic rebuild                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Secondary Index (Flat, incremental)             │ │
│  │  - New chunks since last rebuild                 │ │
│  │  - Exact search (no training needed)             │ │
│  │  - Merged into primary during nightly rebuild    │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Search: Query both indexes, merge results       │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**Workflow**:
```python
# Initial indexing (hours)
faiss_mgr.build_index(all_vectors)
faiss_mgr.add_vectors(all_vectors, all_ids)
faiss_mgr.save_cpu_index()

# New commits (seconds)
new_vectors, new_ids = embed_new_chunks(new_commits)
faiss_mgr.update_index(new_vectors, new_ids)  # Fast!

# Search (queries both)
distances, ids = faiss_mgr.search(query, k=50)  # Auto-merges results

# Periodic rebuild (nightly)
faiss_mgr.merge_indexes()  # Consolidate secondary into primary
```

---

## Implementation Phases

### Phase 3a: Git Operations (Week 1) - 15 hours
- Implement `GitClient` with `blame_range` and `file_history`
- Implement `AsyncGitClient` wrapper
- Update `history.py` adapters
- Write unit and integration tests
- Benchmark latency improvement

**Deliverable**: Git operations use GitPython, 75% latency reduction

---

### Phase 3b: HTTP Connection Pooling (Week 1) - 10 hours
- Modify `VLLMClient` for persistent client
- Implement `close()` cleanup method
- Add `embed_batch_async` variant
- Update lifespan shutdown
- Benchmark connection overhead

**Deliverable**: HTTP connections pooled, 33% overhead reduction

---

### Phase 3c: Async I/O Conversion (Week 2) - 15 hours
- Convert `list_paths`, `blame_range`, `file_history` to async
- Update MCP server for async tools
- Convert tests to `pytest-asyncio`
- Write load tests (100 concurrent requests)
- Benchmark concurrency improvement

**Deliverable**: 100+ concurrent requests supported

---

### Phase 3d: Adaptive FAISS Indexing (Week 2) - 12 hours
- Implement adaptive index selection
- Add `estimate_memory_usage` helper
- Update `index_all.py` logging
- Write tests for each index type
- Benchmark training time

**Deliverable**: Training time scales with corpus size

---

### Phase 3e: Incremental Index Updates (Week 3) - 18 hours
- Implement dual-index architecture
- Add `update_index` method
- Implement dual-index search with merge
- Add `--incremental` flag to `index_all.py`
- Write incremental workflow tests
- Benchmark update speed

**Deliverable**: Add 1K chunks in <60s

---

### Phase 3f: Performance Testing & Documentation (Week 3) - 5 hours
- Write end-to-end performance tests
- Update README with tuning guide
- Create migration guide
- Document results

**Deliverable**: Documentation complete, ready for rollout

---

## Performance Targets

### Latency (p95)

| Operation | Baseline | Target | Improvement |
|-----------|----------|--------|-------------|
| `git blame` (10 lines) | 200ms | <50ms | **75%** ↓ |
| `git log` (50 commits) | 150ms | <40ms | **73%** ↓ |
| `embed_batch` overhead | 45ms | <30ms | **33%** ↓ |
| `semantic_search` | 350ms | <150ms | **57%** ↓ |
| `list_paths` (1K files) | 120ms | <50ms | **58%** ↓ |

### Concurrency

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Concurrent requests | 20-25 | 100+ | **5x** ↑ |
| Thread pool usage | 100% | <50% | **2x** efficiency |

### Indexing

| Corpus Size | Old Time | New Time | Improvement |
|-------------|----------|----------|-------------|
| 5K vectors | 20s | <10s | **2x** ↓ |
| 50K vectors | 180s | <60s | **3x** ↓ |
| 100K vectors | 300s | <180s | **1.7x** ↓ |
| Incremental (1K) | Hours | <60s | **100x+** ↓ |

---

## Testing Strategy

### Performance Benchmarks
- Git operations: subprocess vs GitPython latency
- HTTP pooling: connection overhead measurement
- Async I/O: load test with 100 concurrent requests
- Adaptive indexing: training time for small/medium/large
- Incremental updates: time to add 1K chunks

### Integration Tests
- Git integration: real repo with Unicode, large history
- Async adapters: concurrent MCP tool calls
- Dual-index search: verify correctness and merge logic
- Incremental workflow: end-to-end indexing → update → merge

### Load Tests
- 100 concurrent semantic_search requests
- Sustained load: 1000 requests over 60 seconds
- Memory monitoring during heavy indexing

---

## Rollout Plan

### Stage 1: Deploy Git + HTTP Optimizations (Week 1)
- Low risk: backward compatible
- Monitor latency metrics
- Rollback plan: revert to subprocess if issues

### Stage 2: Deploy Async I/O (Week 2)
- Medium risk: async conversion
- Monitor concurrency and thread usage
- Gradual rollout: one adapter at a time

### Stage 3: Deploy Adaptive Indexing (Week 2-3)
- Low risk: index selection is internal
- Indexes need one-time rebuild (optional)
- Monitor training time and recall

### Stage 4: Deploy Incremental Updates (Week 3)
- Medium risk: dual-index architecture
- Thorough testing before production
- Monitor search quality and latency

---

## Success Criteria

✅ **Latency**:
- Git operations <50ms p95 (75% reduction)
- Embedding batches <30ms overhead (33% reduction)
- Semantic search <150ms p95 (57% reduction)

✅ **Concurrency**:
- Server handles 100+ concurrent requests
- No thread exhaustion under sustained load

✅ **Indexing**:
- Small corpus training <10s
- Incremental updates <60s for 1K chunks

✅ **Quality**:
- Zero pyright/pyrefly/ruff errors
- All tests pass (100% backward compatibility)
- Search recall unchanged (>95%)

---

## Dependencies

- **GitPython**: >=3.1.43 (pure Python, ~500KB)
- **httpx**: Already dependency (built-in pooling)
- **pytest-asyncio**: For async test support
- **Git binary**: Must be available on deployment

---

## FAQ

### Q: Will this break existing deployments?
**A**: No. All changes are internal optimizations. API signatures preserved.

### Q: Do I need to rebuild indexes?
**A**: Optional but recommended. Adaptive indexing will choose better parameters for your corpus size.

### Q: How do I use incremental updates?
**A**: Run `index_all.py --incremental` to add new chunks without full rebuild. Merge periodically with `merge_indexes()`.

### Q: What if GitPython is slower than subprocess?
**A**: Unlikely based on benchmarks, but fallback to `pygit2` available (faster, requires libgit2).

### Q: Will search quality change?
**A**: No. Adaptive indexing maintains >95% recall. Incremental updates use exact search for new chunks.

### Q: How much memory overhead for dual-index?
**A**: ~20-30% increase. Secondary index is optional (controlled via env var).

---

## Contact & Support

- **Owner**: CodeIntel Team
- **Slack**: #codeintel-mcp
- **Issues**: GitHub Issues (label: `performance`)
- **Docs**: `docs/performance/optimization-guide.md` (after implementation)

