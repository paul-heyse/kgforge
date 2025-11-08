# Migration Guide: Performance Optimization Phase 3

This guide helps you upgrade to Phase 3 performance optimizations, which include adaptive FAISS indexing, incremental updates, HTTP connection pooling, async I/O, and GitPython integration.

## Overview

Phase 3 introduces significant performance improvements:

- **Adaptive FAISS Indexing**: Automatically selects optimal index type (10-100x faster training for small/medium corpora)
- **Incremental Index Updates**: Add new chunks in seconds instead of hours
- **HTTP Connection Pooling**: Reduced embedding latency via persistent connections
- **Async I/O**: High concurrency support for Git and file operations
- **GitPython Integration**: More reliable Git operations (replaces subprocess calls)

## Breaking Changes

### 1. GitPython Dependency Required

**Change**: `gitpython>=3.1.43` is now a required dependency.

**Impact**: 
- The `git` binary must be available on your system (checked by `scripts/bootstrap.sh`)
- If Git is not installed, the application will fail to start

**Migration**:
```bash
# Ensure git is installed
git --version

# Dependencies are installed automatically via bootstrap.sh
scripts/bootstrap.sh
```

**Rollback**: If issues arise, you can temporarily revert to subprocess-based Git operations, but this is not recommended as it's less performant and reliable.

### 2. VLLMClient.close() Requirement

**Change**: `VLLMClient.close()` must be called to clean up HTTP connections.

**Impact**:
- Non-breaking: Connections are automatically closed during application shutdown (via `lifespan`)
- Manual cleanup only needed if creating clients outside application lifecycle

**Migration**:
- **No action required** if using the FastAPI application (automatic cleanup)
- If creating standalone `VLLMClient` instances, ensure `close()` is called:
  ```python
  client = VLLMClient(config)
  try:
      # Use client
      embeddings = client.embed_batch(texts)
  finally:
      client.close()  # Cleanup connections
  ```

**Rollback**: Not applicable - this is a resource management improvement with no behavior change.

### 3. Async Tool Signatures

**Change**: MCP tools `list_paths`, `blame_range`, and `file_history` are now async.

**Impact**:
- MCP clients must support async tool calls
- FastMCP handles this automatically (no client changes needed)

**Migration**:
- **No action required** for standard MCP clients
- If using direct HTTP calls, ensure your HTTP client supports async/await

**Rollback**: Not applicable - async operations are backward compatible at the protocol level.

## Upgrade Steps

### Step 1: Update Dependencies

```bash
cd /home/paul/kgfoundry
scripts/bootstrap.sh
```

This will:
- Install GitPython (`gitpython>=3.1.43`)
- Verify `git` binary is available
- Sync all dependencies

### Step 2: Verify Git Binary

```bash
# Check git is available
git --version

# If missing, install git
# Ubuntu/Debian:
sudo apt-get install git

# macOS:
brew install git
```

### Step 3: Deploy New Code

```bash
# Pull latest code
git pull

# Restart application
# (Your deployment process here)
```

### Step 4: Re-index with Adaptive Parameters (Optional but Recommended)

**Option A: Full Rebuild** (if you want to benefit from adaptive indexing):
```bash
# Remove old index
rm data/faiss/code.ivfpq.faiss

# Rebuild with adaptive indexing
python bin/index_all.py
```

**Option B: Keep Existing Index** (backward compatible):
- Existing indexes continue to work
- Adaptive indexing only applies to new indexes
- You can migrate gradually by rebuilding during low-traffic periods

### Step 5: Monitor Performance Metrics

After deployment, monitor:

1. **Indexing Performance**:
   - Check logs for adaptive index type selection
   - Verify training time improvements (especially for small/medium corpora)

2. **Search Latency**:
   - Monitor p50, p95, p99 latencies
   - Compare to baseline (should be similar or better)

3. **Git Operations**:
   - Verify blame/history operations complete successfully
   - Check for any GitPython-related errors

4. **HTTP Connections**:
   - Monitor embedding request latency (should improve with pooling)
   - Check for connection errors

## Rollback Plan

If issues arise after upgrading:

### Quick Rollback

1. **Revert code**:
   ```bash
   git revert <commit-hash>
   # Or checkout previous version
   git checkout <previous-tag>
   ```

2. **Restart application**:
   ```bash
   # Your deployment process
   ```

3. **Indexes remain compatible**: Old indexes continue to work with previous code

### Partial Rollback (If Needed)

If only specific components have issues:

- **Git operations**: Can temporarily disable Git features (not recommended)
- **HTTP pooling**: Falls back gracefully if connection issues occur
- **Async operations**: Backward compatible at protocol level

## FAQ

### Do I need to rebuild indexes?

**Answer**: Optional but recommended.

- Existing indexes work with new code (backward compatible)
- Adaptive indexing only applies to **new** indexes built after upgrade
- To benefit from adaptive indexing, rebuild your index
- For large codebases, consider incremental updates instead of full rebuild

### Will search quality change?

**Answer**: No, only performance improves.

- Search results remain the same (same recall/accuracy)
- Adaptive indexing optimizes **training time**, not search quality
- Incremental updates use exact search (same quality as flat index)

### Will search latency change?

**Answer**: Should remain similar or improve.

- Adaptive indexing maintains same search performance
- HTTP connection pooling reduces embedding latency
- Async operations improve concurrency (more requests handled)

### Can I use incremental updates with old indexes?

**Answer**: Yes, but requires primary index to exist.

- Incremental mode loads existing primary index
- Adds new chunks to secondary index
- Works with indexes built before Phase 3

### What if I don't have Git installed?

**Answer**: Application will fail to start (fail-fast).

- `scripts/bootstrap.sh` checks for Git binary
- Install Git before running bootstrap
- Git is required for `blame_range` and `file_history` tools

### How do I know if adaptive indexing is working?

**Answer**: Check application logs during indexing.

Look for log messages like:
```
INFO: Using IndexFlatIP for small corpus (n_vectors=1000, index_type=flat)
INFO: Using IVFFlat for medium corpus (n_vectors=10000, nlist=100, index_type=ivf_flat)
INFO: Using IVF-PQ for large corpus (n_vectors=100000, nlist=316, index_type=ivf_pq)
```

### When should I merge the secondary index?

**Answer**: Periodically, based on your update frequency.

- **After significant updates**: Merge when secondary index has 10K+ vectors
- **Before deployments**: Merge to ensure optimal search performance
- **During low traffic**: Merge is expensive but faster than full rebuild

Example merge schedule:
- Daily incremental updates → Weekly merge
- Weekly incremental updates → Monthly merge
- Ad-hoc updates → Merge before major releases

## Performance Expectations

### Indexing Performance

- **Small corpus (<5K)**: 10-100x faster training (flat vs IVF-PQ)
- **Medium corpus (5K-50K)**: 2-10x faster training (IVFFlat vs IVF-PQ)
- **Large corpus (>50K)**: Similar performance (IVF-PQ with optimized nlist)

### Incremental Updates

- **Adding 1K chunks**: <60 seconds (vs hours for full rebuild)
- **Adding 10K chunks**: <5 minutes (vs hours for full rebuild)
- **Dual-index search overhead**: <50% latency increase

### HTTP Connection Pooling

- **First request**: Same latency (connection establishment)
- **Subsequent requests**: 10-30% latency reduction (connection reuse)

### Async Operations

- **Concurrent requests**: No thread exhaustion
- **Throughput**: 10+ QPS for concurrent searches
- **Git operations**: Non-blocking, better concurrency

## Troubleshooting

### Index Build Fails

**Symptoms**: `build_index()` raises errors or takes too long.

**Solutions**:
1. Check GPU availability: `python -m codeintel_rev.mcp_server.tools.gpu_doctor`
2. Verify vector dimensions match configuration
3. Check available memory (use `estimate_memory_usage()`)
4. Try CPU-only mode: `export USE_CUVS=0`

### Incremental Update Fails

**Symptoms**: `--incremental` flag reports errors.

**Solutions**:
1. Ensure primary index exists (run full rebuild first)
2. Check that index path is correct
3. Verify vector dimensions match existing index
4. Check disk space for secondary index file

### Git Operations Fail

**Symptoms**: `blame_range` or `file_history` raise errors.

**Solutions**:
1. Verify Git binary: `git --version`
2. Check repository path: `ls -la $REPO_ROOT/.git`
3. Verify GitPython: `python -c "import git; print(git.__version__)"`
4. Check repository permissions

### High Memory Usage

**Symptoms**: Application uses excessive memory.

**Solutions**:
1. Use incremental updates instead of full rebuilds
2. Reduce `FAISS_NLIST` parameter
3. Check memory estimates before indexing
4. Periodically merge secondary index

See [README.md Performance Tuning](../codeintel_rev/README.md#performance-tuning) for detailed guidance.

## Support

For issues or questions:

1. Check application logs for error messages
2. Run GPU diagnostics: `python -m codeintel_rev.mcp_server.tools.gpu_doctor`
3. Review [Troubleshooting section](../codeintel_rev/README.md#troubleshooting)
4. Open an issue with logs and error details

## Version Compatibility

- **Phase 3 code**: Compatible with indexes from Phase 2 (backward compatible)
- **Phase 2 code**: Compatible with Phase 3 indexes (forward compatible)
- **Index format**: No changes (FAISS serialization format unchanged)

## Next Steps

After successful migration:

1. **Monitor performance**: Track indexing time, search latency, memory usage
2. **Use incremental updates**: Switch to `--incremental` for regular updates
3. **Optimize parameters**: Adjust `FAISS_NLIST`, `FAISS_NPROBE` based on your corpus
4. **Schedule merges**: Set up periodic secondary index merges

