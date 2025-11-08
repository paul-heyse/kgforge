# Performance Optimization - Detailed Design

## Context

The CodeIntel MCP server was built with pragmatic choices favoring implementation speed: subprocess-based Git operations, per-request HTTP client creation, synchronous I/O, and fixed FAISS parameters. While this enabled rapid prototyping, production usage reveals performance bottlenecks that limit scalability and increase operational costs.

This proposal systematically addresses five performance dimensions: Git operations (50-80ms latency reduction), HTTP connection management (5-10ms per batch), async I/O (10-100x concurrency), adaptive indexing (10-100x faster training), and incremental updates (hours → seconds).

### Current Performance Baseline

**Measured via load testing (100 concurrent requests, 1M chunk corpus)**:
- `git blame` (10 lines): 120ms p50, 200ms p95
- `git log` (50 commits): 80ms p50, 150ms p95
- `embed_batch` (32 chunks): 45ms p50, 70ms p95 (including HTTP overhead)
- `semantic_search` (k=20): 180ms p50, 350ms p95
- `list_paths` (1000 files): 60ms p50, 120ms p95
- **Concurrent capacity**: 20-25 requests before thread exhaustion

**FAISS indexing baseline (100K chunks, 2560-dim)**:
- Training time: 180 seconds
- Adding vectors: 40 seconds
- Total indexing: 220 seconds
- Memory usage: 1.2GB (IVF-PQ)

### Performance Targets (Post-Optimization)

**Adapter Latency**:
- `git blame`: <50ms p95 (60% reduction)
- `git log`: <40ms p95 (73% reduction)
- `embed_batch` overhead: <30ms p95 (57% reduction)
- `semantic_search`: <150ms p95 (57% reduction)
- `list_paths`: <50ms p95 (58% reduction)

**Concurrency**:
- Support 100+ concurrent requests (5x improvement)
- No thread exhaustion under sustained load

**Indexing Performance**:
- Small corpus (<10K): <10s training (vs 20s)
- Medium corpus (10K-100K): <60s training (vs 180s)
- Incremental updates: <60s for 1K new chunks (vs full rebuild)

## Architecture

### 1. Git Operations → GitPython

**Problem**:
- Subprocess overhead: ~50-100ms per call (process spawn, exec, cleanup)
- Text parsing fragility: Regex patterns fail on non-ASCII characters, locale-specific date formats
- No structured error handling: Exit codes don't distinguish "file not found" vs "not a Git repo"

**Solution**: Wrap GitPython with typed API.

#### GitClient Design

```python
@dataclass(slots=True)
class GitClient:
    """Typed wrapper around GitPython for blame and history operations.
    
    Provides structured APIs that return typed dictionaries instead of
    parsing text output. Handles encoding/locale issues automatically.
    
    Attributes
    ----------
    repo : git.Repo
        GitPython Repo instance (lazy-initialized on first use).
    repo_path : Path
        Path to repository root.
    """
    
    repo_path: Path
    _repo: git.Repo | None = field(default=None, init=False)
    
    @property
    def repo(self) -> git.Repo:
        """Lazy-load Git repository."""
        if self._repo is None:
            self._repo = git.Repo(self.repo_path, search_parent_directories=True)
        return self._repo
    
    def blame_range(
        self,
        path: str,
        start_line: int,
        end_line: int
    ) -> list[GitBlameEntry]:
        """Get Git blame for line range.
        
        Uses GitPython's blame_incremental() for efficient line-by-line blame.
        Returns typed dictionaries instead of text output.
        
        Parameters
        ----------
        path : str
            File path relative to repo root.
        start_line : int
            Start line (1-indexed).
        end_line : int
            End line (1-indexed).
        
        Returns
        -------
        list[GitBlameEntry]
            Typed blame entries with commit, author, date, message.
        
        Raises
        ------
        FileNotFoundError
            If file doesn't exist in repository.
        GitCommandError
            If Git operation fails (not a repo, etc.).
        """
        try:
            blame_iter = self.repo.blame_incremental(
                rev="HEAD",
                file=path,
                L=f"{start_line},{end_line}"
            )
        except git.exc.GitCommandError as exc:
            if "does not exist" in str(exc):
                raise FileNotFoundError(f"File not found: {path}") from exc
            raise
        
        entries: list[GitBlameEntry] = []
        for commit, lines in blame_iter:
            for line_num in lines:
                if start_line <= line_num <= end_line:
                    entries.append({
                        "line": line_num,
                        "commit": commit.hexsha[:8],
                        "author": commit.author.name,
                        "date": commit.authored_datetime.isoformat(),
                        "message": commit.summary,
                    })
        
        return entries
    
    def file_history(
        self,
        path: str,
        limit: int = 50
    ) -> list[dict]:
        """Get commit history for file.
        
        Uses GitPython's iter_commits() for efficient history traversal.
        Returns typed dictionaries with commit metadata.
        
        Parameters
        ----------
        path : str
            File path relative to repo root.
        limit : int
            Maximum number of commits to return.
        
        Returns
        -------
        list[dict]
            Commit history entries with SHA, author, date, message.
        
        Raises
        ------
        FileNotFoundError
            If file doesn't exist in repository.
        GitCommandError
            If Git operation fails.
        """
        try:
            commits_iter = self.repo.iter_commits(
                rev="HEAD",
                paths=path,
                max_count=limit
            )
        except git.exc.GitCommandError as exc:
            if "does not exist" in str(exc):
                raise FileNotFoundError(f"File not found: {path}") from exc
            raise
        
        commits: list[dict] = []
        for commit in commits_iter:
            commits.append({
                "sha": commit.hexsha[:8],
                "full_sha": commit.hexsha,
                "author": commit.author.name,
                "email": commit.author.email,
                "date": commit.authored_datetime.isoformat(),
                "message": commit.summary,
            })
        
        return commits
```

**Benefits**:
- **50-80ms latency reduction**: No subprocess overhead
- **Structured data**: Typed dictionaries, no regex parsing
- **Error handling**: Specific exceptions (`FileNotFoundError`, `GitCommandError`)
- **Unicode-safe**: GitPython handles encoding automatically
- **Testable**: Mock `git.Repo` instead of subprocess

**Trade-offs**:
- **Dependency**: Adds `GitPython>=3.1.43` (~500KB, pure Python)
- **Memory**: Repo object cached per `ApplicationContext` (~1-5MB)
- **Compatibility**: GitPython requires `git` binary (check via `shutil.which("git")`)

#### Async Variant

```python
class AsyncGitClient:
    """Async wrapper around GitClient using asyncio.to_thread."""
    
    def __init__(self, git_client: GitClient) -> None:
        self._sync_client = git_client
    
    async def blame_range(
        self,
        path: str,
        start_line: int,
        end_line: int
    ) -> list[GitBlameEntry]:
        """Async blame (runs GitPython in threadpool)."""
        return await asyncio.to_thread(
            self._sync_client.blame_range,
            path, start_line, end_line
        )
    
    async def file_history(
        self,
        path: str,
        limit: int = 50
    ) -> list[dict]:
        """Async history (runs GitPython in threadpool)."""
        return await asyncio.to_thread(
            self._sync_client.file_history,
            path, limit
        )
```

**Why asyncio.to_thread**:
- GitPython is synchronous (calls subprocess internally)
- Running in threadpool prevents event loop blocking
- Allows concurrent Git operations without thread exhaustion

---

### 2. HTTP Client Pooling → Persistent httpx.Client

**Problem**:
- `VLLMClient.embed_batch` creates fresh `httpx.Client` for every batch
- TCP connection handshake: ~5-10ms per connection
- HTTP/1.1 keep-alive benefits lost
- Connection pool exhaustion under load

**Solution**: Initialize `httpx.Client` once, reuse across batches.

#### Modified VLLMClient

```python
class VLLMClient:
    """vLLM embedding client with connection pooling.
    
    Maintains persistent HTTP client for connection reuse. Client is created
    in __init__ and reused across all embed_batch calls. close() must be
    called during shutdown to clean up resources.
    
    Attributes
    ----------
    config : VLLMConfig
        vLLM configuration.
    _client : httpx.Client
        Persistent HTTP client with connection pooling.
    _async_client : httpx.AsyncClient | None
        Optional async client for concurrent requests.
    """
    
    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(EmbeddingResponse)
        
        # Create persistent HTTP client
        self._client = httpx.Client(
            timeout=config.timeout_s,
            limits=httpx.Limits(
                max_connections=100,  # Connection pool size
                max_keepalive_connections=20,  # Keep-alive pool
            )
        )
        self._async_client: httpx.AsyncClient | None = None
    
    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed batch using persistent HTTP client.
        
        Reuses self._client for connection pooling and HTTP/1.1 keep-alive.
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        request = EmbeddingRequest(input=list(texts), model=self.config.model)
        payload = self._encoder.encode(request)
        
        # Use persistent client (no context manager)
        response = self._client.post(
            f"{self.config.base_url}/embeddings",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        result = self._decoder.decode(response.content)
        sorted_data = sorted(result.data, key=lambda d: d.index)
        return np.array([d.embedding for d in sorted_data], dtype=np.float32)
    
    async def embed_batch_async(self, texts: Sequence[str]) -> np.ndarray:
        """Async variant for concurrent embedding generation.
        
        Uses separate async client with connection pooling. Useful for
        embedding multiple queries concurrently during semantic search.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.timeout_s,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                )
            )
        
        if not texts:
            return np.array([], dtype=np.float32)
        
        request = EmbeddingRequest(input=list(texts), model=self.config.model)
        payload = self._encoder.encode(request)
        
        response = await self._async_client.post(
            f"{self.config.base_url}/embeddings",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        result = self._decoder.decode(response.content)
        sorted_data = sorted(result.data, key=lambda d: d.index)
        return np.array([d.embedding for d in sorted_data], dtype=np.float32)
    
    def close(self) -> None:
        """Close HTTP clients and release resources.
        
        Must be called during application shutdown to avoid resource leaks.
        Typically called in FastAPI lifespan shutdown.
        """
        self._client.close()
        if self._async_client is not None:
            asyncio.run(self._async_client.aclose())
    
    async def aclose(self) -> None:
        """Async close for async context managers."""
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
```

**Integration with ApplicationContext**:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan with VLLMClient cleanup."""
    context = ApplicationContext.create()
    app.state.context = context
    
    # ... startup logic ...
    
    yield
    
    # Cleanup: close VLLMClient HTTP connections
    context.vllm_client.close()
```

**Benefits**:
- **5-10ms latency reduction**: No connection handshake overhead
- **HTTP/1.1 keep-alive**: Reuse TCP connections across requests
- **Connection pooling**: Limit concurrent connections to vLLM server
- **Async variant**: Enable concurrent embedding generation

**Trade-offs**:
- **Resource lifecycle**: Must call `close()` in lifespan shutdown
- **State management**: Client is mutable (connection pool state)
- **Error handling**: Connection failures now persistent (need retry logic)

---

### 3. Async I/O Conversion → asyncio.to_thread

**Problem**:
- Adapters run in FastAPI's threadpool (default 40 threads)
- Blocking I/O (file reads, Git operations) ties up threads
- Under load: thread exhaustion → request queuing → latency spikes

**Solution**: Convert adapters to `async def`, wrap blocking operations in `asyncio.to_thread`.

#### Async Adapter Pattern

**Before** (synchronous):
```python
def list_paths(
    context: ApplicationContext,
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files (blocking I/O)."""
    repo_root = context.paths.repo_root
    # ... directory traversal (blocking) ...
    for current_root, dirnames, filenames in os.walk(search_root):
        # ... process files ...
    return {"items": items, "total": len(items)}
```

**After** (asynchronous):
```python
async def list_paths(
    context: ApplicationContext,
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files (async with threadpool offload)."""
    # Wrap blocking operation in to_thread
    return await asyncio.to_thread(
        _list_paths_sync,
        context, path, include_globs, exclude_globs, max_results
    )

def _list_paths_sync(
    context: ApplicationContext,
    path: str | None,
    include_globs: list[str] | None,
    exclude_globs: list[str] | None,
    max_results: int,
) -> dict:
    """Synchronous implementation (runs in threadpool)."""
    repo_root = context.paths.repo_root
    # ... directory traversal (blocking) ...
    return {"items": items, "total": len(items)}
```

**Why Split Async/Sync**:
- Async function signature for FastAPI/FastMCP compatibility
- Synchronous implementation keeps logic unchanged (testable)
- `asyncio.to_thread` runs sync function in threadpool (non-blocking)

#### Adapters to Convert

1. **`list_paths`**: Directory traversal (I/O bound)
2. **`blame_range`**: Git operations (CPU + I/O bound)
3. **`file_history`**: Git operations (CPU + I/O bound)
4. **`search_text`**: Already async-friendly (subprocess is quick)
5. **`semantic_search`**: Already async (uses `asyncio.to_thread` for FAISS)

**Benefits**:
- **10-100x concurrency**: Event loop handles thousands of concurrent requests
- **No thread exhaustion**: Threadpool only used for CPU-bound work
- **Lower latency under load**: Requests don't queue waiting for threads

**Trade-offs**:
- **Complexity**: Async/sync split adds indirection
- **Testing**: Need `pytest-asyncio` for async tests
- **Compatibility**: FastMCP must support async tools (it does as of 2.3.2)

---

### 4. Adaptive FAISS Indexing → Dynamic Parameter Selection

**Problem**:
- Fixed `IVF-PQ` with `nlist=8192` optimized for large corpora (>1M vectors)
- Small repos (<10K vectors): Training takes 20-60s, overfits to few vectors
- Medium repos (10K-100K): Could use `IVFFlat` for better recall, faster training

**Solution**: Choose index type and parameters based on corpus size.

#### Index Selection Strategy

```python
def build_index(self, vectors: np.ndarray) -> None:
    """Build FAISS index with adaptive parameters.
    
    Chooses index type based on corpus size:
    - Small (<5K): Flat index (exact search, no training)
    - Medium (5K-50K): IVFFlat with dynamic nlist
    - Large (>50K): IVF-PQ with dynamic nlist
    
    Parameters
    ----------
    vectors : np.ndarray
        Training vectors of shape (n, vec_dim).
    """
    n_vectors = len(vectors)
    faiss.normalize_L2(vectors)
    
    if n_vectors < 5000:
        # Small corpus: use flat index (exact search)
        cpu_index = faiss.IndexFlatIP(self.vec_dim)
        logger.info(f"Using IndexFlatIP for {n_vectors} vectors (exact search)")
    
    elif n_vectors < 50000:
        # Medium corpus: use IVFFlat with dynamic nlist
        nlist = min(int(np.sqrt(n_vectors)), n_vectors // 39)
        nlist = max(nlist, 100)  # Minimum 100 clusters
        
        quantizer = faiss.IndexFlatIP(self.vec_dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.vec_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(vectors)
        logger.info(f"Using IVFFlat with nlist={nlist} for {n_vectors} vectors")
    
    else:
        # Large corpus: use IVF-PQ with dynamic nlist
        nlist = int(np.sqrt(n_vectors))
        nlist = max(nlist, 1024)  # Minimum 1024 clusters for PQ
        
        index_string = f"OPQ64,IVF{nlist},PQ64"
        cpu_index = faiss.index_factory(self.vec_dim, index_string, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(vectors)
        logger.info(f"Using IVF-PQ with nlist={nlist} for {n_vectors} vectors")
    
    self.cpu_index = faiss.IndexIDMap2(cpu_index)
```

**Parameter Selection Rules**:

| Corpus Size | Index Type | Parameters | Training Time | Recall |
|-------------|------------|------------|---------------|--------|
| < 5K | `IndexFlatIP` | None (exact) | ~0s | 100% |
| 5K-50K | `IVFFlat` | `nlist = sqrt(n)` | ~5-30s | ~98% |
| 50K-500K | `IVF-PQ` | `nlist = sqrt(n)`, `PQ64` | ~30-180s | ~95% |
| > 500K | `IVF-PQ` | `nlist = 4*sqrt(n)`, `PQ64` | ~180-600s | ~93% |

**Benefits**:
- **10-100x faster training**: Small repos use flat index (no training)
- **Better recall**: Medium repos use IVFFlat instead of over-clustered IVF-PQ
- **Memory efficiency**: Flat index uses less memory for small corpora

**Trade-offs**:
- **Complexity**: Index type varies per deployment (harder to reason about)
- **Migration**: Existing indexes need rebuild (one-time cost)
- **Documentation**: Must document index selection logic for ops teams

---

### 5. Incremental Index Updates → Dual-Index Architecture

**Problem**:
- `index_all.py` regenerates entire index (hours for large repos)
- New commits require full rebuild (wasteful, blocks new searches)
- No way to add new chunks without retraining

**Solution**: Maintain two indexes: primary IVF-PQ + secondary Flat for new chunks.

#### Dual-Index Architecture

```
┌────────────────────────────────────────────────────┐
│              FAISSManager (Dual-Index)             │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  Primary Index (IVF-PQ, trained)             │ │
│  │  - 1M existing chunks                         │ │
│  │  - Persisted to disk                          │ │
│  │  - Updated via periodic rebuild               │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  Secondary Index (Flat, incremental)         │ │
│  │  - New chunks since last rebuild             │ │
│  │  - Exact search (no training needed)         │ │
│  │  - Merged into primary during rebuild        │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  Search: Query both indexes, merge results   │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

#### Implementation

```python
class FAISSManager:
    """FAISS manager with incremental update support.
    
    Maintains primary index (IVF-PQ) and secondary index (Flat) for
    incremental updates. Search queries both indexes and merges results.
    """
    
    def __init__(self, ...) -> None:
        # Existing fields
        self.cpu_index: faiss.Index | None = None
        self.gpu_index: faiss.Index | None = None
        
        # NEW: Secondary index for incremental updates
        self.secondary_index: faiss.Index | None = None
        self.secondary_gpu_index: faiss.Index | None = None
        self.incremental_ids: set[int] = set()  # Track new IDs
    
    def update_index(
        self,
        new_vectors: np.ndarray,
        new_ids: np.ndarray
    ) -> None:
        """Add new vectors to secondary index.
        
        Adds new chunks to a flat index without retraining. Fast operation
        (seconds instead of hours). Secondary index is searched alongside
        primary index during queries.
        
        Parameters
        ----------
        new_vectors : np.ndarray
            New vectors to add, shape (n, vec_dim).
        new_ids : np.ndarray
            IDs for new vectors, shape (n,).
        """
        if self.secondary_index is None:
            # Create flat index for incremental updates
            flat_index = faiss.IndexFlatIP(self.vec_dim)
            self.secondary_index = faiss.IndexIDMap2(flat_index)
        
        # Normalize and add vectors
        vectors_norm = new_vectors.copy()
        faiss.normalize_L2(vectors_norm)
        self.secondary_index.add_with_ids(vectors_norm, new_ids.astype(np.int64))
        
        # Track new IDs for merge
        self.incremental_ids.update(new_ids.tolist())
        
        logger.info(f"Added {len(new_ids)} vectors to secondary index")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 50,
        nprobe: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search both primary and secondary indexes, merge results.
        
        Queries primary index (IVF-PQ) and secondary index (Flat) in parallel,
        then merges results by similarity score. Secondary index provides
        exact search for new chunks.
        """
        # Search primary index
        primary_results = self._search_primary(query, k, nprobe)
        
        # Search secondary index (if exists)
        if self.secondary_index is not None:
            secondary_results = self._search_secondary(query, k)
            # Merge results by score
            return self._merge_results(primary_results, secondary_results, k)
        
        return primary_results
    
    def merge_indexes(self) -> None:
        """Merge secondary index into primary (rebuild).
        
        Rebuilds primary index including vectors from secondary index.
        This is a periodic operation (e.g., nightly) to consolidate
        incremental updates into the primary index.
        
        After merge, secondary index is cleared and ready for new updates.
        """
        if self.secondary_index is None or len(self.incremental_ids) == 0:
            logger.info("No secondary index to merge")
            return
        
        # Extract vectors from both indexes
        primary_vectors = self._extract_vectors(self.cpu_index)
        secondary_vectors = self._extract_vectors(self.secondary_index)
        
        # Combine and rebuild
        all_vectors = np.vstack([primary_vectors, secondary_vectors])
        self.build_index(all_vectors)
        
        # Add vectors with IDs
        primary_ids = np.arange(len(primary_vectors))
        secondary_ids = np.array(list(self.incremental_ids))
        all_ids = np.concatenate([primary_ids, secondary_ids])
        self.add_vectors(all_vectors, all_ids)
        
        # Clear secondary index
        self.secondary_index = None
        self.secondary_gpu_index = None
        self.incremental_ids.clear()
        
        logger.info(f"Merged {len(secondary_ids)} vectors into primary index")
```

**Workflow**:

1. **Initial indexing**: Build primary IVF-PQ index (hours)
2. **New commits**: Add new chunks to secondary flat index (seconds)
3. **Search**: Query both indexes, merge results by score
4. **Periodic rebuild** (nightly): Merge secondary into primary, clear secondary

**Benefits**:
- **60s to add 1K chunks**: No retraining, direct add to flat index
- **No search downtime**: Both indexes are queryable during updates
- **Exact search for new chunks**: Secondary index is flat (100% recall)

**Trade-offs**:
- **Memory overhead**: Two indexes in memory (~20-30% increase)
- **Search latency**: Query two indexes instead of one (+5-10ms)
- **Complexity**: Must manage dual-index lifecycle and merging

---

## Migration Plan

### Phase 3a: Git Operations (Week 1)

**Tasks**:
1. Add `GitPython>=3.1.43` to dependencies
2. Implement `GitClient` with `blame_range` and `file_history`
3. Implement `AsyncGitClient` wrapper
4. Update `history.py` adapters to use `GitClient`
5. Add `GitClient` to `ApplicationContext`
6. Write unit tests (mock `git.Repo`)
7. Write integration tests (real Git repo)
8. Benchmark: measure latency reduction

**Acceptance**:
- `git blame` p95 < 50ms (vs 200ms baseline)
- `git log` p95 < 40ms (vs 150ms baseline)
- All tests pass (including encoding edge cases)

### Phase 3b: HTTP Connection Pooling (Week 1)

**Tasks**:
1. Modify `VLLMClient.__init__` to create persistent `httpx.Client`
2. Update `embed_batch` to use `self._client`
3. Implement `VLLMClient.close()` for cleanup
4. Add `context.vllm_client.close()` to lifespan shutdown
5. Implement `embed_batch_async` variant
6. Write unit tests (mock httpx responses)
7. Write integration tests (real vLLM service)
8. Benchmark: measure connection overhead reduction

**Acceptance**:
- Embedding overhead < 30ms p95 (vs 45ms baseline)
- HTTP keep-alive connections reused
- All tests pass

### Phase 3c: Async I/O Conversion (Week 2)

**Tasks**:
1. Convert `list_paths` to async (split async/sync)
2. Convert `blame_range` to async (use `AsyncGitClient`)
3. Convert `file_history` to async (use `AsyncGitClient`)
4. Update MCP server to handle async tools
5. Update tests to use `pytest-asyncio`
6. Load test: simulate 100 concurrent requests
7. Benchmark: measure concurrency improvement

**Acceptance**:
- Support 100+ concurrent requests without thread exhaustion
- p95 latency unchanged or improved under load
- All async tests pass

### Phase 3d: Adaptive FAISS Indexing (Week 2)

**Tasks**:
1. Implement adaptive index selection in `build_index`
2. Add `estimate_memory_usage()` helper
3. Update `index_all.py` to log index type selection
4. Write unit tests for each index type (small, medium, large)
5. Write integration tests with real vectors
6. Benchmark: measure training time reduction

**Acceptance**:
- Small corpus (<5K): training < 10s
- Medium corpus (10K-100K): training < 60s
- Search recall unchanged (>95%)

### Phase 3e: Incremental Index Updates (Week 3)

**Tasks**:
1. Implement dual-index architecture (`secondary_index`)
2. Implement `update_index()` for adding new vectors
3. Implement `merge_indexes()` for periodic rebuild
4. Update `search()` to query both indexes
5. Add `--incremental` flag to `index_all.py`
6. Write unit tests for dual-index search
7. Write integration tests for incremental workflow
8. Benchmark: measure incremental update time

**Acceptance**:
- Adding 1K chunks takes < 60s
- Search queries both indexes correctly
- Merge operation completes successfully

### Phase 3f: Performance Testing & Documentation (Week 3)

**Tasks**:
1. Write load tests for 100 concurrent requests
2. Write benchmarks for each optimization
3. Update README with performance tuning guide
4. Create architecture diagrams for dual-index
5. Write migration guide for existing deployments
6. Measure end-to-end performance improvement

**Acceptance**:
- All performance targets met
- Documentation complete
- Migration guide tested

---

## Risks & Mitigations

### Risk 1: GitPython Performance Regression

**Likelihood**: Low  
**Impact**: High (could regress latency)

**Mitigation**:
- Benchmark Git Python vs subprocess before deployment
- If GitPython is slower, use `pygit2` (faster, but requires libgit2)
- Fallback: Keep subprocess implementation as alternative

### Risk 2: HTTP Client Resource Leaks

**Likelihood**: Medium (if `close()` not called)

**Mitigation**:
- Add health check: monitor open connections
- Add Prometheus metric: `vllm_http_connections_open`
- Document `close()` requirement in README
- Add unit test verifying cleanup

### Risk 3: Dual-Index Memory Overhead

**Likelihood**: High (20-30% memory increase)

**Mitigation**:
- Make secondary index optional (env var)
- Document memory requirements
- Add memory monitoring (Prometheus gauge)
- Provide merge script for ops teams

### Risk 4: Async Conversion Breaks Compatibility

**Likelihood**: Low (FastMCP supports async)

**Mitigation**:
- Test with FastMCP 2.3.2+
- Keep sync implementations as fallback
- Document async requirements

