# Implementation Tasks - Phase 3: Performance Optimization

## Overview

This document provides an exhaustive breakdown of implementation tasks for Phase 3 performance optimization. Tasks are organized into 6 phases spanning 3 weeks (75 hours total).

**Phase Breakdown**:
- **Phase 3a**: Git Operations (Week 1) - 15 hours
- **Phase 3b**: HTTP Connection Pooling (Week 1) - 10 hours
- **Phase 3c**: Async I/O Conversion (Week 2) - 15 hours
- **Phase 3d**: Adaptive FAISS Indexing (Week 2) - 12 hours
- **Phase 3e**: Incremental Index Updates (Week 3) - 18 hours
- **Phase 3f**: Performance Testing & Documentation (Week 3) - 5 hours

## Phase 3a: Git Operations (Week 1)

### Task 1: Add GitPython Dependency
**File**: `pyproject.toml`

**Description**: Add GitPython as a required dependency for typed Git operations.

**Subtasks**:
1. Add `gitpython>=3.1.43` to `[project.dependencies]` in `pyproject.toml`
2. Run `uv sync` to install GitPython
3. Verify installation: `uv run python -c "import git; print(git.__version__)"`
4. Document GitPython version requirements in `README.md`
5. Add Git binary requirement check to `scripts/bootstrap.sh`:
   ```bash
   if ! command -v git &> /dev/null; then
       echo "Error: git binary not found. Please install git."
       exit 1
   fi
   ```

**Acceptance**:
- GitPython installed and importable
- Bootstrap script checks for git binary
- `uv sync` completes without errors

**Time Estimate**: 1 hour

---

### Task 2: Implement GitClient Core
**File**: `codeintel_rev/io/git_client.py` (NEW)

**Description**: Create typed wrapper around GitPython for blame and history operations.

**Subtasks**:
1. Create module with comprehensive docstring explaining GitPython wrapper rationale
2. Define `GitClient` dataclass:
   ```python
   @dataclass(slots=True)
   class GitClient:
       repo_path: Path
       _repo: git.Repo | None = field(default=None, init=False)
   ```
3. Implement lazy `repo` property:
   ```python
   @property
   def repo(self) -> git.Repo:
       if self._repo is None:
           self._repo = git.Repo(self.repo_path, search_parent_directories=True)
       return self._repo
   ```
4. Add module-level logger: `LOGGER = get_logger(__name__)`
5. Add type hints for all methods (strict mode compliant)
6. Add comprehensive NumPy docstrings with Examples sections

**Acceptance**:
- `GitClient` dataclass defined
- Lazy repo initialization works
- Pyright reports zero errors
- Docstrings present for all public APIs

**Time Estimate**: 2 hours

---

### Task 3: Implement GitClient.blame_range
**File**: `codeintel_rev/io/git_client.py`

**Description**: Implement typed blame operation using GitPython.

**Subtasks**:
1. Implement `blame_range` method signature:
   ```python
   def blame_range(
       self,
       path: str,
       start_line: int,
       end_line: int
   ) -> list[GitBlameEntry]:
   ```
2. Use GitPython's `blame_incremental` for line-by-line blame:
   ```python
   blame_iter = self.repo.blame_incremental(
       rev="HEAD",
       file=path,
       L=f"{start_line},{end_line}"
   )
   ```
3. Convert GitPython commit objects to `GitBlameEntry` dicts:
   ```python
   {
       "line": line_num,
       "commit": commit.hexsha[:8],
       "author": commit.author.name,
       "date": commit.authored_datetime.isoformat(),
       "message": commit.summary,
   }
   ```
4. Handle errors:
   - `GitCommandError` with "does not exist" → raise `FileNotFoundError`
   - Other `GitCommandError` → propagate
5. Add structured logging for blame operations
6. Write comprehensive docstring with Parameters, Returns, Raises, Examples

**Acceptance**:
- `blame_range` returns typed `GitBlameEntry` list
- FileNotFoundError raised for missing files
- Docstring includes runnable example
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 4: Implement GitClient.file_history
**File**: `codeintel_rev/io/git_client.py`

**Description**: Implement typed commit history using GitPython.

**Subtasks**:
1. Implement `file_history` method signature:
   ```python
   def file_history(
       self,
       path: str,
       limit: int = 50
   ) -> list[dict]:
   ```
2. Use GitPython's `iter_commits` for history traversal:
   ```python
   commits_iter = self.repo.iter_commits(
       rev="HEAD",
       paths=path,
       max_count=limit
   )
   ```
3. Convert commit objects to dicts:
   ```python
   {
       "sha": commit.hexsha[:8],
       "full_sha": commit.hexsha,
       "author": commit.author.name,
       "email": commit.author.email,
       "date": commit.authored_datetime.isoformat(),
       "message": commit.summary,
   }
   ```
4. Handle errors (same as blame_range)
5. Add structured logging
6. Write comprehensive docstring

**Acceptance**:
- `file_history` returns commit list
- Error handling matches blame_range
- Docstring complete
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 5: Implement AsyncGitClient
**File**: `codeintel_rev/io/git_client.py`

**Description**: Create async wrapper using asyncio.to_thread.

**Subtasks**:
1. Define `AsyncGitClient` class:
   ```python
   class AsyncGitClient:
       def __init__(self, git_client: GitClient) -> None:
           self._sync_client = git_client
   ```
2. Implement async `blame_range`:
   ```python
   async def blame_range(
       self, path: str, start_line: int, end_line: int
   ) -> list[GitBlameEntry]:
       return await asyncio.to_thread(
           self._sync_client.blame_range, path, start_line, end_line
       )
   ```
3. Implement async `file_history`:
   ```python
   async def file_history(
       self, path: str, limit: int = 50
   ) -> list[dict]:
       return await asyncio.to_thread(
           self._sync_client.file_history, path, limit
       )
   ```
4. Add comprehensive docstrings explaining asyncio.to_thread usage
5. Document why GitPython is sync (subprocess-based internally)

**Acceptance**:
- `AsyncGitClient` wraps sync client
- Async methods use `asyncio.to_thread`
- Docstrings explain async pattern
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 6: Add GitClient to ApplicationContext
**File**: `codeintel_rev/app/config_context.py`

**Description**: Initialize GitClient in ApplicationContext for dependency injection.

**Subtasks**:
1. Import `GitClient` from `codeintel_rev.io.git_client`
2. Add `git_client: GitClient` field to `ApplicationContext` dataclass
3. Add `async_git_client: AsyncGitClient` field
4. Update `ApplicationContext.create()`:
   ```python
   git_client = GitClient(repo_path=paths.repo_root)
   async_git_client = AsyncGitClient(git_client)
   ```
5. Update class docstring to document new fields
6. Add logging for GitClient initialization

**Acceptance**:
- `ApplicationContext` has `git_client` and `async_git_client` fields
- Clients initialized in `create()` method
- Docstrings updated
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 7: Update history.py Adapters
**File**: `codeintel_rev/mcp_server/adapters/history.py`

**Description**: Replace subprocess Git calls with GitClient.

**Subtasks**:
1. Remove subprocess imports (`run_subprocess`, `SubprocessError`, etc.)
2. Import `FileNotFoundError` from builtin exceptions (no longer from path_utils)
3. Update `blame_range` adapter:
   - Remove `_invoke_git` call
   - Replace with `context.git_client.blame_range(path, start_line, end_line)`
   - Update error handling: catch `FileNotFoundError`, `git.exc.GitCommandError`
   - Remove text parsing logic (`_parse_blame_porcelain`)
4. Update `file_history` adapter:
   - Remove `_invoke_git` call
   - Replace with `context.git_client.file_history(path, limit)`
   - Remove text parsing logic
5. Delete helper functions: `_invoke_git`, `_parse_blame_porcelain`, `_parse_blame_block`, `_is_blame_header`, `_to_iso_timestamp`
6. Update docstrings to reflect GitPython usage
7. Add logging for Git operations

**Acceptance**:
- Adapters use `GitClient` instead of subprocess
- All helper functions removed (code reduction)
- Error handling preserved
- Docstrings updated
- Pyright clean

**Time Estimate**: 3 hours

---

### Task 8: Write GitClient Unit Tests
**File**: `tests/codeintel_rev/test_git_client.py` (NEW)

**Description**: Comprehensive unit tests for GitClient.

**Subtasks**:
1. Create test fixtures:
   - Mock `git.Repo` with `unittest.mock`
   - Mock commit objects with author, date, message
2. Test `blame_range`:
   - Happy path: returns typed entries
   - File not found: raises `FileNotFoundError`
   - Git error: raises `GitCommandError`
   - Unicode handling: non-ASCII author names
3. Test `file_history`:
   - Happy path: returns commit list
   - Empty history: returns empty list
   - Limit parameter: respects max_count
4. Test lazy repo initialization:
   - Repo not created until first access
   - Subsequent calls reuse same repo
5. Test `AsyncGitClient`:
   - Async methods call sync client via `asyncio.to_thread`
   - Results match sync client
6. Use `@pytest.mark.parametrize` for edge cases
7. Aim for 100% code coverage

**Acceptance**:
- All unit tests pass
- 100% coverage for `git_client.py`
- Tests use mocking (no real Git operations)
- Pyright clean

**Time Estimate**: 3 hours

---

### Task 9: Write GitClient Integration Tests
**File**: `tests/codeintel_rev/integration/test_git_integration.py` (NEW)

**Description**: Integration tests with real Git repository.

**Subtasks**:
1. Create test fixture: initialize real Git repo in temp directory
2. Populate repo with test commits:
   - Add file with multiple lines
   - Create 5-10 commits with different authors
3. Test `blame_range` with real repo:
   - Verify commit SHAs match expected
   - Verify author names correct
   - Verify dates in ISO 8601 format
4. Test `file_history` with real repo:
   - Verify commit order (newest first)
   - Verify limit parameter works
5. Test error cases with real repo:
   - File not in repo
   - Invalid line range
6. Use `@pytest.mark.integration` marker
7. Clean up temp repo after tests

**Acceptance**:
- Integration tests pass with real Git repo
- Tests verify actual Git data
- Temp repo cleaned up
- Tests marked with `@pytest.mark.integration`

**Time Estimate**: 2 hours

---

## Phase 3b: HTTP Connection Pooling (Week 1)

### Task 10: Modify VLLMClient.__init__
**File**: `codeintel_rev/io/vllm_client.py`

**Description**: Initialize persistent HTTP client for connection pooling.

**Subtasks**:
1. Add `_client: httpx.Client` field to `VLLMClient`
2. Add `_async_client: httpx.AsyncClient | None` field (initially None)
3. Update `__init__`:
   ```python
   self._client = httpx.Client(
       timeout=config.timeout_s,
       limits=httpx.Limits(
           max_connections=100,
           max_keepalive_connections=20,
       )
   )
   ```
4. Document connection pool parameters in docstring
5. Add logging for client initialization

**Acceptance**:
- `_client` initialized in `__init__`
- Connection limits configured
- Docstrings updated
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 11: Update VLLMClient.embed_batch
**File**: `codeintel_rev/io/vllm_client.py`

**Description**: Use persistent client instead of creating fresh client.

**Subtasks**:
1. Remove `with httpx.Client(...) as client:` context manager
2. Use `self._client.post(...)` directly:
   ```python
   response = self._client.post(
       f"{self.config.base_url}/embeddings",
       content=payload,
       headers={"Content-Type": "application/json"},
   )
   ```
3. Update docstring to mention connection reuse
4. Add logging for batch embedding (include batch size)

**Acceptance**:
- `embed_batch` uses persistent client
- No context manager (client not closed)
- Docstring updated
- Pyright clean

**Time Estimate**: 30 minutes

---

### Task 12: Implement VLLMClient.close
**File**: `codeintel_rev/io/vllm_client.py`

**Description**: Add cleanup method for resource release.

**Subtasks**:
1. Implement `close()` method:
   ```python
   def close(self) -> None:
       """Close HTTP clients and release resources."""
       self._client.close()
       if self._async_client is not None:
           asyncio.run(self._async_client.aclose())
   ```
2. Implement async variant:
   ```python
   async def aclose(self) -> None:
       """Async close for async context managers."""
       self._client.close()
       if self._async_client is not None:
           await self._async_client.aclose()
   ```
3. Add comprehensive docstrings explaining when to call
4. Document that `close()` must be called in lifespan shutdown

**Acceptance**:
- `close()` and `aclose()` methods implemented
- Both clients closed properly
- Docstrings explain lifecycle
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 13: Implement VLLMClient.embed_batch_async
**File**: `codeintel_rev/io/vllm_client.py`

**Description**: Add async variant for concurrent embedding.

**Subtasks**:
1. Implement `embed_batch_async` method:
   ```python
   async def embed_batch_async(self, texts: Sequence[str]) -> np.ndarray:
       if self._async_client is None:
           self._async_client = httpx.AsyncClient(
               timeout=self.config.timeout_s,
               limits=httpx.Limits(
                   max_connections=100,
                   max_keepalive_connections=20,
               )
           )
       # ... encode request ...
       response = await self._async_client.post(...)
       # ... decode response ...
   ```
2. Lazy-initialize `_async_client` on first call
3. Add comprehensive docstring explaining async benefits
4. Add logging for async embedding

**Acceptance**:
- `embed_batch_async` implemented
- Async client lazy-initialized
- Docstring complete
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 14: Add VLLMClient Cleanup to Lifespan
**File**: `codeintel_rev/app/main.py`

**Description**: Call VLLMClient.close() during shutdown.

**Subtasks**:
1. Update `lifespan` function shutdown section:
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       context = ApplicationContext.create()
       # ... startup ...
       yield
       # Shutdown: close VLLMClient
       context.vllm_client.close()
   ```
2. Add logging for cleanup
3. Add comment explaining why cleanup is needed
4. Document in inline comment that close() prevents resource leaks

**Acceptance**:
- `close()` called in lifespan shutdown
- Logging confirms cleanup
- Comment documents rationale
- Pyright clean

**Time Estimate**: 30 minutes

---

### Task 15: Write VLLMClient Unit Tests
**File**: `tests/codeintel_rev/test_vllm_client.py`

**Description**: Unit tests for connection pooling.

**Subtasks**:
1. Test `__init__`:
   - Verify `_client` created
   - Verify connection limits set
2. Test `embed_batch` with mocked httpx:
   - Mock `_client.post()` response
   - Verify persistent client used (no new client creation)
3. Test `embed_batch_async`:
   - Mock async client response
   - Verify lazy initialization
4. Test `close()`:
   - Verify both clients closed
5. Test `aclose()`:
   - Verify async close works
6. Use `unittest.mock` to mock httpx clients

**Acceptance**:
- All unit tests pass
- Mocking verifies client reuse
- 100% coverage for new code
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 16: Benchmark HTTP Connection Overhead
**File**: `tests/codeintel_rev/benchmarks/test_vllm_performance.py` (NEW)

**Description**: Measure connection pooling improvement.

**Subtasks**:
1. Create benchmark comparing old vs new implementation:
   - Old: create new client per batch
   - New: reuse persistent client
2. Measure 100 batches (32 chunks each):
   - Time per batch
   - Total time
   - Connection setup overhead
3. Use `pytest-benchmark` plugin
4. Mock vLLM responses (focus on HTTP overhead)
5. Assert new implementation faster by ≥20%
6. Document results in benchmark report

**Acceptance**:
- Benchmark shows ≥20% improvement
- Results documented
- Test marked with `@pytest.mark.benchmark`

**Time Estimate**: 2 hours

---

## Phase 3c: Async I/O Conversion (Week 2)

### Task 17: Convert list_paths to Async
**File**: `codeintel_rev/mcp_server/adapters/files.py`

**Description**: Make list_paths async with threadpool offload.

**Subtasks**:
1. Rename current `list_paths` to `_list_paths_sync`
2. Create new async `list_paths`:
   ```python
   async def list_paths(
       context: ApplicationContext,
       path: str | None = None,
       include_globs: list[str] | None = None,
       exclude_globs: list[str] | None = None,
       max_results: int = 1000,
   ) -> dict:
       return await asyncio.to_thread(
           _list_paths_sync,
           context, path, include_globs, exclude_globs, max_results
       )
   ```
3. Update docstring to mention async pattern
4. Add logging for async operation
5. Update `__all__` if needed

**Acceptance**:
- `list_paths` is async
- Sync implementation in `_list_paths_sync`
- Docstring updated
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 18: Convert blame_range to Async
**File**: `codeintel_rev/mcp_server/adapters/history.py`

**Description**: Make blame_range async using AsyncGitClient.

**Subtasks**:
1. Change function signature to `async def`:
   ```python
   async def blame_range(
       context: ApplicationContext,
       path: str,
       start_line: int,
       end_line: int,
   ) -> dict:
   ```
2. Use `context.async_git_client` instead of `context.git_client`:
   ```python
   try:
       entries = await context.async_git_client.blame_range(
           path, start_line, end_line
       )
   except FileNotFoundError:
       return {"blame": [], "error": "File not found"}
   ```
3. Update docstring
4. Add logging

**Acceptance**:
- `blame_range` is async
- Uses `AsyncGitClient`
- Docstring updated
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 19: Convert file_history to Async
**File**: `codeintel_rev/mcp_server/adapters/history.py`

**Description**: Make file_history async using AsyncGitClient.

**Subtasks**:
1. Change function signature to `async def`
2. Use `context.async_git_client.file_history(...)`
3. Update docstring
4. Add logging

**Acceptance**:
- `file_history` is async
- Uses `AsyncGitClient`
- Docstring updated
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 20: Update MCP Server for Async Tools
**File**: `codeintel_rev/mcp_server/server.py`

**Description**: Ensure MCP tool decorators support async.

**Subtasks**:
1. Verify FastMCP version supports async tools (≥2.3.2)
2. Update tool decorators if needed (should work as-is)
3. Test async tool invocation:
   - Call async adapter from MCP tool
   - Verify event loop handling
4. Add logging for async tool calls

**Acceptance**:
- Async tools work with FastMCP
- No event loop errors
- All tools callable

**Time Estimate**: 1 hour

---

### Task 21: Update Adapter Tests for Async
**File**: `tests/codeintel_rev/test_files_adapter.py`, `test_history_adapter.py`

**Description**: Convert tests to use pytest-asyncio.

**Subtasks**:
1. Add `pytest-asyncio` dependency to `pyproject.toml`
2. Update test functions to `async def`:
   ```python
   @pytest.mark.asyncio
   async def test_list_paths_async():
       result = await list_paths(context, path="src")
       assert isinstance(result["items"], list)
   ```
3. Use `await` for adapter calls
4. Update fixtures if needed (async fixtures)
5. Verify all async tests pass

**Acceptance**:
- All tests updated to async
- Tests pass with `pytest-asyncio`
- No event loop errors

**Time Estimate**: 3 hours

---

### Task 22: Write Async Load Tests
**File**: `tests/codeintel_rev/load/test_concurrent_adapters.py` (NEW)

**Description**: Simulate 100 concurrent requests.

**Subtasks**:
1. Create load test fixture:
   - Spawn 100 concurrent async tasks
   - Each task calls `list_paths`, `blame_range`, or `semantic_search`
2. Measure concurrency:
   - Time to complete all requests
   - Peak memory usage
   - Thread pool usage
3. Compare to baseline (sync implementation):
   - Should handle 5x more concurrent requests
   - Latency should not degrade under load
4. Use `asyncio.gather` for concurrent execution
5. Mark with `@pytest.mark.load`

**Acceptance**:
- Load test passes with 100 concurrent requests
- No thread exhaustion
- Latency within targets

**Time Estimate**: 4 hours

---

### Task 23: Benchmark Async vs Sync Performance
**File**: `tests/codeintel_rev/benchmarks/test_async_adapters.py` (NEW)

**Description**: Measure async conversion benefits.

**Subtasks**:
1. Benchmark `list_paths`:
   - Single request: latency should be similar
   - 100 concurrent requests: async should be 5-10x faster
2. Benchmark `blame_range`:
   - Single request: similar latency
   - 100 concurrent: async faster
3. Use `pytest-benchmark` plugin
4. Document results showing concurrency improvement

**Acceptance**:
- Benchmarks show async benefits under load
- Single-request latency unchanged
- Results documented

**Time Estimate**: 3 hours

---

## Phase 3d: Adaptive FAISS Indexing (Week 2)

### Task 24: Implement Adaptive Index Selection
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Choose index type based on corpus size.

**Subtasks**:
1. Update `build_index` signature to remove hardcoded `index_string`:
   ```python
   def build_index(self, vectors: np.ndarray) -> None:
   ```
2. Add index selection logic:
   ```python
   n_vectors = len(vectors)
   if n_vectors < 5000:
       # Flat index
       cpu_index = faiss.IndexFlatIP(self.vec_dim)
       logger.info(f"Using IndexFlatIP for {n_vectors} vectors")
   elif n_vectors < 50000:
       # IVFFlat
       nlist = min(int(np.sqrt(n_vectors)), n_vectors // 39)
       nlist = max(nlist, 100)
       quantizer = faiss.IndexFlatIP(self.vec_dim)
       cpu_index = faiss.IndexIVFFlat(
           quantizer, self.vec_dim, nlist, faiss.METRIC_INNER_PRODUCT
       )
       cpu_index.train(vectors)
       logger.info(f"Using IVFFlat with nlist={nlist}")
   else:
       # IVF-PQ
       nlist = int(np.sqrt(n_vectors))
       nlist = max(nlist, 1024)
       index_string = f"OPQ64,IVF{nlist},PQ64"
       cpu_index = faiss.index_factory(
           self.vec_dim, index_string, faiss.METRIC_INNER_PRODUCT
       )
       cpu_index.train(vectors)
       logger.info(f"Using IVF-PQ with nlist={nlist}")
   ```
3. Update docstring to explain adaptive selection
4. Add structured logging with selected index type

**Acceptance**:
- Index selection based on corpus size
- Logging shows selected index type
- Docstring updated
- Pyright clean

**Time Estimate**: 3 hours

---

### Task 25: Add estimate_memory_usage Helper
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Estimate memory requirements for capacity planning.

**Subtasks**:
1. Implement `estimate_memory_usage` method:
   ```python
   def estimate_memory_usage(
       self,
       n_vectors: int
   ) -> dict[str, int]:
       """Estimate memory usage in bytes.
       
       Returns
       -------
       dict[str, int]
           Memory estimates: cpu_index, gpu_index, total.
       """
       vec_size = self.vec_dim * 4  # float32 = 4 bytes
       
       if n_vectors < 5000:
           cpu_mem = n_vectors * vec_size  # Flat index
       elif n_vectors < 50000:
           nlist = int(np.sqrt(n_vectors))
           cpu_mem = (nlist * vec_size) + (n_vectors * 8)  # IVFFlat
       else:
           nlist = int(np.sqrt(n_vectors))
           cpu_mem = (nlist * vec_size) + (n_vectors * 64)  # IVF-PQ (64 bytes/vector)
       
       gpu_mem = cpu_mem * 1.2  # GPU has ~20% overhead
       
       return {
           "cpu_index_bytes": cpu_mem,
           "gpu_index_bytes": gpu_mem,
           "total_bytes": cpu_mem + gpu_mem,
       }
   ```
2. Add comprehensive docstring with examples
3. Log estimated memory on index build

**Acceptance**:
- Memory estimation works for all index types
- Docstring includes examples
- Logging shows estimates
- Pyright clean

**Time Estimate**: 2 hours

---

### Task 26: Update index_all.py for Adaptive Indexing
**File**: `codeintel_rev/bin/index_all.py`

**Description**: Log index type selection during indexing.

**Subtasks**:
1. Update logging to show selected index type:
   ```python
   logger.info(f"Building index for {len(vectors)} vectors")
   faiss_mgr.build_index(vectors)
   # FAISSManager logs index type internally
   ```
2. Add memory estimate logging:
   ```python
   mem_est = faiss_mgr.estimate_memory_usage(len(vectors))
   logger.info(f"Estimated memory: {mem_est['total_bytes'] / 1e9:.2f} GB")
   ```
3. Document adaptive indexing in script docstring

**Acceptance**:
- Logging shows index type and memory estimate
- Script docstring updated
- No breaking changes

**Time Estimate**: 1 hour

---

### Task 27: Write Adaptive Indexing Tests
**File**: `tests/codeintel_rev/test_faiss_manager_adaptive.py` (NEW)

**Description**: Test index selection for different corpus sizes.

**Subtasks**:
1. Test small corpus (<5K):
   - Generate 1000 random vectors
   - Build index
   - Verify `IndexFlatIP` used (check index type)
2. Test medium corpus (5K-50K):
   - Generate 20K vectors
   - Build index
   - Verify `IVFFlat` used
   - Verify nlist calculated correctly
3. Test large corpus (>50K):
   - Generate 100K vectors
   - Build index
   - Verify `IVF-PQ` used
4. Test memory estimation:
   - Verify estimates are reasonable
   - Compare to actual memory usage
5. Use `@pytest.mark.parametrize` for different sizes

**Acceptance**:
- All index types tested
- Memory estimates within 20% of actual
- Tests pass
- Pyright clean

**Time Estimate**: 4 hours

---

### Task 28: Benchmark Adaptive Indexing
**File**: `tests/codeintel_rev/benchmarks/test_faiss_performance.py` (NEW)

**Description**: Measure training time improvement.

**Subtasks**:
1. Benchmark small corpus (5K vectors):
   - Old: Fixed IVF-PQ (slow)
   - New: Flat index (fast)
   - Assert ≥10x speedup
2. Benchmark medium corpus (50K vectors):
   - Old: Fixed IVF-PQ with nlist=8192
   - New: IVFFlat with dynamic nlist
   - Assert ≥2x speedup
3. Benchmark large corpus (100K vectors):
   - Compare training times
   - Should be similar (both use IVF-PQ)
4. Use `pytest-benchmark` plugin
5. Document results

**Acceptance**:
- Small corpus: ≥10x speedup
- Medium corpus: ≥2x speedup
- Large corpus: similar performance
- Results documented

**Time Estimate**: 2 hours

---

## Phase 3e: Incremental Index Updates (Week 3)

### Task 29: Add Secondary Index Fields
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Add fields for dual-index architecture.

**Subtasks**:
1. Add new fields to `__init__`:
   ```python
   self.secondary_index: faiss.Index | None = None
   self.secondary_gpu_index: faiss.Index | None = None
   self.incremental_ids: set[int] = set()
   ```
2. Update class docstring to document dual-index architecture
3. Add diagram in docstring showing primary + secondary indexes
4. Document that secondary index is optional (env var controlled)

**Acceptance**:
- New fields added
- Docstring updated with architecture diagram
- Pyright clean

**Time Estimate**: 1 hour

---

### Task 30: Implement update_index Method
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Add vectors to secondary flat index.

**Subtasks**:
1. Implement `update_index` method:
   ```python
   def update_index(
       self,
       new_vectors: np.ndarray,
       new_ids: np.ndarray
   ) -> None:
       """Add new vectors to secondary index.
       
       Fast operation (seconds) for incremental updates.
       Secondary index is flat (exact search, no training).
       """
       if self.secondary_index is None:
           flat_index = faiss.IndexFlatIP(self.vec_dim)
           self.secondary_index = faiss.IndexIDMap2(flat_index)
       
       vectors_norm = new_vectors.copy()
       faiss.normalize_L2(vectors_norm)
       self.secondary_index.add_with_ids(
           vectors_norm, new_ids.astype(np.int64)
       )
       
       self.incremental_ids.update(new_ids.tolist())
       logger.info(f"Added {len(new_ids)} vectors to secondary index")
   ```
2. Add comprehensive docstring
3. Add structured logging
4. Handle case where secondary already has IDs (skip duplicates)

**Acceptance**:
- `update_index` adds to secondary flat index
- Logging confirms addition
- Docstring complete
- Pyright clean

**Time Estimate**: 3 hours

---

### Task 31: Implement Dual-Index Search
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Search both indexes and merge results.

**Subtasks**:
1. Update `search` method to query both indexes:
   ```python
   def search(
       self, query: np.ndarray, k: int = 50, nprobe: int = 128
   ) -> tuple[np.ndarray, np.ndarray]:
       # Search primary
       primary_dists, primary_ids = self._search_primary(query, k, nprobe)
       
       # Search secondary (if exists)
       if self.secondary_index is not None:
           secondary_dists, secondary_ids = self._search_secondary(query, k)
           # Merge by score
           return self._merge_results(
               primary_dists, primary_ids,
               secondary_dists, secondary_ids,
               k
           )
       
       return primary_dists, primary_ids
   ```
2. Implement `_search_primary` helper:
   ```python
   def _search_primary(
       self, query: np.ndarray, k: int, nprobe: int
   ) -> tuple[np.ndarray, np.ndarray]:
       index = self._active_index()  # GPU or CPU
       index.nprobe = nprobe
       query_norm = query.copy().astype(np.float32)
       faiss.normalize_L2(query_norm)
       return index.search(query_norm, k)
   ```
3. Implement `_search_secondary` helper (similar)
4. Implement `_merge_results` helper:
   ```python
   def _merge_results(
       self,
       dists1: np.ndarray,
       ids1: np.ndarray,
       dists2: np.ndarray,
       ids2: np.ndarray,
       k: int
   ) -> tuple[np.ndarray, np.ndarray]:
       """Merge results from two indexes by score."""
       # Combine distances and IDs
       all_dists = np.concatenate([dists1, dists2], axis=1)
       all_ids = np.concatenate([ids1, ids2], axis=1)
       
       # Sort by distance (descending for inner product)
       sorted_indices = np.argsort(-all_dists, axis=1)
       
       # Take top k
       top_k_indices = sorted_indices[:, :k]
       merged_dists = np.take_along_axis(all_dists, top_k_indices, axis=1)
       merged_ids = np.take_along_axis(all_ids, top_k_indices, axis=1)
       
       return merged_dists, merged_ids
   ```
5. Add comprehensive docstrings for all helpers
6. Add logging for merged search

**Acceptance**:
- Dual-index search works correctly
- Results merged by score
- Docstrings complete
- Pyright clean

**Time Estimate**: 5 hours

---

### Task 32: Implement merge_indexes Method
**File**: `codeintel_rev/io/faiss_manager.py`

**Description**: Rebuild primary index with secondary vectors.

**Subtasks**:
1. Implement `merge_indexes` method:
   ```python
   def merge_indexes(self) -> None:
       """Merge secondary into primary (periodic rebuild).
       
       Rebuilds primary index including secondary vectors.
       Clears secondary after merge.
       """
       if self.secondary_index is None or len(self.incremental_ids) == 0:
           logger.info("No secondary index to merge")
           return
       
       # Extract vectors from both indexes
       primary_vectors, primary_ids = self._extract_all_vectors(self.cpu_index)
       secondary_vectors, secondary_ids = self._extract_all_vectors(
           self.secondary_index
       )
       
       # Combine
       all_vectors = np.vstack([primary_vectors, secondary_vectors])
       all_ids = np.concatenate([primary_ids, secondary_ids])
       
       # Rebuild primary
       self.build_index(all_vectors)
       self.add_vectors(all_vectors, all_ids)
       
       # Clear secondary
       self.secondary_index = None
       self.secondary_gpu_index = None
       self.incremental_ids.clear()
       
       logger.info(f"Merged {len(secondary_ids)} vectors into primary")
   ```
2. Implement `_extract_all_vectors` helper:
   ```python
   def _extract_all_vectors(
       self, index: faiss.Index
   ) -> tuple[np.ndarray, np.ndarray]:
       """Extract all vectors and IDs from index."""
       n_vectors = index.ntotal
       vectors = np.empty((n_vectors, self.vec_dim), dtype=np.float32)
       ids = np.empty(n_vectors, dtype=np.int64)
       
       for i in range(n_vectors):
           vectors[i] = index.reconstruct(i)
           ids[i] = index.id_map.at(i)
       
       return vectors, ids
   ```
3. Add comprehensive docstrings
4. Add structured logging

**Acceptance**:
- Merge rebuilds primary with secondary vectors
- Secondary cleared after merge
- Docstrings complete
- Pyright clean

**Time Estimate**: 4 hours

---

### Task 33: Add --incremental Flag to index_all.py
**File**: `codeintel_rev/bin/index_all.py`

**Description**: Support incremental indexing mode.

**Subtasks**:
1. Add `--incremental` CLI flag using `typer`:
   ```python
   @app.command()
   def main(
       incremental: bool = typer.Option(
           False,
           "--incremental",
           help="Add new chunks to existing index instead of rebuilding"
       )
   ):
   ```
2. Add logic for incremental mode:
   ```python
   if incremental:
       # Load existing index
       faiss_mgr.load_cpu_index()
       
       # Identify new chunks (not in existing index)
       existing_ids = set(...)  # Get from index
       new_chunks = [c for c in chunks if c.id not in existing_ids]
       
       # Add to secondary index
       new_vectors = embed_chunks(new_chunks)
       new_ids = np.array([c.id for c in new_chunks])
       faiss_mgr.update_index(new_vectors, new_ids)
       
       # Save both indexes
       faiss_mgr.save_cpu_index()
       faiss_mgr.save_secondary_index()  # NEW method
   else:
       # Full rebuild (existing logic)
       ...
   ```
3. Implement `save_secondary_index` and `load_secondary_index` in `FAISSManager`
4. Add logging for incremental mode
5. Document flag in CLI help

**Acceptance**:
- `--incremental` flag works
- New chunks added to secondary index
- Full rebuild still works (default)
- Logging shows mode

**Time Estimate**: 3 hours

---

### Task 34: Write Incremental Indexing Tests
**File**: `tests/codeintel_rev/test_faiss_incremental.py` (NEW)

**Description**: Test dual-index workflow.

**Subtasks**:
1. Test `update_index`:
   - Add vectors to secondary
   - Verify secondary index created
   - Verify IDs tracked
2. Test dual-index search:
   - Build primary with 1000 vectors
   - Add 100 vectors to secondary
   - Search: verify results from both indexes
3. Test merge:
   - Build primary + secondary
   - Merge
   - Verify primary contains all vectors
   - Verify secondary cleared
4. Test incremental workflow end-to-end:
   - Initial indexing
   - Add new chunks
   - Search
   - Merge
   - Search again
5. Use `@pytest.mark.parametrize` for different sizes

**Acceptance**:
- All incremental tests pass
- Dual-index search verified
- Merge verified
- End-to-end workflow works

**Time Estimate**: 5 hours

---

### Task 35: Benchmark Incremental Updates
**File**: `tests/codeintel_rev/benchmarks/test_incremental_performance.py` (NEW)

**Description**: Measure incremental update speed.

**Subtasks**:
1. Benchmark adding 1K new chunks:
   - Old: Full rebuild (hours)
   - New: Incremental update (seconds)
   - Assert <60s for incremental
2. Benchmark search latency with dual-index:
   - Measure overhead of querying two indexes
   - Should be <10ms additional latency
3. Benchmark merge operation:
   - Measure time to merge secondary into primary
   - Document rebuild time
4. Use `pytest-benchmark` plugin
5. Document results showing incremental benefits

**Acceptance**:
- Incremental updates <60s for 1K chunks
- Dual-index search overhead <10ms
- Results documented

**Time Estimate**: 3 hours

---

## Phase 3f: Performance Testing & Documentation (Week 3)

### Task 36: Write End-to-End Performance Tests
**File**: `tests/codeintel_rev/integration/test_performance_integration.py` (NEW)

**Description**: Measure complete system performance.

**Subtasks**:
1. Test full indexing pipeline:
   - 100K chunks
   - Measure total time
   - Verify adaptive indexing used
2. Test search performance:
   - 1000 queries
   - Measure p50, p95, p99 latencies
   - Compare to baseline
3. Test concurrent search:
   - 100 concurrent queries
   - Measure throughput
   - Verify no thread exhaustion
4. Test Git operations:
   - Blame + history for 100 files
   - Measure latencies
   - Compare to baseline
5. Mark with `@pytest.mark.integration` and `@pytest.mark.performance`

**Acceptance**:
- All performance targets met
- Tests document improvements
- Comparison to baseline included

**Time Estimate**: 3 hours

---

### Task 37: Update README with Performance Guide
**File**: `codeintel_rev/README.md`

**Description**: Document performance tuning recommendations.

**Subtasks**:
1. Add "Performance Tuning" section:
   - Explain adaptive indexing behavior
   - Document incremental update workflow
   - Provide memory sizing guidance
2. Add "HTTP Connection Pooling" section:
   - Explain connection limits
   - Document `close()` requirement
3. Add "Async Operations" section:
   - Explain async benefits
   - Document concurrency limits
4. Add troubleshooting:
   - High memory usage → adjust index parameters
   - Slow indexing → use incremental mode
   - Thread exhaustion → async already enabled

**Acceptance**:
- README has performance section
- Tuning guidance clear
- Troubleshooting covers common issues

**Time Estimate**: 2 hours

---

### Task 38: Create Migration Guide
**File**: `docs/migration/performance-optimization-v3.md` (NEW)

**Description**: Guide for upgrading to Phase 3.

**Subtasks**:
1. Document breaking changes:
   - `VLLMClient.close()` must be called (non-breaking but important)
   - GitPython dependency added (breaking if git binary missing)
2. Provide upgrade steps:
   - Step 1: Update dependencies (`uv sync`)
   - Step 2: Verify git binary present
   - Step 3: Deploy new code
   - Step 4: Re-index with adaptive parameters (optional)
   - Step 5: Monitor performance metrics
3. Document rollback plan:
   - If issues arise, rollback to previous version
   - Indexes are backward compatible
4. Add FAQ:
   - "Do I need to rebuild indexes?" → Optional but recommended
   - "Will search quality change?" → No, only performance improves

**Acceptance**:
- Migration guide complete
- Upgrade steps tested
- FAQ covers common questions

**Time Estimate**: 2 hours (already included in previous estimate)

---

## Summary

**Total Tasks**: 38  
**Total Estimated Time**: 75 hours (3 weeks at 25 hours/week)

**Task Breakdown by Phase**:
- Phase 3a (Git): 9 tasks, 15 hours
- Phase 3b (HTTP): 7 tasks, 10 hours
- Phase 3c (Async): 7 tasks, 15 hours
- Phase 3d (Adaptive): 5 tasks, 12 hours
- Phase 3e (Incremental): 7 tasks, 18 hours
- Phase 3f (Testing/Docs): 3 tasks, 5 hours

**Critical Path**:
- Git operations must complete before async conversion (Week 1)
- HTTP pooling independent (parallel with Git)
- Async conversion blocks on Git completion (Week 2 start)
- Adaptive indexing independent (parallel with async)
- Incremental indexing requires adaptive completion (Week 3)
- Testing/docs in parallel with incremental (Week 3)

**Dependencies**:
- Phases 1 & 2 must be complete (ApplicationContext, scope management)
- Git binary must be available on deployment targets
- vLLM service must support connection pooling (it does)
- FastMCP must support async tools (≥2.3.2)

