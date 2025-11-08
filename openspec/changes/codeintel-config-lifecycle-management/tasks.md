## 1. Foundation - Configuration Context Infrastructure

### 1.1 Create Configuration Context Module
- [ ] 1.1.1 Create `codeintel_rev/app/config_context.py` module file
  - [ ] Add module docstring explaining centralized configuration lifecycle
  - [ ] Add imports: `dataclasses`, `pathlib`, `typing`, settings, errors, logging
  - [ ] Add `from __future__ import annotations` at top

- [ ] 1.1.2 Implement `ResolvedPaths` dataclass
  ```python
  @dataclass(slots=True, frozen=True)
  class ResolvedPaths:
      """Canonicalized filesystem paths for runtime operations."""
      repo_root: Path
      data_dir: Path
      vectors_dir: Path
      faiss_index: Path
      duckdb_path: Path
      scip_index: Path
  ```
  - [ ] Add comprehensive docstring with field descriptions
  - [ ] Verify frozen=True for immutability
  - [ ] Add Examples section showing typical usage

- [ ] 1.1.3 Implement `resolve_application_paths()` function
  ```python
  def resolve_application_paths(settings: Settings) -> ResolvedPaths:
      """Resolve all configured paths to absolute paths."""
  ```
  - [ ] Load repo_root from settings
  - [ ] Validate repo_root exists and is directory (raise ConfigurationError if not)
  - [ ] Create helper `_resolve(path_str: str) -> Path` for relative→absolute conversion
  - [ ] Resolve all 6 paths (data_dir, vectors_dir, faiss_index, duckdb_path, scip_index)
  - [ ] Return ResolvedPaths instance
  - [ ] Add structured logging for path resolution
  - [ ] Add docstring with Parameters, Returns, Raises sections

- [ ] 1.1.4 Implement `ApplicationContext` dataclass
  ```python
  @dataclass(slots=True)
  class ApplicationContext:
      """Application-wide context holding all configuration and long-lived clients."""
      settings: Settings
      paths: ResolvedPaths
      vllm_client: VLLMClient
      faiss_manager: FAISSManager
      _faiss_lock: Lock = field(default_factory=Lock, init=False)
      _faiss_loaded: bool = field(default=False, init=False)
      _faiss_gpu_attempted: bool = field(default=False, init=False)
  ```
  - [ ] Add comprehensive class docstring
  - [ ] Document all attributes (public and private)
  - [ ] Add Notes section explaining lifecycle

- [ ] 1.1.5 Implement `ApplicationContext.create()` factory method
  ```python
  @classmethod
  def create(cls) -> ApplicationContext:
      """Create application context from environment variables."""
  ```
  - [ ] Call `load_settings()` once
  - [ ] Call `resolve_application_paths(settings)` with error handling
  - [ ] Create `VLLMClient(settings.vllm)`
  - [ ] Create `FAISSManager(...)` with all parameters from settings
  - [ ] Log successful creation with key paths
  - [ ] Return ApplicationContext instance
  - [ ] Add docstring with Returns and Raises sections

- [ ] 1.1.6 Implement `ApplicationContext.ensure_faiss_ready()` method
  ```python
  def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
      """Load FAISS index (once) and attempt GPU clone."""
  ```
  - [ ] Copy logic from old `ServiceContext.ensure_faiss_ready()`
  - [ ] Use `self._faiss_lock` for thread safety
  - [ ] Check if `self.paths.faiss_index` exists
  - [ ] Load CPU index on first call
  - [ ] Attempt GPU clone on first call (conditional)
  - [ ] Return (ready, limits, error) tuple
  - [ ] Add comprehensive docstring

- [ ] 1.1.7 Implement `ApplicationContext.open_catalog()` context manager
  ```python
  @contextmanager
  def open_catalog(self) -> Iterator[DuckDBCatalog]:
      """Yield a DuckDB catalog context manager."""
  ```
  - [ ] Create `DuckDBCatalog` instance with paths from context
  - [ ] Open catalog in try block
  - [ ] Yield catalog
  - [ ] Close catalog in finally block
  - [ ] Add docstring with Yields section

### 1.2 Create Readiness Probe Module
- [ ] 1.2.1 Create `codeintel_rev/app/readiness.py` module file
  - [ ] Add module docstring explaining readiness checks
  - [ ] Add imports: `asyncio`, `dataclasses`, `httpx`, `pathlib`, `typing`
  - [ ] Add `from __future__ import annotations` at top

- [ ] 1.2.2 Implement `CheckResult` dataclass
  ```python
  @dataclass(slots=True, frozen=True)
  class CheckResult:
      """Outcome of a single readiness check."""
      healthy: bool
      detail: str | None = None
  ```
  - [ ] Add `as_payload()` method returning dict
  - [ ] Add comprehensive docstring
  - [ ] Add Examples section

- [ ] 1.2.3 Implement `ReadinessProbe` class
  ```python
  class ReadinessProbe:
      """Manages readiness checks across core dependencies."""
  ```
  - [ ] Add `__init__(self, context: ApplicationContext)` storing context
  - [ ] Add `_lock: asyncio.Lock` field
  - [ ] Add `_last_checks: dict[str, CheckResult]` cache
  - [ ] Add comprehensive class docstring

- [ ] 1.2.4 Implement `ReadinessProbe.initialize()` async method
  ```python
  async def initialize(self) -> None:
      """Prime readiness state on application startup."""
  ```
  - [ ] Call `await self.refresh()`
  - [ ] Add docstring

- [ ] 1.2.5 Implement `ReadinessProbe.refresh()` async method
  ```python
  async def refresh(self) -> Mapping[str, CheckResult]:
      """Recompute readiness checks asynchronously."""
  ```
  - [ ] Run `_run_checks()` in thread pool via `asyncio.to_thread`
  - [ ] Lock and update `_last_checks` cache
  - [ ] Return checks dict
  - [ ] Add docstring with Returns section

- [ ] 1.2.6 Implement `ReadinessProbe._run_checks()` sync method
  ```python
  def _run_checks(self) -> dict[str, CheckResult]:
      """Execute all readiness checks synchronously."""
  ```
  - [ ] Check repo_root exists (call `_check_directory`)
  - [ ] Check data_dir exists (create if missing)
  - [ ] Check vectors_dir exists (create if missing)
  - [ ] Check faiss_index file exists (required)
  - [ ] Check duckdb_path file exists (required)
  - [ ] Check scip_index file exists (optional)
  - [ ] Check vLLM service reachable (call `_check_vllm_connection`)
  - [ ] Return dict of all CheckResults
  - [ ] Add comprehensive docstring

- [ ] 1.2.7 Implement `ReadinessProbe._check_directory()` static method
  ```python
  @staticmethod
  def _check_directory(path: Path, *, create: bool = False) -> CheckResult:
      """Ensure a directory exists (creating it if requested)."""
  ```
  - [ ] Try to create directory if `create=True`
  - [ ] Check if path is directory
  - [ ] Return CheckResult with healthy status and optional detail
  - [ ] Handle OSError gracefully
  - [ ] Add docstring with Parameters and Returns

- [ ] 1.2.8 Implement `ReadinessProbe._check_file()` static method
  ```python
  @staticmethod
  def _check_file(path: Path, *, description: str, optional: bool = False) -> CheckResult:
      """Validate existence of a filesystem resource."""
  ```
  - [ ] Check if file exists
  - [ ] If optional and missing, return healthy with detail
  - [ ] If required and missing, return unhealthy
  - [ ] Handle OSError gracefully
  - [ ] Add docstring

- [ ] 1.2.9 Implement `ReadinessProbe._check_vllm_connection()` method
  ```python
  def _check_vllm_connection(self) -> CheckResult:
      """Verify vLLM service is reachable with a lightweight health check."""
  ```
  - [ ] Validate vLLM URL format (scheme, netloc)
  - [ ] Try HTTP GET to `{vllm_url}/health` with 2s timeout
  - [ ] Return CheckResult based on response
  - [ ] Handle httpx.HTTPError gracefully
  - [ ] Add docstring

- [ ] 1.2.10 Implement `ReadinessProbe.shutdown()` async method
  ```python
  async def shutdown(self) -> None:
      """Clear readiness state on shutdown."""
  ```
  - [ ] Lock and clear `_last_checks` dict
  - [ ] Add docstring

### 1.3 Unit Tests for Configuration Context
- [ ] 1.3.1 Create `tests/codeintel_rev/test_config_context.py` test file
  - [ ] Add imports: `pytest`, `Path`, `os`, config modules
  - [ ] Add `from __future__ import annotations` at top

- [ ] 1.3.2 Implement test fixtures
  ```python
  @pytest.fixture
  def test_repo(tmp_path: Path, monkeypatch) -> Path:
      """Set up a minimal test repository environment."""
  ```
  - [ ] Create repo_root directory
  - [ ] Create data/vectors/faiss directory structure
  - [ ] Create empty index files (faiss, duckdb)
  - [ ] Monkeypatch environment variables
  - [ ] Return repo_root

- [ ] 1.3.3 Write path resolution tests
  - [ ] `test_resolve_application_paths_success` — valid repo root
  - [ ] `test_resolve_application_paths_missing_repo_root` — ConfigurationError
  - [ ] `test_resolve_application_paths_not_directory` — ConfigurationError
  - [ ] `test_resolve_application_paths_relative_conversion` — relative→absolute

- [ ] 1.3.4 Write ApplicationContext tests
  - [ ] `test_application_context_create` — all fields initialized
  - [ ] `test_application_context_create_invalid_config` — raises ConfigurationError
  - [ ] `test_application_context_ensure_faiss_ready` — lazy loading
  - [ ] `test_application_context_ensure_faiss_ready_cached` — caching works
  - [ ] `test_application_context_open_catalog` — context manager

- [ ] 1.3.5 Run tests and verify coverage
  ```bash
  uv run pytest tests/codeintel_rev/test_config_context.py -v --cov=codeintel_rev/app/config_context
  ```
  - [ ] Verify 95%+ line coverage
  - [ ] Fix any failing tests
  - [ ] Verify no Ruff/pyright/pyrefly errors

### 1.4 Unit Tests for Readiness Probe
- [ ] 1.4.1 Create `tests/codeintel_rev/test_readiness.py` test file
  - [ ] Add imports: `pytest`, `asyncio`, readiness module
  - [ ] Create `mock_context` fixture

- [ ] 1.4.2 Write CheckResult tests
  - [ ] `test_check_result_as_payload_healthy` — no detail
  - [ ] `test_check_result_as_payload_unhealthy` — with detail

- [ ] 1.4.3 Write ReadinessProbe tests
  - [ ] `test_readiness_probe_initialize` — calls refresh
  - [ ] `test_readiness_probe_all_healthy` — all checks pass
  - [ ] `test_readiness_probe_missing_faiss` — one check fails
  - [ ] `test_readiness_probe_vllm_unreachable` — network error handled
  - [ ] `test_readiness_probe_caching` — refresh updates state
  - [ ] `test_readiness_probe_shutdown` — clears state

- [ ] 1.4.4 Run tests and verify coverage
  ```bash
  uv run pytest tests/codeintel_rev/test_readiness.py -v --cov=codeintel_rev/app/readiness
  ```
  - [ ] Verify 95%+ line coverage
  - [ ] Fix any failing tests

## 2. FastAPI Integration

### 2.1 Update Application Main Module
- [ ] 2.1.1 Modify `codeintel_rev/app/main.py`
  - [ ] Add imports for `ApplicationContext`, `ReadinessProbe`, `ConfigurationError`
  - [ ] Add `asyncio` import for `to_thread`

- [ ] 2.1.2 Implement `_preload_faiss_index()` helper function
  ```python
  def _preload_faiss_index(context: ApplicationContext) -> bool:
      """Pre-load FAISS index during startup to avoid first-request latency."""
  ```
  - [ ] Try to load CPU index
  - [ ] Try to clone to GPU
  - [ ] Log success or degraded mode
  - [ ] Return True/False for success
  - [ ] Add comprehensive docstring

- [ ] 2.1.3 Refactor `lifespan()` function
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
      """Application lifespan manager with explicit configuration initialization."""
  ```
  - [ ] Wrap everything in try/except for ConfigurationError
  - [ ] Phase 1: Load configuration via `ApplicationContext.create()`
  - [ ] Phase 2: Initialize readiness probe
  - [ ] Phase 3: Optional FAISS pre-loading (if `FAISS_PRELOAD=1`)
  - [ ] Store context in `app.state.context`
  - [ ] Store readiness in `app.state.readiness`
  - [ ] Log all phases with structured logging
  - [ ] Re-raise ConfigurationError to prevent startup
  - [ ] Add shutdown logging in finally block
  - [ ] Update docstring with detailed startup sequence

- [ ] 2.1.4 Update `/readyz` endpoint to use ReadinessProbe
  ```python
  @app.get("/readyz")
  async def readyz(request: Request) -> JSONResponse:
      """Readiness check endpoint."""
  ```
  - [ ] Extract readiness from `request.app.state.readiness`
  - [ ] Call `await readiness.refresh()`
  - [ ] Build response payload from CheckResults
  - [ ] Return overall ready status
  - [ ] Update docstring

### 2.2 Update Settings Configuration
- [ ] 2.2.1 Modify `codeintel_rev/config/settings.py`
  - [ ] Add `faiss_preload: bool` field to `IndexConfig` dataclass (default: False)
  - [ ] Update `IndexConfig` docstring documenting new field
  - [ ] Add `FAISS_PRELOAD` to environment variable list in `load_settings()` docstring

- [ ] 2.2.2 Update `load_settings()` function
  ```python
  faiss_preload=os.environ.get("FAISS_PRELOAD", "0").lower() in {"1", "true", "yes"}
  ```
  - [ ] Add environment variable parsing for `FAISS_PRELOAD`
  - [ ] Document behavior in docstring

### 2.3 Integration Tests for Application Startup
- [ ] 2.3.1 Create `tests/codeintel_rev/test_app_lifespan.py` test file
  - [ ] Add imports: `pytest`, `FastAPI`, `TestClient`, app module
  - [ ] Create `test_repo` fixture (reuse from previous tests)

- [ ] 2.3.2 Write startup success tests
  - [ ] `test_app_startup_with_valid_config` — app starts successfully
  - [ ] `test_app_healthz_endpoint` — /healthz returns 200
  - [ ] `test_app_readyz_endpoint_healthy` — /readyz shows all checks pass

- [ ] 2.3.3 Write startup failure tests
  - [ ] `test_app_startup_fails_missing_faiss_index` — startup fails
  - [ ] `test_app_startup_fails_invalid_repo_root` — ConfigurationError raised
  - [ ] `test_app_readyz_shows_unhealthy_resources` — /readyz shows failures

- [ ] 2.3.4 Write FAISS pre-loading tests
  - [ ] `test_app_startup_with_preload_enabled` — FAISS loaded at startup
  - [ ] `test_app_startup_with_preload_disabled` — FAISS lazy-loaded

- [ ] 2.3.5 Run integration tests
  ```bash
  uv run pytest tests/codeintel_rev/test_app_lifespan.py -v
  ```
  - [ ] Fix any failing tests
  - [ ] Verify TestClient handles lifespan correctly

## 3. Adapter Refactoring

### 3.1 Refactor Files Adapter
- [ ] 3.1.1 Modify `codeintel_rev/mcp_server/adapters/files.py`
  - [ ] Add `ApplicationContext` import
  - [ ] Add `if TYPE_CHECKING: from codeintel_rev.app.config_context import ApplicationContext`

- [ ] 3.1.2 Update `set_scope()` function signature
  ```python
  def set_scope(context: ApplicationContext, scope: ScopeIn) -> dict:
      """Set query scope for subsequent operations."""
  ```
  - [ ] Add `context` parameter (first position)
  - [ ] Update docstring with Parameters section
  - [ ] Mark context as "currently unused, reserved for future multi-repo support"

- [ ] 3.1.3 Update `list_paths()` function signature
  ```python
  def list_paths(
      context: ApplicationContext,
      path: str | None = None,
      include_globs: list[str] | None = None,
      exclude_globs: list[str] | None = None,
      max_results: int = 1000,
  ) -> dict:
  ```
  - [ ] Add `context` parameter (first position)
  - [ ] Replace `settings = load_settings()` with `repo_root = context.paths.repo_root`
  - [ ] Remove manual path resolution logic
  - [ ] Update docstring with context parameter

- [ ] 3.1.4 Update `open_file()` function signature
  ```python
  def open_file(
      context: ApplicationContext,
      path: str,
      start_line: int | None = None,
      end_line: int | None = None,
  ) -> dict:
  ```
  - [ ] Add `context` parameter (first position)
  - [ ] Replace `settings = load_settings()` with `repo_root = context.paths.repo_root`
  - [ ] Update docstring

- [ ] 3.1.5 Remove unused imports
  - [ ] Remove `from codeintel_rev.config.settings import load_settings` if not needed elsewhere

- [ ] 3.1.6 Create adapter unit tests
  - [ ] Create `tests/codeintel_rev/adapters/test_files_adapter.py`
  - [ ] Write `test_list_paths_with_context`
  - [ ] Write `test_open_file_with_context`
  - [ ] Write `test_set_scope_with_context`
  - [ ] Use mocked `ApplicationContext` fixture

- [ ] 3.1.7 Run adapter tests
  ```bash
  uv run pytest tests/codeintel_rev/adapters/test_files_adapter.py -v
  uv run pyright codeintel_rev/mcp_server/adapters/files.py
  ```

### 3.2 Refactor History Adapter
- [ ] 3.2.1 Modify `codeintel_rev/mcp_server/adapters/history.py`
  - [ ] Add `ApplicationContext` import
  - [ ] Add TYPE_CHECKING import

- [ ] 3.2.2 Update `blame_range()` function signature
  ```python
  def blame_range(
      context: ApplicationContext,
      path: str,
      start_line: int,
      end_line: int,
  ) -> dict:
  ```
  - [ ] Add `context` parameter
  - [ ] Replace `load_settings()` with `context.paths.repo_root`
  - [ ] Update docstring

- [ ] 3.2.3 Update `file_history()` function signature
  ```python
  def file_history(
      context: ApplicationContext,
      path: str,
      limit: int = 50,
  ) -> dict:
  ```
  - [ ] Add `context` parameter
  - [ ] Replace `load_settings()` with `context.paths.repo_root`
  - [ ] Update docstring

- [ ] 3.2.4 Remove unused imports

- [ ] 3.2.5 Create adapter unit tests
  - [ ] Create `tests/codeintel_rev/adapters/test_history_adapter.py`
  - [ ] Write `test_blame_range_with_context`
  - [ ] Write `test_file_history_with_context`

- [ ] 3.2.6 Run adapter tests
  ```bash
  uv run pytest tests/codeintel_rev/adapters/test_history_adapter.py -v
  uv run pyright codeintel_rev/mcp_server/adapters/history.py
  ```

### 3.3 Refactor Text Search Adapter
- [ ] 3.3.1 Modify `codeintel_rev/mcp_server/adapters/text_search.py`
  - [ ] Add `ApplicationContext` import
  - [ ] Add TYPE_CHECKING import

- [ ] 3.3.2 Update `search_text()` function signature
  ```python
  def search_text(
      context: ApplicationContext,
      query: str,
      *,
      regex: bool = False,
      case_sensitive: bool = False,
      paths: Sequence[str] | None = None,
      max_results: int = 50,
  ) -> dict:
  ```
  - [ ] Add `context` parameter (first position)
  - [ ] Replace `settings = load_settings()` with `context.paths.repo_root`
  - [ ] Update docstring

- [ ] 3.3.3 Remove unused imports

- [ ] 3.3.4 Create adapter unit tests
  - [ ] Create `tests/codeintel_rev/adapters/test_text_search_adapter.py`
  - [ ] Write `test_search_text_with_context`
  - [ ] Mock subprocess calls for ripgrep

- [ ] 3.3.5 Run adapter tests
  ```bash
  uv run pytest tests/codeintel_rev/adapters/test_text_search_adapter.py -v
  uv run pyright codeintel_rev/mcp_server/adapters/text_search.py
  ```

### 3.4 Refactor Semantic Adapter
- [ ] 3.4.1 Modify `codeintel_rev/mcp_server/adapters/semantic.py`
  - [ ] Add `ApplicationContext` import
  - [ ] Add TYPE_CHECKING import

- [ ] 3.4.2 Update `semantic_search()` async function signature
  ```python
  async def semantic_search(
      context: ApplicationContext,
      query: str,
      limit: int = 20,
  ) -> AnswerEnvelope:
  ```
  - [ ] Add `context` parameter (first position)
  - [ ] Pass context to `_semantic_search_sync`
  - [ ] Update docstring

- [ ] 3.4.3 Update `_semantic_search_sync()` function signature
  ```python
  def _semantic_search_sync(
      context: ApplicationContext,
      query: str,
      limit: int,
  ) -> AnswerEnvelope:
  ```
  - [ ] Add `context` parameter
  - [ ] Replace `context = get_service_context()` with passed context
  - [ ] Use `context.settings`, `context.ensure_faiss_ready()`, etc.
  - [ ] Update docstring

- [ ] 3.4.4 Remove `get_service_context()` import

- [ ] 3.4.5 Create adapter unit tests
  - [ ] Create `tests/codeintel_rev/adapters/test_semantic_adapter.py`
  - [ ] Write `test_semantic_search_with_context`
  - [ ] Mock vLLM client and FAISS manager

- [ ] 3.4.6 Run adapter tests
  ```bash
  uv run pytest tests/codeintel_rev/adapters/test_semantic_adapter.py -v
  uv run pyright codeintel_rev/mcp_server/adapters/semantic.py
  ```

## 4. MCP Server Integration

### 4.1 Update MCP Server Tool Wrappers
- [ ] 4.1.1 Modify `codeintel_rev/mcp_server/server.py`
  - [ ] Add imports: `from fastapi import Request`
  - [ ] Add `ApplicationContext` import

- [ ] 4.1.2 Implement `_get_context()` helper function
  ```python
  def _get_context(request: Request) -> ApplicationContext:
      """Extract ApplicationContext from FastAPI state."""
      context: ApplicationContext | None = request.app.state.get("context")
      if context is None:
          raise RuntimeError("ApplicationContext not initialized in app state")
      return context
  ```
  - [ ] Add comprehensive docstring
  - [ ] Add error handling for missing context
  - [ ] Document when this can happen (startup failure)

- [ ] 4.1.3 Update `set_scope()` tool wrapper
  ```python
  @mcp.tool()
  def set_scope(request: Request, scope: ScopeIn) -> dict:
      """Set query scope for subsequent operations."""
      context = _get_context(request)
      return files_adapter.set_scope(context, scope)
  ```
  - [ ] Add `request: Request` parameter (first position)
  - [ ] Call `_get_context(request)`
  - [ ] Pass context to adapter function

- [ ] 4.1.4 Update `list_paths()` tool wrapper
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter
  - [ ] Update docstring if needed

- [ ] 4.1.5 Update `open_file()` tool wrapper
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter

- [ ] 4.1.6 Update `search_text()` tool wrapper
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter

- [ ] 4.1.7 Update `semantic_search()` tool wrapper
  ```python
  @mcp.tool()
  async def semantic_search(
      request: Request,
      query: str,
      limit: int = 20,
  ) -> AnswerEnvelope:
      """Semantic code search using embeddings."""
      context = _get_context(request)
      return await semantic_adapter.semantic_search(context, query, limit)
  ```
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter
  - [ ] Verify async/await chain works

- [ ] 4.1.8 Update `blame_range()` tool wrapper
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter

- [ ] 4.1.9 Update `file_history()` tool wrapper
  - [ ] Add `request: Request` parameter
  - [ ] Extract context and pass to adapter

- [ ] 4.1.10 Update `file_resource()` resource handler
  ```python
  @mcp.resource("file://{path}")
  def file_resource(request: Request, path: str) -> str:
      """Serve file content as resource."""
      context = _get_context(request)
      file_result = files_adapter.open_file(context, path)
      # ... rest of logic
  ```
  - [ ] Add `request: Request` parameter if not present
  - [ ] Extract context and pass to adapter

### 4.2 MCP Server Integration Tests
- [ ] 4.2.1 Create `tests/codeintel_rev/test_mcp_server.py` test file
  - [ ] Add imports: `TestClient`, `app`, all adapters

- [ ] 4.2.2 Write tool wrapper tests
  - [ ] `test_set_scope_endpoint` — calls adapter with context
  - [ ] `test_list_paths_endpoint` — calls adapter with context
  - [ ] `test_open_file_endpoint` — calls adapter with context
  - [ ] `test_search_text_endpoint` — calls adapter with context
  - [ ] `test_semantic_search_endpoint` — calls adapter with context
  - [ ] `test_blame_range_endpoint` — calls adapter with context
  - [ ] `test_file_history_endpoint` — calls adapter with context

- [ ] 4.2.3 Write error handling tests
  - [ ] `test_missing_context_raises_error` — RuntimeError if context not in state

- [ ] 4.2.4 Run MCP server tests
  ```bash
  uv run pytest tests/codeintel_rev/test_mcp_server.py -v
  ```

## 5. Cleanup and Documentation

### 5.1 Remove Deprecated Code
- [ ] 5.1.1 Delete `codeintel_rev/mcp_server/service_context.py`
  - [ ] Verify no imports remain in codebase: `git grep "from.*service_context import"`
  - [ ] Verify no imports remain: `git grep "get_service_context"`
  - [ ] Delete file: `rm codeintel_rev/mcp_server/service_context.py`

- [ ] 5.1.2 Remove unused imports from all adapters
  - [ ] Search for `from codeintel_rev.config.settings import load_settings` in adapters
  - [ ] Remove if present and unused

- [ ] 5.1.3 Run Ruff to clean up unused imports
  ```bash
  uv run ruff check --fix codeintel_rev/
  ```

### 5.2 Create Configuration Documentation
- [ ] 5.2.1 Create `codeintel_rev/docs/CONFIGURATION.md`
  - [ ] Section: "Configuration Management Overview"
  - [ ] Section: "Configuration Loading Sequence" (with diagram)
  - [ ] Section: "Key Environment Variables" (with table)
  - [ ] Section: "Configuration Best Practices"
  - [ ] Subsection: Development configuration
  - [ ] Subsection: Production configuration
  - [ ] Subsection: Kubernetes configuration
  - [ ] Section: "Configuration Lifecycle" (startup/runtime/shutdown)
  - [ ] Section: "Troubleshooting" (common config errors)

- [ ] 5.2.2 Update `codeintel_rev/README.md`
  - [ ] Update "Configuration" section to reference new `CONFIGURATION.md`
  - [ ] Add note about `FAISS_PRELOAD` environment variable
  - [ ] Update "Quick Start" section with new startup behavior

### 5.3 Full Quality Gates
- [ ] 5.3.1 Run Ruff format and check
  ```bash
  uv run ruff format codeintel_rev/ tests/codeintel_rev/
  uv run ruff check --fix codeintel_rev/ tests/codeintel_rev/
  ```
  - [ ] Fix any Ruff errors
  - [ ] Verify zero warnings

- [ ] 5.3.2 Run pyright type checking
  ```bash
  uv run pyright --warnings --pythonversion=3.13 codeintel_rev/ tests/codeintel_rev/
  ```
  - [ ] Fix any type errors
  - [ ] Verify zero errors and warnings

- [ ] 5.3.3 Run pyrefly semantic checking
  ```bash
  uv run pyrefly check
  ```
  - [ ] Fix any semantic errors
  - [ ] Verify zero errors

- [ ] 5.3.4 Run full test suite
  ```bash
  uv run pytest tests/codeintel_rev/ -v --cov=codeintel_rev --cov-report=html
  ```
  - [ ] Verify all tests pass
  - [ ] Verify 95%+ coverage on new code
  - [ ] Review coverage report for gaps

- [ ] 5.3.5 Check for import cycle issues
  ```bash
  python tools/check_imports.py
  ```
  - [ ] Fix any circular import issues

- [ ] 5.3.6 Check for new suppressions
  ```bash
  python tools/check_new_suppressions.py codeintel_rev/
  ```
  - [ ] Verify zero new suppressions

### 5.4 Integration Testing
- [ ] 5.4.1 Manual integration tests
  - [ ] Start application: `uvicorn codeintel_rev.app.main:app --port 8000`
  - [ ] Verify `/healthz` returns 200
  - [ ] Verify `/readyz` returns all checks healthy
  - [ ] Test MCP tool: `list_paths`
  - [ ] Test MCP tool: `search_text`
  - [ ] Test MCP tool: `semantic_search`
  - [ ] Test MCP tool: `blame_range`
  - [ ] Test MCP tool: `file_history`
  - [ ] Verify logs show no errors

- [ ] 5.4.2 Test configuration variations
  - [ ] Test with `FAISS_PRELOAD=0` (lazy loading)
  - [ ] Test with `FAISS_PRELOAD=1` (eager loading)
  - [ ] Test with missing FAISS index (should fail startup)
  - [ ] Test with invalid `REPO_ROOT` (should fail startup)

- [ ] 5.4.3 Performance verification
  - [ ] Measure startup time with/without pre-loading
  - [ ] Measure first semantic search latency (lazy vs eager)
  - [ ] Verify no regression in request handling time
  - [ ] Document timing numbers

## 6. OpenSpec Validation

### 6.1 Create Capability Spec Delta
- [ ] 6.1.1 Create `openspec/changes/codeintel-config-lifecycle-management/specs/codeintel-configuration/spec.md`
  - [ ] Add "## ADDED Requirements" section
  - [ ] Write requirements for centralized configuration lifecycle
  - [ ] Write requirements for explicit dependency injection
  - [ ] Write requirements for fail-fast startup behavior
  - [ ] Write requirements for comprehensive health checks
  - [ ] Add scenarios for each requirement

### 6.2 Validate OpenSpec Change
- [ ] 6.2.1 Run OpenSpec validation
  ```bash
  openspec validate codeintel-config-lifecycle-management --strict
  ```
  - [ ] Fix any validation errors
  - [ ] Verify all requirements have scenarios
  - [ ] Verify delta operations are correct

### 6.3 Prepare PR Evidence
- [ ] 6.3.1 Collect command outputs
  - [ ] Capture `uv run ruff format && uv run ruff check --fix` output
  - [ ] Capture `uv run pyright --warnings --pythonversion=3.13` output
  - [ ] Capture `uv run pyrefly check` output
  - [ ] Capture `uv run pytest -q` output
  - [ ] Capture `openspec validate codeintel-config-lifecycle-management --strict` output

- [ ] 6.3.2 Create PR description
  - [ ] Link to proposal.md
  - [ ] Paste tasks.md progress (all checked)
  - [ ] Paste command outputs
  - [ ] Add summary of changes
  - [ ] Document any breaking changes (internal only)
  - [ ] Document rollback procedure

## 7. Final Checklist

- [ ] 7.1 All tests pass
- [ ] 7.2 Zero Ruff/pyright/pyrefly errors
- [ ] 7.3 95%+ test coverage on new code
- [ ] 7.4 Documentation complete
- [ ] 7.5 OpenSpec validation passes
- [ ] 7.6 Manual integration tests pass
- [ ] 7.7 Performance verified (no regression)
- [ ] 7.8 All TODOs in code removed
- [ ] 7.9 CHANGELOG.md updated (if applicable)
- [ ] 7.10 Ready for PR submission

