## Context

### Current State: Configuration Anti-Patterns

The CodeIntel MCP server exhibits three configuration management anti-patterns that violate AGENTS.md design principles:

#### 1. Redundant Settings Loading

Every adapter function calls `load_settings()` independently:

```python
# codeintel_rev/mcp_server/adapters/files.py (CURRENT - BAD)
def list_paths(...) -> dict:
    settings = load_settings()  # ❌ Re-parses environment variables
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    # ...

def open_file(...) -> dict:
    settings = load_settings()  # ❌ Again!
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    # ...
```

**Problem**: Each request triggers 4+ calls to `load_settings()`, which:
- Re-parses 20+ environment variables per call
- Allocates new `Settings` objects redundantly
- Prevents configuration caching and optimization
- Violates DRY principle across 4 adapters

#### 2. Inconsistent Context Access Patterns

The semantic adapter uses a different pattern:

```python
# codeintel_rev/mcp_server/adapters/semantic.py (CURRENT - INCONSISTENT)
def _semantic_search_sync(...) -> AnswerEnvelope:
    context = get_service_context()  # Uses @lru_cache singleton
    max_results = context.settings.limits.max_results
    # ...
```

While other adapters use direct loading:

```python
# codeintel_rev/mcp_server/adapters/text_search.py (CURRENT - INCONSISTENT)
def search_text(...) -> dict:
    settings = load_settings()  # Direct environment parsing
    repo_root = Path(settings.paths.repo_root).resolve()
    # ...
```

**Problem**: Two different configuration access patterns:
- `get_service_context()` provides caching but uses global singleton
- `load_settings()` is explicit but redundant
- No consistent pattern for new adapter development
- Testing requires different mocking strategies per adapter

#### 3. Deferred FAISS Initialization

FAISS index loading is deferred until first semantic search:

```python
# codeintel_rev/mcp_server/service_context.py (CURRENT - DEFERRED)
def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
    # Only loads on first call
    if not self._faiss_loaded:
        self.faiss_manager.load_cpu_index()
        self._faiss_loaded = True
```

**Problem**: First-request latency and lack of startup validation:
- First semantic search takes 2-10 seconds (index loading time)
- Application appears "ready" before FAISS is actually available
- Health checks can't validate FAISS index existence at startup
- GPU initialization failures discovered too late

### Design Review Findings

The design review document ([`codeintel_rev/CodeIntel MCP: Design Review and Improve.md`](../../codeintel_rev/CodeIntel%20MCP:%20Design%20Review%20and%20Improve.md)) identified these issues in Section 1:

> "The current implementation loads configuration from environment variables repeatedly in each adapter... This can be streamlined by initializing configuration once and sharing it... Leveraging the existing ServiceContext singleton is a good start... Extending this pattern, you might inject or globally expose ServiceContext.settings so adapters use get_service_context().settings rather than load_settings() each time."

However, the review correctly identifies that **global singletons are not the ideal solution**. Instead, we need:
1. **Explicit dependency injection** via FastAPI application state
2. **Configuration lifecycle** tied to application startup/shutdown
3. **Fail-fast behavior** for missing or invalid configuration

## Goals

### Primary Goals

1. **Eliminate Redundant Configuration Loading**
   - Load settings once during application startup
   - Share single `Settings` instance across all requests
   - Zero environment variable parsing during request handling

2. **Centralize Configuration Lifecycle Management**
   - Configuration loaded in FastAPI `lifespan()` function
   - Stored in `app.state` for explicit dependency injection
   - Fail-fast on startup if configuration is invalid

3. **Implement Comprehensive Health Checks**
   - Validate all required resources (FAISS index, DuckDB, vLLM) at startup
   - Provide detailed `/readyz` endpoint for Kubernetes readiness probes
   - Optional FAISS pre-loading for consistent response times

4. **Maintain 100% AGENTS.MD Compliance**
   - No global state or singletons
   - Explicit dependency injection everywhere
   - RFC 9457 Problem Details for all errors
   - Structured logging with context fields
   - Zero Ruff/pyright/pyrefly errors

5. **Future-Proof for Multi-Repository Support**
   - Architecture supports per-repository contexts
   - Single-repo mode remains simple and efficient
   - Clear migration path to multi-repo mode

### Non-Goals

- **NOT changing adapter public APIs** (MCP tools remain unchanged from client perspective)
- **NOT modifying settings schema** (environment variables stay the same)
- **NOT adding new features** (purely refactoring existing functionality)
- **NOT implementing multi-repo support now** (architecture ready, implementation deferred)

## Decisions

### Decision 1: ApplicationContext Dataclass

**What**: Create `ApplicationContext` dataclass as single source of configuration truth.

**Why**: 
- Frozen dataclass is immutable and thread-safe
- Explicit fields with type hints (pyright-friendly)
- Can be easily mocked for unit tests
- Natural place for long-lived clients (vLLM, FAISS)

**Alternative Considered**: Keep `ServiceContext` singleton with `@lru_cache`
- **Rejected**: Global singletons violate AGENTS.MD (testing difficulty, implicit dependencies)

**Implementation**:

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

### Decision 2: Resolved Paths Dataclass

**What**: Create `ResolvedPaths` dataclass to eliminate path resolution duplication.

**Why**:
- Every adapter currently resolves `repo_root`, `faiss_index`, etc. independently
- Path resolution logic scattered across 4+ files
- No single place to validate path existence/permissions
- Centralizing eliminates 50+ lines of duplicated code

**Alternative Considered**: Keep path resolution in adapters
- **Rejected**: Duplication leads to bugs (different resolution logic in different places)

**Implementation**:

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

### Decision 3: FastAPI Lifespan Integration

**What**: Load configuration in FastAPI `lifespan()` function, store in `app.state`.

**Why**:
- FastAPI lifespan is the idiomatic place for application initialization
- `app.state` provides request-level access without global state
- Exceptions in lifespan prevent application from starting (fail-fast)
- Natural place for startup health checks

**Alternative Considered**: Module-level singleton initialization
- **Rejected**: Can't control initialization timing or error handling; testing nightmare

**Implementation**:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager with explicit configuration initialization."""
    context = ApplicationContext.create()
    app.state.context = context
    
    readiness = ReadinessProbe(context)
    await readiness.initialize()
    app.state.readiness = readiness
    
    try:
        yield
    finally:
        await readiness.shutdown()
```

### Decision 4: Explicit Dependency Injection in Adapters

**What**: All adapter functions accept `ApplicationContext` as first parameter.

**Why**:
- Makes dependencies explicit (easy to trace, test, and understand)
- No hidden global state or magic
- Junior developers can see exactly what each function needs
- Mocking for tests is straightforward

**Alternative Considered**: Pass only `Settings` object
- **Rejected**: Adapters also need `vllm_client`, `faiss_manager`, and path helpers; passing context is cleaner

**Implementation Pattern**:

```python
# BEFORE (BAD)
def list_paths(...) -> dict:
    settings = load_settings()  # Hidden dependency
    repo_root = Path(settings.paths.repo_root).resolve()
    # ...

# AFTER (GOOD)
def list_paths(context: ApplicationContext, ...) -> dict:
    repo_root = context.paths.repo_root  # Explicit dependency
    # ...
```

### Decision 5: MCP Server Context Extraction

**What**: MCP tool wrappers extract `ApplicationContext` from FastAPI request state.

**Why**:
- FastMCP provides `Request` object to tool handlers
- `request.app.state` is the FastAPI way to access application-level state
- One helper function (`_get_context`) centralizes extraction logic

**Alternative Considered**: Pass context as tool parameter
- **Rejected**: MCP protocol doesn't support custom dependency injection; would break client compatibility

**Implementation**:

```python
def _get_context(request: Request) -> ApplicationContext:
    """Extract ApplicationContext from FastAPI state."""
    context: ApplicationContext | None = request.app.state.get("context")
    if context is None:
        raise RuntimeError("ApplicationContext not initialized")
    return context

@mcp.tool()
def list_paths(request: Request, path: str | None = None, ...) -> dict:
    context = _get_context(request)
    return files_adapter.list_paths(context, path, ...)
```

### Decision 6: Comprehensive Readiness Probe

**What**: Create `ReadinessProbe` class with checks for all critical resources.

**Why**:
- Kubernetes requires `/readyz` endpoint for readiness probes
- Current implementation only checks file existence (doesn't validate content)
- Should verify: FAISS index loadable, DuckDB accessible, vLLM reachable
- Fail-fast prevents deploying broken configuration

**Implementation**:

```python
class ReadinessProbe:
    """Manages readiness checks across core dependencies."""
    
    def _run_checks(self) -> dict[str, CheckResult]:
        results: dict[str, CheckResult] = {}
        results["repo_root"] = self._check_directory(paths.repo_root)
        results["faiss_index"] = self._check_file(paths.faiss_index, ...)
        results["duckdb_catalog"] = self._check_file(paths.duckdb_path, ...)
        results["vllm_service"] = self._check_vllm_connection()
        return results
```

### Decision 7: Optional FAISS Pre-loading

**What**: Add `FAISS_PRELOAD` environment variable to control eager vs lazy loading.

**Why**:
- Development: Lazy loading speeds up startup (don't wait for FAISS)
- Production: Eager loading ensures consistent response times (no first-request latency)
- Kubernetes: Pre-loading allows readiness probe to validate FAISS actually works

**Default**: `FAISS_PRELOAD=0` (lazy) for backward compatibility and fast development iteration

**Implementation**:

```python
# In lifespan():
if context.settings.index.faiss_preload:
    LOGGER.info("Pre-loading FAISS index during startup")
    await asyncio.to_thread(_preload_faiss_index, context)
```

### Decision 8: Remove ServiceContext Singleton

**What**: Delete `codeintel_rev/mcp_server/service_context.py` entirely.

**Why**:
- Functionality fully replaced by `ApplicationContext`
- `@lru_cache` singleton pattern no longer needed
- Simplifies codebase (one less abstraction layer)
- Tests become simpler (no cache clearing needed)

**Migration**: All `get_service_context()` calls replaced with `context` parameter

## Detailed Design

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Application                                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  lifespan()                                             ││
│  │  1. ApplicationContext.create()                         ││
│  │  2. ReadinessProbe.initialize()                         ││
│  │  3. Optional: preload FAISS                            ││
│  │  4. Store in app.state                                 ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  app.state.context: ApplicationContext                  ││
│  │  - settings: Settings (immutable)                       ││
│  │  - paths: ResolvedPaths (absolute)                     ││
│  │  - vllm_client: VLLMClient (persistent)                ││
│  │  - faiss_manager: FAISSManager (lazy GPU)              ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Server (server.py)                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  @mcp.tool() decorators                                 ││
│  │  1. Extract context via _get_context(request)          ││
│  │  2. Pass context to adapter functions                  ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Adapters (files, text_search, history, semantic)           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  All functions accept ApplicationContext as first param││
│  │  - Use context.paths.repo_root (no path resolution)   ││
│  │  - Use context.settings.limits.max_results            ││
│  │  - Use context.vllm_client for embeddings             ││
│  │  - Use context.faiss_manager for search               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Configuration Loading Sequence

```
1. FastAPI Startup
   ├─ lifespan() entered
   │
2. Load Configuration
   ├─ load_settings() reads environment variables ONCE
   ├─ resolve_application_paths() validates all paths
   │  ├─ Raises ConfigurationError if repo_root missing
   │  ├─ Converts relative paths to absolute
   │  └─ Returns ResolvedPaths dataclass
   │
3. Initialize Clients
   ├─ VLLMClient(settings.vllm) created
   ├─ FAISSManager(...) created (CPU index not loaded yet)
   ├─ ApplicationContext assembled
   │
4. Health Checks
   ├─ ReadinessProbe.initialize()
   │  ├─ Check repo_root exists
   │  ├─ Check data directories (create if missing)
   │  ├─ Check FAISS index file exists
   │  ├─ Check DuckDB catalog exists
   │  ├─ Check vLLM service reachable
   │  └─ Return CheckResult per resource
   │
5. Optional FAISS Pre-loading
   ├─ If FAISS_PRELOAD=1:
   │  ├─ faiss_manager.load_cpu_index()
   │  ├─ faiss_manager.clone_to_gpu()
   │  └─ Log success or degraded mode
   │
6. Store Context
   ├─ app.state.context = context
   ├─ app.state.readiness = readiness
   │
7. Application Ready
   └─ Serve requests
```

### Request Handling Sequence

```
1. Client Request
   ├─ POST /mcp/tools/list_paths
   │
2. FastMCP Routing
   ├─ Calls @mcp.tool() decorated function
   ├─ Injects FastAPI Request object
   │
3. Context Extraction
   ├─ list_paths(request, path=None, ...)
   ├─ context = _get_context(request)
   │  └─ Retrieves app.state.context
   │
4. Adapter Call
   ├─ files_adapter.list_paths(context, path, ...)
   │  ├─ Uses context.paths.repo_root (already resolved)
   │  ├─ No load_settings() call
   │  └─ Returns result dict
   │
5. Response
   └─ FastMCP serializes to JSON
```

### Error Handling Philosophy

All configuration errors follow RFC 9457 Problem Details:

```python
# Configuration validation
if not repo_root.exists():
    raise ConfigurationError(
        f"Repository root does not exist: {repo_root}",
        context={"repo_root": str(repo_root), "source": "REPO_ROOT env var"}
    )

# Adapter errors
if not context.paths.faiss_index.exists():
    return _error_envelope(
        VectorSearchError(
            "FAISS index not available",
            context={"faiss_index": str(context.paths.faiss_index)}
        ),
        limits=["FAISS index missing - run indexing pipeline"]
    )
```

## Risks & Trade-offs

### Risk 1: Breaking Changes to Internal APIs

**Risk**: Adapter function signatures change (adding `context` parameter).

**Impact**: Internal code calling adapters directly would break.

**Mitigation**: 
- No external callers (adapters only called by MCP server)
- Incremental refactoring with tests at each step
- Backward compatibility layer if needed (wrapper functions)

**Severity**: Low (internal only)

### Risk 2: FastAPI State Management Complexity

**Risk**: Developers unfamiliar with `app.state` might struggle to understand flow.

**Impact**: Junior developers may not know where configuration comes from.

**Mitigation**:
- Comprehensive documentation with diagrams
- Clear helper function (`_get_context`) with docstring
- Examples in every adapter showing usage pattern

**Severity**: Low (documentation mitigates)

### Risk 3: FAISS Pre-loading Increases Startup Time

**Risk**: With `FAISS_PRELOAD=1`, startup takes 2-10 seconds longer.

**Impact**: Slower development iteration; Kubernetes readiness delay.

**Mitigation**:
- Default to lazy loading (`FAISS_PRELOAD=0`)
- Document trade-off in configuration guide
- Kubernetes `initialDelaySeconds` accounts for pre-loading time

**Severity**: Low (configurable behavior)

### Risk 4: Multi-Repository Architecture Premature

**Risk**: Designing for multi-repo when only single-repo needed adds complexity.

**Impact**: Code harder to understand for current single-repo use case.

**Mitigation**:
- Keep multi-repo support minimal (just architecture, not implementation)
- `RepositoryRegistry` marked as future enhancement
- Current code stays simple and single-repo focused

**Severity**: Very Low (isolated to design docs)

### Trade-off Analysis

| Aspect | Before | After | Trade-off |
|--------|--------|-------|-----------|
| **Configuration Loading** | 4+ calls per request | 1 call at startup | ✅ Faster requests, ✅ Consistent config |
| **First Request Latency** | 2-10 seconds (FAISS load) | 0 ms (if pre-loaded) | ✅ Consistent OR ⚠️ Slower startup (user choice) |
| **Code Complexity** | Scattered logic | Centralized context | ✅ Easier to understand, ⚠️ New abstraction |
| **Testing** | Global singletons hard to mock | Explicit injection easy to mock | ✅ Much easier testing |
| **Startup Time** | Fast (lazy loading) | Medium (if pre-loaded) | ⚠️ Slower startup OR ✅ Consistent latency |
| **Error Detection** | Runtime (first query) | Startup (fail-fast) | ✅ Errors caught early |

**Overall Assessment**: The trade-offs heavily favor the new design. The only downside is optional (FAISS pre-loading), and startup time increase is minimal (2-10 seconds) for production deployments.

## Migration Strategy

### Phase 1: Foundation (Days 1-2)

**Goal**: Create new infrastructure without touching existing code.

**Tasks**:
1. Create `codeintel_rev/app/config_context.py` with:
   - `ResolvedPaths` dataclass
   - `resolve_application_paths()` function
   - `ApplicationContext` dataclass
   - `ApplicationContext.create()` factory method

2. Create `codeintel_rev/app/readiness.py` with:
   - `CheckResult` dataclass
   - `ReadinessProbe` class
   - Health check methods for all resources

3. Write unit tests:
   - `tests/codeintel_rev/test_config_context.py`
   - Test path resolution with valid/invalid inputs
   - Test `ApplicationContext.create()` with mocked environment

**Verification**:
```bash
uv run pytest tests/codeintel_rev/test_config_context.py -v
uv run pyright codeintel_rev/app/config_context.py
uv run pyrefly check
```

**Rollback**: Delete new files (no impact on existing code)

### Phase 2: FastAPI Integration (Days 3-4)

**Goal**: Integrate configuration lifecycle into application startup.

**Tasks**:
1. Modify `codeintel_rev/app/main.py`:
   - Update `lifespan()` to create `ApplicationContext`
   - Initialize `ReadinessProbe`
   - Store both in `app.state`
   - Add optional FAISS pre-loading

2. Update `/readyz` endpoint to use `ReadinessProbe`

3. Add `FAISS_PRELOAD` to `config/settings.py`

4. Write integration tests:
   - `tests/codeintel_rev/test_app_lifespan.py`
   - Test successful startup with valid config
   - Test startup failure with missing FAISS index
   - Test readiness probe state

**Verification**:
```bash
uv run pytest tests/codeintel_rev/test_app_lifespan.py -v
# Manual test: Start app and check /readyz
uvicorn codeintel_rev.app.main:app --port 8000
curl http://localhost:8000/readyz | jq
```

**Rollback**: Revert `main.py` changes (context not used yet, no impact)

### Phase 3: Adapter Refactoring (Days 5-7)

**Goal**: Refactor adapters one at a time to use injected context.

**Order** (lowest risk first):
1. `files.py` (simplest, no external dependencies)
2. `history.py` (only uses repo_root and path utils)
3. `text_search.py` (uses repo_root and settings)
4. `semantic.py` (most complex, uses vLLM and FAISS)

**Per-Adapter Steps**:
1. Add `context: ApplicationContext` as first parameter
2. Replace `load_settings()` with `context.settings`
3. Replace path resolution with `context.paths.*`
4. Update docstrings
5. Write unit tests with mocked context
6. Run quality gates (Ruff, pyright, pyrefly, pytest)

**Verification** (per adapter):
```bash
uv run pytest tests/codeintel_rev/adapters/test_<adapter>_adapter.py -v
uv run pyright codeintel_rev/mcp_server/adapters/<adapter>.py
```

**Rollback**: Git revert individual adapter commits (isolated changes)

### Phase 4: MCP Server Integration (Day 8)

**Goal**: Wire up context injection in MCP tool wrappers.

**Tasks**:
1. Add `_get_context(request: Request)` helper to `server.py`
2. Update all `@mcp.tool()` functions:
   - Add `request: Request` parameter
   - Call `_get_context(request)`
   - Pass context to adapter functions
3. Update docstrings

**Verification**:
```bash
uv run pytest tests/codeintel_rev/test_mcp_server.py -v
# Integration test: Run full request flow
python -m tests.codeintel_rev.test_integration
```

**Rollback**: Revert `server.py` (breaks adapters until fixed, but testable)

### Phase 5: Cleanup (Day 9)

**Goal**: Remove deprecated code and finalize documentation.

**Tasks**:
1. Delete `codeintel_rev/mcp_server/service_context.py`
2. Remove unused imports from adapters
3. Run full test suite
4. Generate documentation

**Verification**:
```bash
uv run pytest -q
uv run ruff format && uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
make artifacts && git diff --exit-code
```

**Rollback**: Restore `service_context.py` if issues found (unlikely at this point)

### Rollback Plan (Emergency)

If critical issues discovered after merge:

1. **Immediate**: Revert merge commit
2. **Within 1 hour**: Verify revert restores functionality
3. **Within 4 hours**: Root cause analysis
4. **Within 1 day**: Fix identified issues in new PR
5. **Document**: What went wrong, what was missed in testing

## Alternatives Considered

### Alternative 1: Keep ServiceContext Singleton

**Description**: Extend `ServiceContext` to include settings, keep `@lru_cache` pattern.

**Pros**:
- Minimal changes to existing code
- Semantic adapter already uses this pattern
- Simple to implement

**Cons**:
- Global singletons violate AGENTS.MD principles
- Hard to test (must clear cache between tests)
- No control over initialization timing
- Can't fail-fast on startup (lazy initialization)

**Verdict**: ❌ Rejected (violates design principles)

### Alternative 2: Pass Settings Object Only

**Description**: Pass `Settings` to adapters instead of full `ApplicationContext`.

**Pros**:
- Simpler interface (one object vs dataclass with multiple fields)
- Clear what each adapter needs

**Cons**:
- Adapters also need `vllm_client`, `faiss_manager`, `paths`
- Would need to pass 4+ parameters to each adapter function
- Doesn't solve path resolution duplication
- No place to store lazy-initialized resources

**Verdict**: ❌ Rejected (insufficient; context is cleaner)

### Alternative 3: Dependency Injection Framework

**Description**: Use a DI framework like `dependency-injector` or `injector`.

**Pros**:
- Industry-standard approach
- Automatic dependency wiring
- Sophisticated lifecycle management

**Cons**:
- Additional dependency (adds complexity)
- Overkill for this use case (only 4 adapters)
- FastAPI already provides `app.state` mechanism
- Harder for junior developers to understand

**Verdict**: ❌ Rejected (over-engineering)

### Alternative 4: Thread-Local Storage

**Description**: Store configuration in `threading.local()` or `contextvars`.

**Pros**:
- No need to pass parameters
- Works across function boundaries

**Cons**:
- Hidden dependencies (magic behavior)
- Hard to test and debug
- Doesn't work with async code (unless using contextvars)
- Still need initialization mechanism

**Verdict**: ❌ Rejected (implicit dependencies are anti-pattern)

## Testing Strategy

### Unit Tests

**Coverage Target**: 95% line coverage for new code

**Test Modules**:

1. `tests/codeintel_rev/test_config_context.py`:
   - `test_resolve_application_paths_success` — valid repo root
   - `test_resolve_application_paths_missing_root` — ConfigurationError raised
   - `test_resolve_application_paths_not_directory` — ConfigurationError raised
   - `test_application_context_create` — all clients initialized
   - `test_application_context_ensure_faiss_ready` — lazy loading works
   - `test_application_context_ensure_faiss_ready_cached` — caching works
   - `test_application_context_open_catalog` — context manager works

2. `tests/codeintel_rev/test_readiness.py`:
   - `test_readiness_probe_all_healthy` — all checks pass
   - `test_readiness_probe_missing_faiss` — fails gracefully
   - `test_readiness_probe_vllm_unreachable` — network error handled
   - `test_readiness_probe_caching` — refresh updates state

3. `tests/codeintel_rev/adapters/test_*_adapter.py` (4 files):
   - Each adapter: test with mocked `ApplicationContext`
   - Verify no `load_settings()` calls
   - Verify context fields accessed correctly

### Integration Tests

**Test Scenarios**:

1. **Application Startup Success**:
   - Given: Valid configuration and all resources exist
   - When: Application starts
   - Then: `/healthz` returns 200, `/readyz` returns all checks healthy

2. **Application Startup Failure - Missing FAISS**:
   - Given: FAISS index file missing
   - When: Application attempts startup
   - Then: Startup fails with clear ConfigurationError

3. **First Request Without Pre-loading**:
   - Given: `FAISS_PRELOAD=0` and first semantic search
   - When: Request arrives
   - Then: FAISS loads on-demand, search succeeds

4. **First Request With Pre-loading**:
   - Given: `FAISS_PRELOAD=1` and first semantic search
   - When: Request arrives
   - Then: FAISS already loaded, search succeeds immediately

### Manual Testing Checklist

**Before PR**:
- [ ] Start application with valid config → `/healthz` and `/readyz` work
- [ ] Start application with missing FAISS → startup fails with clear error
- [ ] Call each MCP tool (list_paths, search_text, semantic_search, etc.) → all work
- [ ] Toggle `FAISS_PRELOAD=0` vs `FAISS_PRELOAD=1` → observe startup time difference
- [ ] Check logs → no errors, clear initialization messages
- [ ] Run under load (ab -n 100 -c 10) → no crashes, consistent response times

## Documentation Updates

### New Documentation

1. **`codeintel_rev/docs/CONFIGURATION.md`**:
   - Configuration management overview
   - Configuration loading sequence
   - Environment variables reference
   - Best practices (development vs production)
   - Kubernetes deployment guide
   - Troubleshooting section

2. **Inline Code Documentation**:
   - All new classes/functions have NumPy-style docstrings
   - Examples showing usage patterns
   - Cross-references to related components

3. **Architecture Diagrams**:
   - Component architecture (included in this design doc)
   - Configuration loading sequence (included in this design doc)
   - Request handling flow (included in this design doc)

### Updated Documentation

1. **`codeintel_rev/README.md`**:
   - Update "Configuration" section to reference new `CONFIGURATION.md`
   - Add note about `FAISS_PRELOAD` environment variable
   - Update "Quick Start" to mention fail-fast behavior

2. **Adapter Docstrings**:
   - Update all adapter functions to document `context` parameter
   - Update examples to show context usage

## Success Criteria

### Must Have (Blocking)

- ✅ Zero `load_settings()` calls in adapters
- ✅ All adapters accept `ApplicationContext` parameter
- ✅ FastAPI lifespan loads configuration once
- ✅ ConfigurationError prevents startup if config invalid
- ✅ `/readyz` endpoint validates all resources
- ✅ All tests pass (unit + integration)
- ✅ Zero Ruff/pyright/pyrefly errors
- ✅ Documentation complete

### Should Have (Non-Blocking)

- ✅ `FAISS_PRELOAD` environment variable implemented
- ✅ Multi-repo architecture documented (not implemented)
- ✅ Performance benchmarks showing no regression
- ✅ Example Kubernetes deployment YAML

### Could Have (Future)

- 🔜 Multi-repository support implementation
- 🔜 Configuration hot-reloading (requires architecture change)
- 🔜 Configuration UI/API (not in scope)

