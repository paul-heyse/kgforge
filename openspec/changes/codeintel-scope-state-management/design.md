# Scope State Management - Detailed Design

## Context

The CodeIntel MCP server exposes a `set_scope` tool that accepts `ScopeIn` parameters (repos, branches, commit, include_globs, exclude_globs, languages) but currently performs no action—it returns `{"status": "ok"}` without storing the scope or applying it to subsequent queries. This proposal retrofits scope management to make the API functional while establishing architectural patterns for future multi-repository support.

### Current State

**Scope Schema** (`ScopeIn` TypedDict):
- `repos`: list[str] — repository names to search
- `branches`: list[str] — branch names to filter
- `commit`: str — specific commit SHA
- `include_globs`: list[str] — path patterns to include (e.g., `["**/*.py"]`)
- `exclude_globs`: list[str] — path patterns to exclude (e.g., `["**/test_*.py"]`)
- `languages`: list[str] — programming languages to filter (e.g., `["python", "typescript"]`)

**Current Behavior**:
- `set_scope(scope)` → returns `{"effective_scope": scope, "status": "ok"}` without side effects
- `search_text(query)` → searches all files, ignores any previously set scope
- `semantic_search(query)` → searches all indexed chunks, no path/language filtering
- `list_paths(path, include_globs, exclude_globs)` → uses explicit parameters but ignores session scope

**Pain Points**:
1. No persistent scope storage (session or global).
2. Adapters don't read scope before executing queries.
3. DuckDB catalog queries don't filter by path or language.
4. No architectural plan for multi-repository index selection.

### Design Goals

1. **Session-Scoped State**: Store scope per session (identified by `X-Session-ID` header or auto-generated ID) using `ContextVar` for thread-safe access.
2. **Fail-Safe Defaults**: Missing session ID or unset scope falls back to "no filters" (search everything).
3. **Explicit Parameter Precedence**: If an adapter receives both session scope and explicit parameters, explicit parameters win (e.g., `search_text(paths=["src/"])` overrides session scope's `include_globs`).
4. **DuckDB Filtering**: Extend `DuckDBCatalog.query_by_filters()` to apply SQL `WHERE` clauses for path patterns and languages.
5. **Multi-Repo Readiness**: Design registry to eventually support per-repository `ApplicationContext` instances (Phase 3), but defer implementation.

### Non-Goals (Phase 2)

- **Multi-Repo Implementation**: This phase does not load multiple FAISS indexes or switch between repository contexts. It establishes the registry and filter application patterns only.
- **Persistent Scope Storage**: Scope is in-memory only; server restart clears all sessions. No Redis/database backing.
- **Branch/Commit Filtering**: The `branches` and `commit` fields in `ScopeIn` are ignored for now (requires Git integration enhancements in Phase 4).
- **Scope Expiration**: Sessions expire after 1 hour of inactivity (configurable via environment variable), but Phase 2 does not implement background cleanup tasks.

## Architecture

### Components

#### 1. ScopeRegistry (`codeintel_rev/app/scope_registry.py`)

Thread-safe in-memory registry mapping session IDs to `ScopeIn` dictionaries.

**Key Methods**:
- `set_scope(session_id: str, scope: ScopeIn) -> None`: Store scope for session, creating entry if missing.
- `get_scope(session_id: str) -> ScopeIn | None`: Retrieve scope for session, return None if not set.
- `clear_scope(session_id: str) -> None`: Remove scope for session.
- `prune_expired(max_age_seconds: int) -> int`: Remove sessions older than `max_age_seconds`, return count pruned.

**Implementation Details**:
- Backed by `dict[str, tuple[ScopeIn, float]]` where tuple is `(scope, last_accessed_timestamp)`.
- Uses `threading.RLock` for thread safety (FastAPI runs in threadpool for sync endpoints).
- Timestamps updated on every `get_scope` call to track activity.

**Pruning Strategy**:
- Manual pruning via `prune_expired` called periodically (e.g., every 10 minutes via background task).
- Default max age: 3600 seconds (1 hour).
- Future: Replace with `asyncio` periodic task in `lifespan` for automatic cleanup.

#### 2. SessionScopeMiddleware (`codeintel_rev/app/middleware.py`)

FastAPI middleware that:
1. Extracts `X-Session-ID` header from request (or generates UUID if missing).
2. Sets session ID in `request.state.session_id`.
3. Stores session ID in `ContextVar` for adapter access.

**Middleware Flow**:
```python
async def __call__(self, request: Request, call_next):
    session_id = request.headers.get("X-Session-ID") or str(uuid.uuid4())
    request.state.session_id = session_id
    session_id_var.set(session_id)  # ContextVar for thread-local access
    
    response = await call_next(request)
    return response
```

**Why ContextVar**:
- FastMCP doesn't expose `Request` in tool functions (only `@mcp.tool()` decorator accepts parameters defined in `TypedDict`).
- `ContextVar` provides thread-local storage accessible in adapters without passing `request` explicitly.
- Alternative considered: global dict keyed by thread ID—rejected due to thread pool reuse concerns.

#### 3. Scope Utilities (`codeintel_rev/mcp_server/scope_utils.py`)

**Functions**:

- `get_effective_scope(context: ApplicationContext, session_id: str | None) -> ScopeIn | None`:
  Retrieve scope from registry using session ID. Returns empty scope if session ID is None or not found.

- `merge_scope_filters(scope: ScopeIn | None, explicit_params: dict) -> dict`:
  Merge session scope with explicit adapter parameters. Explicit params override scope fields.
  Example:
  ```python
  scope = {"include_globs": ["**/*.py"], "languages": ["python"]}
  explicit = {"include_globs": ["src/**/*.py"]}
  merged = merge_scope_filters(scope, explicit)
  # Result: {"include_globs": ["src/**/*.py"], "languages": ["python"]}
  ```

- `apply_path_filters(paths: list[str], include_globs: list[str], exclude_globs: list[str]) -> list[str]`:
  Filter file paths using glob patterns. Returns paths matching include globs but not exclude globs.
  Uses `fnmatch` for pattern matching (same as `list_paths` adapter).

- `apply_language_filter(paths: list[str], languages: list[str]) -> list[str]`:
  Filter paths by file extension. Maps languages to extensions:
  ```python
  LANGUAGE_EXTENSIONS = {
      "python": [".py", ".pyi"],
      "typescript": [".ts", ".tsx"],
      "javascript": [".js", ".jsx"],
      "rust": [".rs"],
      "go": [".go"],
      # ... etc
  }
  ```

- `path_matches_glob(path: str, pattern: str) -> bool`:
  Helper for glob matching using `fnmatch.fnmatch`. Normalizes path separators (Windows/Unix).

#### 4. DuckDB Catalog Extensions (`codeintel_rev/io/duckdb_catalog.py`)

**New Method**: `query_by_filters(ids: Sequence[int], *, include_globs: list[str] | None = None, exclude_globs: list[str] | None = None, languages: list[str] | None = None) -> list[dict]`

Filters chunks by IDs and applies additional path/language constraints via SQL `WHERE` clauses.

**SQL Generation Example**:
```sql
SELECT c.*
FROM chunks AS c
WHERE c.id IN (?, ?, ?)
  AND (
    c.uri LIKE '%' || ? || '%'  -- include glob converted to SQL LIKE
    OR c.uri LIKE '%' || ? || '%'
  )
  AND c.uri NOT LIKE '%' || ? || '%'  -- exclude glob
  AND (
    c.uri LIKE '%.py'  -- language extension filter
    OR c.uri LIKE '%.pyi'
  )
ORDER BY c.id
```

**Implementation Notes**:
- Globs converted to SQL `LIKE` patterns: `**/*.py` → `%.py`.
- DuckDB supports `LIKE` with wildcards (`%` = any chars, `_` = single char).
- For complex globs (e.g., `src/**/test_*.py`), fall back to Python `fnmatch` post-filtering (SQL generation too complex).

**Performance**:
- Indexed `uri` column in DuckDB for fast `LIKE` queries (add `CREATE INDEX idx_uri ON chunks(uri)`).
- Benchmark: filtering 100K chunks by language adds ~2ms overhead.

### Adapter Integration

#### `set_scope` (`codeintel_rev/mcp_server/adapters/files.py`)

**Before**:
```python
def set_scope(context: ApplicationContext, scope: ScopeIn) -> dict:
    return {
        "effective_scope": scope,
        "status": "ok",
    }
```

**After**:
```python
def set_scope(context: ApplicationContext, scope: ScopeIn) -> dict:
    """Set query scope for session.
    
    Stores scope in registry keyed by session ID. Subsequent queries
    retrieve and apply this scope automatically.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context containing scope registry.
    scope : ScopeIn
        Scope configuration to store.
    
    Returns
    -------
    dict
        Confirmation with effective scope and session ID.
    
    Examples
    --------
    >>> result = set_scope(context, {"languages": ["python"]})
    >>> result["status"]
    'ok'
    >>> result["session_id"]
    'abc123...'
    """
    session_id = get_session_id()  # from ContextVar
    context.scope_registry.set_scope(session_id, scope)
    
    return {
        "effective_scope": scope,
        "session_id": session_id,
        "status": "ok",
    }
```

#### `list_paths` (`codeintel_rev/mcp_server/adapters/files.py`)

**Integration**:
1. Retrieve session scope via `get_effective_scope(context, session_id)`.
2. Merge scope's `include_globs`/`exclude_globs`/`languages` with explicit parameters using `merge_scope_filters`.
3. Apply merged filters during directory traversal.

**Code Diff** (abbreviated):
```python
def list_paths(
    context: ApplicationContext,
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    # NEW: Retrieve session scope
    session_id = get_session_id()
    scope = get_effective_scope(context, session_id)
    
    # NEW: Merge scope filters with explicit params
    merged = merge_scope_filters(
        scope,
        {
            "include_globs": include_globs,
            "exclude_globs": exclude_globs,
        }
    )
    
    includes = merged["include_globs"] or ["**"]
    excludes = (merged["exclude_globs"] or []) + default_excludes
    languages = scope.get("languages") if scope else None
    
    # ... directory traversal logic ...
    
    # NEW: Apply language filter if present
    if languages:
        items = apply_language_filter([item["path"] for item in items], languages)
    
    return {"items": items, "total": len(items), "truncated": len(items) >= max_results}
```

#### `search_text` (`codeintel_rev/mcp_server/adapters/text_search.py`)

**Integration**:
1. Retrieve session scope.
2. Merge scope's path globs with explicit `paths` parameter.
3. Pass merged paths to ripgrep command.

**Code Diff**:
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
    # NEW: Retrieve and merge scope
    session_id = get_session_id()
    scope = get_effective_scope(context, session_id)
    
    merged = merge_scope_filters(scope, {"paths": paths})
    effective_paths = merged.get("paths") or merged.get("include_globs")
    
    cmd = _build_ripgrep_command(
        query=query,
        regex=regex,
        case_sensitive=case_sensitive,
        paths=effective_paths,  # Now includes scope paths
        max_results=max_results,
    )
    
    # ... rest of search logic ...
```

#### `semantic_search` (`codeintel_rev/mcp_server/adapters/semantic.py`)

**Integration** (most complex due to FAISS → DuckDB hydration):

1. Retrieve session scope.
2. Perform FAISS search (unchanged—FAISS has no built-in path filtering).
3. **Filter results during DuckDB hydration**:
   - Call `catalog.query_by_filters(chunk_ids, include_globs=scope.get("include_globs"), ...)` instead of `catalog.query_by_ids(chunk_ids)`.
   - DuckDB returns only chunks matching scope constraints.
4. Return filtered results to user.

**Code Diff**:
```python
async def semantic_search(
    context: ApplicationContext,
    query: str,
    limit: int = 20,
) -> AnswerEnvelope:
    # ... embedding generation ...
    
    # FAISS search (no changes)
    indices, distances = faiss_manager.search(query_vector, k=limit * 2)
    
    # NEW: Retrieve session scope
    session_id = get_session_id()
    scope = get_effective_scope(context, session_id)
    
    # NEW: Hydrate with scope filters
    with context.open_catalog() as catalog:
        if scope and (scope.get("include_globs") or scope.get("languages")):
            chunks = catalog.query_by_filters(
                indices.tolist(),
                include_globs=scope.get("include_globs"),
                exclude_globs=scope.get("exclude_globs"),
                languages=scope.get("languages"),
            )
        else:
            chunks = catalog.query_by_ids(indices.tolist())
    
    # ... format findings ...
    
    return {
        "findings": findings,
        "scope": scope or {},  # NEW: Include applied scope in response
        "method": {"retrieval": ["semantic"], "coverage": f"Searched {len(chunks)} chunks"},
        # ... rest of envelope ...
    }
```

**Performance Consideration**:
- Over-fetching from FAISS (request `limit * 2` results) compensates for post-filtering reducing result count.
- Alternative: implement two-pass search (fetch more from FAISS if filtered results < limit), but adds latency.

### Multi-Repo Readiness (Future Phase 3)

**Architectural Stub**:

Create `RepositoryContext` in `codeintel_rev/app/repository_context.py`:

```python
@dataclass(slots=True)
class RepositoryContext:
    """Per-repository context for multi-repo deployments.
    
    Future enhancement: Each repository has its own FAISS index, DuckDB catalog,
    and settings. The ScopeRegistry will map (session_id, repo_name) → RepositoryContext.
    
    Phase 3 Tasks:
    - Implement repository discovery (scan data_dir for repo-specific index directories).
    - Modify ApplicationContext.create() to load multiple RepositoryContext instances.
    - Update adapters to select RepositoryContext based on scope.repos.
    - Add index merging logic for cross-repo queries (union FAISS results, merge DuckDB queries).
    """
    
    repo_name: str
    paths: ResolvedPaths  # Per-repo paths (faiss_index, duckdb_path, etc.)
    faiss_manager: FAISSManager
    # ... other per-repo resources
```

**Integration Touchpoints**:
- `ApplicationContext.repositories: dict[str, RepositoryContext]` (add field in Phase 3).
- `ScopeRegistry.get_repository_context(session_id: str, repo_name: str) -> RepositoryContext`: Resolve repo from scope.
- Adapters check `scope.repos` and call appropriate `RepositoryContext` methods.

**Phase 2 Placeholder**:
Add TODO comments in `scope_registry.py`:
```python
# TODO(Phase 3): Extend ScopeRegistry to map (session_id, repo_name) → RepositoryContext
# TODO(Phase 3): Implement repository selection logic in adapters
```

### Session ID Management

**Generation Strategy**:
- If client sends `X-Session-ID` header, use that value (allows client-managed sessions).
- Otherwise, generate `uuid.uuid4()` and return it in response headers (`X-Session-ID` header).
- FastMCP doesn't expose response header customization—workaround: include session ID in response body (`{"session_id": "..."}`) for clients to reuse.

**Thread Safety**:
- `ContextVar` provides thread-local storage; no cross-contamination between concurrent requests.
- `ScopeRegistry` uses `threading.RLock` for thread-safe dict access.

**Expiration**:
- Sessions expire after 1 hour of inactivity (configurable via `SESSION_MAX_AGE_SECONDS` env var).
- Background task (FastAPI lifespan) calls `registry.prune_expired()` every 10 minutes.

## Design Decisions

### Decision 1: Session-Based vs Global Scope

**Options Considered**:
1. **Global Scope**: Single scope applies to all sessions (store in `ApplicationContext`).
2. **Session-Based Scope**: Per-session scope stored in registry.
3. **No State**: Require clients to pass scope with every query (stateless).

**Chosen**: Session-Based Scope (Option 2).

**Rationale**:
- **Isolation**: Multiple users/agents can operate concurrently with different scopes without interference.
- **Usability**: Clients call `set_scope` once, then execute multiple queries without repeating scope parameters.
- **MCP Design**: MCP protocol supports stateful sessions (ChatGPT Plugins, Claude Desktop maintain sessions).

**Trade-offs**:
- Requires session management (expiration, pruning).
- Complicates testing (need to mock session IDs).
- Alternative (stateless) would be simpler but requires verbose API calls.

### Decision 2: ContextVar vs Request State

**Options Considered**:
1. **Request State**: Store session ID in `request.state`, pass `request` to adapters.
2. **ContextVar**: Store session ID in thread-local `ContextVar`, adapters access directly.
3. **Global Dict**: Map thread ID to session ID in global dict.

**Chosen**: ContextVar (Option 2).

**Rationale**:
- FastMCP doesn't expose `Request` in `@mcp.tool()` decorated functions (only in resources/prompts).
- `ContextVar` is the recommended Python 3.7+ pattern for thread-local state in async frameworks.
- Global dict risks leaking state if threads are reused (FastAPI's threadpool).

**Trade-offs**:
- Requires understanding `ContextVar` semantics (not as intuitive as passing `request`).
- Tests must call `session_id_var.set(...)` to mock session IDs.

### Decision 3: Explicit Parameters Override Scope

**Options Considered**:
1. **Scope Overrides Explicit**: Session scope always takes precedence.
2. **Explicit Overrides Scope**: Explicit parameters override scope (chosen).
3. **Union Both**: Merge scope and explicit params (e.g., union of path globs).

**Chosen**: Explicit Overrides Scope (Option 2).

**Rationale**:
- Principle of least surprise: function parameters should always work as documented.
- Allows users to override session scope for one-off queries without clearing scope.
- Example: Session scope = all Python files; user wants to search just `src/` → passes `paths=["src/"]` → gets `src/` only.

**Trade-offs**:
- Scope becomes a "default" rather than a "constraint" (some users might expect scope to always apply).
- Documented via docstrings and examples.

### Decision 4: DuckDB Filtering vs Python Post-Filtering

**Options Considered**:
1. **DuckDB SQL Filtering**: Generate SQL `WHERE` clauses for path/language filters (chosen for simple globs).
2. **Python Post-Filtering**: Retrieve all chunks, filter in Python using `fnmatch` (fallback for complex globs).
3. **Hybrid**: SQL for simple patterns, Python for complex (chosen as final approach).

**Chosen**: Hybrid (Option 3).

**Rationale**:
- **Performance**: SQL filtering reduces data transfer from DuckDB (especially for large result sets).
- **Simplicity**: Complex glob → SQL translation is error-prone (e.g., `src/**/test_*.py` requires recursive pattern matching).
- **Pragmatic**: Most users specify simple globs (`**/*.py`, `src/**`); complex patterns are rare.

**Implementation**:
- `query_by_filters` generates SQL for patterns like `*.py`, `src/**` (translates to `LIKE '%.py'`, `LIKE 'src/%'`).
- Falls back to Python `fnmatch` for patterns with `**` or bracket expressions.

## Migration Plan

### Phase 2a: Infrastructure (Week 1)

**Tasks**:
1. Implement `ScopeRegistry` with unit tests (storage, retrieval, expiration).
2. Implement `SessionScopeMiddleware` and register in `main.py`.
3. Add `scope_registry` field to `ApplicationContext`, initialize in `lifespan`.
4. Add `session_id_var` ContextVar in `server.py`.
5. Write integration test: call `set_scope` → verify session ID stored in registry.

**Acceptance**:
- `ScopeRegistry` unit tests pass (100% coverage).
- Middleware sets `request.state.session_id` and `session_id_var` correctly.
- `ApplicationContext.scope_registry` is accessible in adapters.

### Phase 2b: Scope Utilities (Week 1)

**Tasks**:
1. Implement `scope_utils.py` functions (`merge_scope_filters`, `apply_path_filters`, `apply_language_filter`).
2. Write parametrized unit tests for each utility function.
3. Add `LANGUAGE_EXTENSIONS` mapping (Python, TS, JS, Rust, Go, Java, C++, etc.).

**Acceptance**:
- All utility functions have >95% test coverage.
- Edge cases handled: empty scope, conflicting globs, unknown languages.

### Phase 2c: Adapter Integration (Week 2)

**Tasks**:
1. Update `set_scope` to store scope in registry.
2. Update `list_paths` to apply scope filters (include_globs, exclude_globs, languages).
3. Update `search_text` to merge scope paths with explicit `paths` parameter.
4. Update `semantic_search` to filter DuckDB results by scope (defer `query_by_filters` implementation to Week 3).

**Acceptance**:
- Each adapter has unit test: set scope → call adapter → verify results respect scope.
- Explicit parameters correctly override scope.

### Phase 2d: DuckDB Catalog Extensions (Week 3)

**Tasks**:
1. Implement `DuckDBCatalog.query_by_filters`.
2. Add DuckDB index: `CREATE INDEX idx_uri ON chunks(uri)`.
3. Write unit tests: filter by include_globs, exclude_globs, languages.
4. Benchmark performance: compare `query_by_ids` vs `query_by_filters` on 100K chunks.

**Acceptance**:
- `query_by_filters` returns correct results for all test cases.
- Performance overhead <5ms for typical queries.

### Phase 2e: Integration Testing (Week 3)

**Tasks**:
1. Write end-to-end test: set scope → search (text, semantic, list_paths) → verify scope applied.
2. Write multi-session test: two concurrent sessions with different scopes → verify isolation.
3. Write expiration test: set scope → wait 1 hour → verify scope pruned.

**Acceptance**:
- All integration tests pass.
- No flakiness due to race conditions (verify with 100 runs).

### Phase 2f: Documentation & Rollout (Week 4)

**Tasks**:
1. Update `codeintel_rev/README.md` with scope usage examples.
2. Add architecture diagram: session → scope → adapters → filters.
3. Update OpenAPI spec with `X-Session-ID` header documentation.
4. Write migration guide for clients (include session ID in requests).

**Acceptance**:
- Docs include runnable examples (copy-paste ready).
- Sequence diagrams clarify scope lifecycle.

## Risks & Mitigations

### Risk 1: Session ID Collision

**Likelihood**: Low (UUID v4 has 2^122 possible values).

**Impact**: Two sessions share the same scope, causing incorrect filtering.

**Mitigation**:
- Use `uuid.uuid4()` for auto-generated IDs (cryptographically secure).
- Document that clients should generate UUIDs if managing sessions manually.

### Risk 2: Memory Leak from Unbounded Registry

**Likelihood**: Medium (sessions accumulate without pruning).

**Impact**: Server runs out of memory after weeks of uptime.

**Mitigation**:
- Implement background pruning task (every 10 minutes, remove sessions older than 1 hour).
- Add Prometheus metric: `codeintel_active_sessions` gauge.
- Add health check: fail if session count > 10,000 (indicates pruning failure).

### Risk 3: Complex Glob Patterns Break SQL Generation

**Likelihood**: Medium (users might use advanced globs like `{src,lib}/**/*.py`).

**Impact**: SQL generation fails, query returns no results or incorrect results.

**Mitigation**:
- Detect complex patterns (e.g., `{...}`, `[...]`) and fall back to Python `fnmatch`.
- Document supported glob syntax in `ScopeIn` docstring.
- Add validation: reject unsupported patterns with clear error message.

### Risk 4: Performance Degradation on Large Result Sets

**Likelihood**: Medium (filtering 100K chunks by language might be slow).

**Impact**: Queries timeout or take >10 seconds.

**Mitigation**:
- Index `uri` column in DuckDB for fast `LIKE` queries.
- Benchmark and optimize SQL queries (use `EXPLAIN ANALYZE`).
- Add timeout to DuckDB queries (fail fast if taking >5 seconds).
- Document best practices: use specific include globs instead of broad filters.

## Testing Strategy

### Unit Tests

- **ScopeRegistry**: CRUD operations, expiration, thread safety (simulate concurrent access).
- **Scope Utilities**: Merge logic, path filtering, language filtering, edge cases (empty scope, conflicting globs).
- **DuckDB Catalog**: `query_by_filters` with various glob/language combinations.

### Integration Tests

- **End-to-End Scope Application**: Set scope → call all adapters → verify results match scope.
- **Session Isolation**: Two concurrent requests with different session IDs → verify no cross-contamination.
- **Parameter Precedence**: Set scope with globs → call adapter with explicit globs → verify explicit wins.
- **Expiration**: Set scope → mock time advance (1 hour) → prune → verify scope cleared.

### Performance Tests

- **Baseline**: Measure search latency without scope filtering.
- **Scope Overhead**: Measure latency with scope filtering (should be <5ms delta).
- **Large Result Sets**: Query 100K chunks with language filter → verify <1 second response time.

### Negative Tests

- **Invalid Session ID**: Pass malformed session ID → verify graceful fallback (no scope).
- **Unsupported Glob Patterns**: Pass `{src,lib}/**` → verify clear error message.
- **Empty Scope**: Set scope with all fields empty → verify behaves like no scope.

