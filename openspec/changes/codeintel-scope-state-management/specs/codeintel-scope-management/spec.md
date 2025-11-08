# Capability: codeintel-scope-management

**Status**: Draft  
**Version**: 1.0.0  
**Owner**: CodeIntel Team  
**Last Updated**: 2025-11-08

## Purpose

Define the contract for session-scoped scope management in the CodeIntel MCP server, enabling clients to set query constraints (path patterns, languages, repositories) that persist across multiple requests within a session.

## Scope

This capability covers:
- Session-based scope storage and retrieval
- Scope application in all search/file operations (semantic search, text search, file listing, history queries)
- Session lifecycle management (creation, expiration, pruning)
- Precedence rules when explicit parameters conflict with session scope

Out of scope (future capabilities):
- Multi-repository index management (Phase 3)
- Branch/commit-based filtering (Phase 4)
- Persistent scope storage (Redis/database backing)

## Requirements

### FR-SCOPE-001: Session-Scoped Scope Storage

**Priority**: MUST  
**Added**: 2025-11-08

The system MUST maintain per-session scope state that persists across multiple requests within the same session.

**Acceptance Criteria**:
- Each session is identified by a unique session ID (UUID).
- Calling `set_scope(scope)` stores the scope for the current session.
- Subsequent requests with the same session ID retrieve and apply the stored scope.
- Sessions are isolated: concurrent sessions with different scopes do not interfere.

**Verification**:
- Unit test: `test_scope_registry_set_and_get` verifies storage and retrieval.
- Integration test: `test_multi_session_isolation` verifies concurrent session isolation.

---

### FR-SCOPE-002: Session ID Management

**Priority**: MUST  
**Added**: 2025-11-08

The system MUST support both client-provided and auto-generated session IDs.

**Acceptance Criteria**:
- If client sends `X-Session-ID` header, use that value as session ID.
- If header is absent, generate UUID v4 and return it in response (`session_id` field).
- Session ID is stored in thread-local `ContextVar` for adapter access.
- Middleware sets session ID before invoking MCP tool handlers.

**Verification**:
- Unit test: `test_middleware_uses_client_session_id` verifies header extraction.
- Unit test: `test_middleware_generates_session_id` verifies UUID generation.

---

### FR-SCOPE-003: Scope Application in File Listing

**Priority**: MUST  
**Added**: 2025-11-08

The `list_paths` adapter MUST apply session scope filters (include_globs, exclude_globs, languages) when listing files.

**Acceptance Criteria**:
- If session scope has `include_globs`, only files matching patterns are returned.
- If session scope has `exclude_globs`, files matching patterns are excluded.
- If session scope has `languages`, only files with matching extensions are returned.
- Explicit parameters (`include_globs`, `exclude_globs`) override session scope.
- Filters are applied during directory traversal (not post-filtering).

**Verification**:
- Unit test: `test_list_paths_with_scope_globs` verifies glob filtering.
- Unit test: `test_list_paths_with_scope_language` verifies language filtering.
- Integration test: `test_set_scope_then_list_paths` verifies end-to-end filtering.

---

### FR-SCOPE-004: Scope Application in Text Search

**Priority**: MUST  
**Added**: 2025-11-08

The `search_text` adapter MUST apply session scope path filters when searching.

**Acceptance Criteria**:
- If session scope has `include_globs`, only files matching patterns are searched.
- Explicit `paths` parameter overrides session scope.
- Scope paths are passed to ripgrep command (no post-filtering).

**Verification**:
- Unit test: `test_search_text_with_scope_paths` verifies path filtering.
- Unit test: `test_search_text_explicit_override` verifies parameter precedence.
- Integration test: `test_set_scope_then_search_text` verifies end-to-end filtering.

---

### FR-SCOPE-005: Scope Application in Semantic Search

**Priority**: MUST  
**Added**: 2025-11-08

The `semantic_search` adapter MUST apply session scope filters during DuckDB chunk hydration.

**Acceptance Criteria**:
- FAISS search is performed without scope constraints (FAISS has no built-in filtering).
- Chunk IDs from FAISS are filtered via `DuckDBCatalog.query_by_filters` using scope's `include_globs`, `exclude_globs`, and `languages`.
- Response envelope includes `scope` field showing applied scope.
- If filtering reduces results below requested `limit`, all matching results are returned (no error).

**Verification**:
- Unit test: `test_semantic_search_with_scope_filters` verifies DuckDB filtering.
- Unit test: `test_semantic_search_response_includes_scope` verifies envelope field.
- Integration test: `test_set_scope_then_semantic_search` verifies end-to-end filtering.

---

### FR-SCOPE-006: DuckDB Catalog Filtering

**Priority**: MUST  
**Added**: 2025-11-08

The `DuckDBCatalog` MUST provide a `query_by_filters` method that filters chunks by path and language.

**Acceptance Criteria**:
- Method signature: `query_by_filters(ids, *, include_globs=None, exclude_globs=None, languages=None) -> list[dict]`.
- Simple globs (`*.py`, `src/**`) are converted to SQL `LIKE` patterns for efficient filtering.
- Complex globs (e.g., `src/**/test_*.py`) fall back to Python `fnmatch` post-filtering.
- Language filtering uses extension mapping (e.g., `python` → `[".py", ".pyi"]`).
- Results preserve order of input IDs (JOIN with UNNEST).

**Verification**:
- Unit test: `test_query_by_filters_include_globs` verifies glob filtering.
- Unit test: `test_query_by_filters_languages` verifies language filtering.
- Unit test: `test_query_by_filters_complex_globs` verifies fallback to Python filtering.

---

### FR-SCOPE-007: Explicit Parameter Precedence

**Priority**: MUST  
**Added**: 2025-11-08

When both session scope and explicit adapter parameters are present, explicit parameters MUST take precedence.

**Acceptance Criteria**:
- Example: Session scope has `include_globs=["**/*.py"]`; call `list_paths(include_globs=["**/*.ts"])` → returns TypeScript files only.
- Merge logic: explicit params override corresponding scope fields; unspecified params use scope defaults.
- Documented in adapter docstrings and user guide.

**Verification**:
- Unit test: `test_explicit_params_override_scope` (for each adapter).
- Integration test: `test_parameter_precedence` verifies override behavior.

---

### FR-SCOPE-008: Session Expiration

**Priority**: SHOULD  
**Added**: 2025-11-08

The system SHOULD expire sessions after a configurable period of inactivity to prevent memory leaks.

**Acceptance Criteria**:
- Default expiration: 1 hour of inactivity.
- Configurable via `SESSION_MAX_AGE_SECONDS` environment variable.
- Background task prunes expired sessions every 10 minutes.
- Accessing a session updates its last-accessed timestamp (LRU behavior).

**Verification**:
- Unit test: `test_scope_registry_prune_expired` verifies expiration logic.
- Integration test: `test_session_expiration` mocks time and verifies pruning.

---

### FR-SCOPE-009: Thread Safety

**Priority**: MUST  
**Added**: 2025-11-08

The scope registry MUST be thread-safe to support concurrent requests.

**Acceptance Criteria**:
- `ScopeRegistry` uses `threading.RLock` for all dict operations.
- `ContextVar` provides thread-local storage for session IDs.
- Concurrent `set_scope` calls from different threads do not corrupt registry state.

**Verification**:
- Unit test: `test_scope_registry_thread_safety` spawns 10 threads calling `set_scope` concurrently.
- Integration test: `test_concurrent_requests` uses `httpx.AsyncClient` to simulate concurrent MCP calls.

---

### FR-SCOPE-010: Scope Utilities

**Priority**: MUST  
**Added**: 2025-11-08

The system MUST provide utility functions for scope merging and filtering.

**Acceptance Criteria**:
- `get_effective_scope(context, session_id)` retrieves scope from registry.
- `merge_scope_filters(scope, explicit_params)` merges scope with explicit params (explicit wins).
- `apply_path_filters(paths, include_globs, exclude_globs)` filters paths using `fnmatch`.
- `apply_language_filter(paths, languages)` filters paths by file extension.

**Verification**:
- Unit tests for each utility function (see `test_scope_utils.py`).

---

### NFR-SCOPE-001: Performance Overhead

**Priority**: SHOULD  
**Added**: 2025-11-08

Scope filtering SHOULD add <5ms overhead to search operations.

**Acceptance Criteria**:
- Measured via Prometheus histogram: `codeintel_scope_filter_duration_seconds`.
- Benchmark test: `test_scope_performance` compares `query_by_ids` vs `query_by_filters` on 100K chunks.
- Acceptable if 95th percentile overhead <5ms.

**Verification**:
- Benchmark test: `test_scope_filter_performance`.

---

### NFR-SCOPE-002: Memory Efficiency

**Priority**: SHOULD  
**Added**: 2025-11-08

The scope registry SHOULD not consume >100MB memory under normal load (≤1000 active sessions).

**Acceptance Criteria**:
- Each session entry is <1KB (scope dict + timestamp).
- Pruning task runs every 10 minutes to remove expired sessions.
- Health check fails if session count >10,000 (indicates pruning failure).

**Verification**:
- Load test: `test_scope_registry_memory` creates 1000 sessions and measures memory usage.
- Health check test: `test_readiness_probe_registry_health` verifies threshold.

---

## Data Contracts

### ScopeIn Schema

**Format**: TypedDict (JSON Schema compatible)

```python
class ScopeIn(TypedDict, total=False):
    repos: list[str]           # Repository names (future: multi-repo)
    branches: list[str]         # Branch names (future: branch filtering)
    commit: str                 # Commit SHA (future: commit filtering)
    include_globs: list[str]    # Path patterns to include (e.g., ["**/*.py"])
    exclude_globs: list[str]    # Path patterns to exclude (e.g., ["**/test_*.py"])
    languages: list[str]        # Languages (e.g., ["python", "typescript"])
```

**Validation Rules**:
- All fields optional (total=False).
- `repos`, `branches`, `include_globs`, `exclude_globs`, `languages` must be lists (may be empty).
- `commit` must be string (if provided).
- Glob patterns must be valid fnmatch patterns (contain no invalid regex).
- Language names must match known languages (see `LANGUAGE_EXTENSIONS` mapping).

**Examples**:
```json
{
  "languages": ["python"],
  "include_globs": ["src/**/*.py"],
  "exclude_globs": ["**/test_*.py"]
}
```

---

### set_scope Response Schema

**Format**: JSON object

```python
{
    "effective_scope": ScopeIn,  # Scope that was stored
    "session_id": str,            # Session ID (UUID)
    "status": Literal["ok"]       # Always "ok" (errors raise exceptions)
}
```

**Example**:
```json
{
  "effective_scope": {
    "languages": ["python"],
    "include_globs": ["src/**"]
  },
  "session_id": "a3f8b2c1-4d5e-6789-abcd-ef0123456789",
  "status": "ok"
}
```

---

### AnswerEnvelope Extension

**Modified**: Added `scope` field to `AnswerEnvelope` TypedDict.

```python
class AnswerEnvelope(TypedDict, total=False):
    # ... existing fields ...
    scope: ScopeIn  # NEW: Effective scope applied to query
```

**Rationale**: Clients need to know which scope was applied to understand result limitations.

**Backward Compatibility**: Additive change only (existing clients ignore new field).

---

## API Endpoints

### set_scope

**Tool Name**: `set_scope`  
**Input**: `ScopeIn`  
**Output**: `{effective_scope: ScopeIn, session_id: str, status: "ok"}`

**Behavior**:
1. Extract session ID from `ContextVar` (set by middleware).
2. Store scope in registry: `context.scope_registry.set_scope(session_id, scope)`.
3. Return confirmation with session ID.

**Errors**:
- No session ID in ContextVar → `RuntimeError` (should never happen after middleware).

---

### list_paths

**Tool Name**: `list_paths`  
**Input**: `{path: str | None, include_globs: list[str] | None, exclude_globs: list[str] | None, max_results: int}`  
**Output**: `{items: list[dict], total: int, truncated: bool}`

**Scope Integration** (NEW):
- Retrieve session scope via `get_effective_scope(context, session_id)`.
- Merge scope's `include_globs`/`exclude_globs`/`languages` with explicit params.
- Apply merged filters during directory traversal.
- Language filter applied post-traversal (filter paths by extension).

**Precedence**: Explicit params override scope.

---

### search_text

**Tool Name**: `search_text`  
**Input**: `{query: str, regex: bool, case_sensitive: bool, paths: list[str] | None, max_results: int}`  
**Output**: `{matches: list[Match], total: int, truncated: bool}`

**Scope Integration** (NEW):
- Retrieve session scope.
- If explicit `paths` provided, use those (ignore scope).
- Otherwise, use scope's `include_globs` as search paths.
- Pass paths to ripgrep command.

**Precedence**: Explicit `paths` override scope.

---

### semantic_search

**Tool Name**: `semantic_search`  
**Input**: `{query: str, limit: int}`  
**Output**: `AnswerEnvelope` (with `findings`, `scope`, `method`, etc.)

**Scope Integration** (NEW):
- Retrieve session scope.
- Perform FAISS search (unchanged).
- Hydrate results via `catalog.query_by_filters(ids, include_globs=..., languages=...)` if scope has filters.
- Include applied scope in response envelope (`scope` field).

**Filtering**: Applied during DuckDB hydration (post-FAISS).

---

## Observability

### Metrics

**Gauge**: `codeintel_active_sessions`  
Description: Number of active sessions in registry.  
Labels: None  
Updated: On `set_scope` (increment) and `prune_expired` (decrement).

**Counter**: `codeintel_scope_operations_total`  
Description: Total scope operations.  
Labels: `operation` (set, get, clear, prune)  
Updated: On each registry operation.

**Histogram**: `codeintel_scope_filter_duration_seconds`  
Description: Time to apply scope filters.  
Labels: `filter_type` (glob, language, combined)  
Updated: In `query_by_filters` and `apply_language_filter`.

---

### Logs

**Event**: Scope storage  
Level: INFO  
Message: `"Set scope for session"`  
Fields: `session_id`, `scope` (dict)

**Event**: Scope retrieval  
Level: DEBUG  
Message: `"Retrieved scope for session"`  
Fields: `session_id`, `scope` (dict or null)

**Event**: Session pruning  
Level: INFO  
Message: `"Pruned expired sessions"`  
Fields: `pruned_count`, `max_age_seconds`

**Event**: Scope filtering  
Level: DEBUG  
Message: `"Applied scope filters"`  
Fields: `session_id`, `filter_type`, `matched_count`, `duration_ms`

---

## Dependencies

- **Phase 1 (codeintel-config-lifecycle-management)**: Requires `ApplicationContext` for registry initialization.
- **DuckDB**: Requires DuckDB catalog for chunk filtering (existing dependency).
- **FastAPI**: Requires FastAPI middleware system (existing dependency).
- **Python ≥3.11**: Requires `ContextVar` (stdlib since 3.7, enhanced in 3.11).

---

## Future Enhancements (Phase 3+)

### Multi-Repository Support

**Scope Field**: `repos: list[str]`  
**Behavior**: Select repository context based on scope; query multiple repositories if list contains >1 repo.

**Requirements** (deferred to Phase 3):
- `ApplicationContext.repositories: dict[str, RepositoryContext]` (map repo name to context).
- `ScopeRegistry.get_repository_context(session_id, repo_name)` (resolve repo from scope).
- Adapters check `scope.repos` and route to appropriate `RepositoryContext`.
- Index merging logic for cross-repo queries (union FAISS results, merge DuckDB queries).

---

### Branch/Commit Filtering

**Scope Fields**: `branches: list[str]`, `commit: str`  
**Behavior**: Filter results by Git branch or commit SHA.

**Requirements** (deferred to Phase 4):
- Git integration enhancements (query commit history, checkout branches).
- Chunk metadata includes branch/commit info (requires indexing changes).
- DuckDB catalog filters by branch/commit fields.

---

## Migration & Rollout

### Backward Compatibility

**Breaking Changes**: None (additive changes only).

**Existing Behavior Preserved**:
- Clients not calling `set_scope` continue to work (default = no filters).
- Existing queries without session ID get auto-generated ID (transparent).

### Rollout Plan

**Stage 1**: Deploy infrastructure (registry, middleware, utilities).  
**Stage 2**: Update adapters to apply scope (progressive rollout per adapter).  
**Stage 3**: Enable background pruning task.  
**Stage 4**: Monitor metrics and adjust thresholds if needed.

### Client Migration

**Optional Migration**: Clients can adopt scope incrementally.

**Steps for Clients**:
1. Start sending `X-Session-ID` header (optional but recommended for session persistence).
2. Call `set_scope` to establish constraints.
3. Verify subsequent queries respect scope.

**No Migration Required**: Clients using explicit parameters (e.g., `list_paths(include_globs=...)`) continue to work without changes.

---

## Glossary

**Session**: A logical grouping of requests from the same client, identified by session ID.  
**Scope**: A set of query constraints (path globs, languages, repos) applied to searches.  
**Session ID**: A UUID identifying a session (client-provided or auto-generated).  
**ContextVar**: Python's context variable for thread-local storage (stdlib `contextvars`).  
**Glob Pattern**: Wildcard pattern for path matching (e.g., `**/*.py`, `src/**`).  
**fnmatch**: Python stdlib module for Unix shell-style glob matching.  
**LRU**: Least Recently Used (eviction policy based on last access time).

