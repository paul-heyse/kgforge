## Why

The current `set_scope` MCP tool is a no-op: it returns `{"status": "ok"}` without storing or applying the requested scope to subsequent queries. This violates user expectations and Python API best practices—tools should implement advertised functionality. Users calling `set_scope(repos=["main"])` followed by `search_text("query")` expect the search to be constrained to the "main" repository, but the scope is silently ignored across all adapters.

This gap creates three critical issues:

1. **Broken Contract**: The MCP API exposes `ScopeIn` parameters (repos, branches, languages, path globs) that are documented and type-checked but have no runtime effect. This misleads clients into believing they can constrain searches when they cannot.

2. **No Multi-Repo Foundation**: The architecture assumes a single repository with global indexes (one FAISS file, one DuckDB catalog). As the system scales to index multiple repositories, there's no mechanism to select which repository's indexes to query or how to partition results.

3. **Inconsistent Filtering**: Some adapters (`list_paths`, `search_text`) support path-based filtering via direct parameters, but none honor a centrally-set scope. Semantic search ignores scope entirely—it retrieves chunks from all indexed files without path, language, or repository filters.

Fixing scope handling is required to:
- **Honor API Contracts**: Make `set_scope` functionally effective so subsequent queries respect the scope.
- **Enable Multi-Repo Support**: Establish architectural patterns for per-repository contexts, indexes, and metadata.
- **Unify Filtering**: Ensure all search operations (text, semantic, file listing) apply scope constraints consistently via a single source of truth.

## What Changes

This proposal introduces **session-scoped scope management** via FastAPI's session middleware and extends all adapters to apply scope filters. The implementation is staged to support the current single-repository deployment while laying groundwork for future multi-repository expansion.

### Core Changes

- **ADDED**: `ScopeRegistry` in `codeintel_rev/app/scope_registry.py` managing per-session scope state with thread-safe CRUD operations.
- **ADDED**: `SessionScopeMiddleware` in `codeintel_rev/app/middleware.py` attaching session IDs to requests and storing scope in `ContextVar` for thread-local access.
- **MODIFIED**: `ApplicationContext` to include `scope_registry: ScopeRegistry` instance initialized during lifespan.
- **MODIFIED**: `set_scope` adapter to store scope in the registry keyed by session ID instead of returning a no-op response.
- **MODIFIED**: All search/file adapters (`semantic_search`, `search_text`, `list_paths`) to:
  - Retrieve active scope from `ContextVar` (set by middleware).
  - Apply scope filters: path globs filter results, language filters constrain file types.
  - Merge explicit parameters with scope (explicit params take precedence).
- **MODIFIED**: `DuckDBCatalog` to add `query_by_filters` method applying SQL `WHERE` clauses for path patterns and language filtering.
- **ADDED**: Scope application utilities in `codeintel_rev/mcp_server/scope_utils.py`:
  - `merge_scope_filters`: combines session scope with explicit adapter parameters.
  - `apply_path_filters`: filters file paths against include/exclude globs from scope.
  - `apply_language_filter`: matches file extensions to language list from scope.
- **MODIFIED**: `semantic_search` to post-filter FAISS results via DuckDB using scope's path/language constraints.
- **ADDED**: Future-proofing stub `RepositoryContext` in `codeintel_rev/app/repository_context.py` with TODO comments for multi-repo support (Phase 3).

### Testing & Documentation

- **ADDED**: Unit tests (`tests/codeintel_rev/test_scope_registry.py`, `tests/codeintel_rev/test_scope_utils.py`) validating scope storage, retrieval, expiration, and filter merging logic.
- **ADDED**: Integration tests (`tests/codeintel_rev/test_scope_integration.py`) verifying end-to-end scope application: set scope → search → verify results match scope constraints.
- **MODIFIED**: Existing adapter tests to include scope-aware test cases (e.g., `test_semantic_adapter.py` adds `test_semantic_search_with_scope_filters`).
- **ADDED**: OpenAPI schema updates (`openapi/codeintel-scope.yaml`) documenting scope lifecycle and session semantics.
- **MODIFIED**: README and architecture docs (`codeintel_rev/README.md`, `docs/architecture/scope-management.md`) explaining scope usage patterns and multi-repo roadmap.

## Impact

### Specs
- **NEW**: `codeintel-scope-management` capability spec defining scope lifecycle, session binding, and filter application rules.
- **MODIFIED**: Existing MCP tool specs (`semantic-search`, `text-search`, `list-paths`) now reference scope filtering requirements and document precedence rules (explicit parameters override session scope).

### Code
- **Core**: `scope_registry.py`, `middleware.py`, `scope_utils.py` (new modules).
- **Adapters**: `semantic.py`, `text_search.py`, `files.py`, `history.py` (modified to apply scope).
- **Context**: `config_context.py` (add `scope_registry` field), `main.py` (register middleware, initialize registry).
- **Catalog**: `duckdb_catalog.py` (add `query_by_filters` method).
- **Tests**: 6 new test files covering unit and integration scenarios.

### Data Contracts
- **ScopeIn**: Enhanced documentation clarifying single-repo behavior (current) vs multi-repo future state.
- **AnswerEnvelope**: `scope` field now populated with the effective scope that was applied to the query (previously always empty).
- **Session Header**: Introduces `X-Session-ID` header for stateful session tracking (optional; falls back to auto-generated session ID if absent).

### Rollout / Dependencies
- **Phase 1 Complete**: Depends on `codeintel-config-lifecycle-management` being merged (requires `ApplicationContext`).
- **Backward Compatible**: Existing queries without `set_scope` continue to work (default scope = no filters).
- **Incremental Deployment**: Middleware and registry can be deployed without adapter changes; adapters updated one-by-one with feature flags.
- **Multi-Repo Readiness**: Lays architectural foundation for Phase 3 (multi-repo support) but does not implement repository switching (that requires registry infrastructure and separate index loading).

## Success Criteria

1. **Functional Scope**: After calling `set_scope(languages=["python"])`, subsequent `semantic_search` and `search_text` calls return only Python files.
2. **Session Isolation**: Two concurrent sessions with different scopes (Session A: Python only, Session B: TypeScript only) receive correctly filtered results without cross-contamination.
3. **Parameter Precedence**: Explicit adapter parameters (e.g., `search_text(paths=["src/"])`) override session scope when both are present.
4. **Zero Errors**: All type checkers (pyright, pyrefly), linters (ruff), and tests pass with zero suppressions.
5. **Performance**: Scope filtering adds <5ms overhead to search operations (measured via Prometheus metrics).
6. **Documentation**: Architecture docs include sequence diagrams showing scope lifecycle; all public APIs have NumPy-style docstrings with scope examples.

