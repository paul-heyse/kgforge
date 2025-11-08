# CodeIntel Scope State Management (Phase 2)

**Status**: Ready for Implementation  
**Phase**: 2 of 4 (Architecture Improvements)  
**Est. Duration**: 3 weeks (60 hours)  
**Owner**: CodeIntel Team

## Overview

This change proposal implements **session-scoped scope management** for the CodeIntel MCP server, making the `set_scope` tool functionally effective. Currently, `set_scope` is a no-op that returns success without storing or applying the scope—this proposal retrofits the system to honor scope constraints across all search and file operations.

### Problem Statement

Users calling `set_scope(languages=["python"])` expect subsequent queries to be constrained to Python files, but the scope is silently ignored. This breaks the API contract and prevents users from efficiently narrowing searches to relevant code subsets.

### Solution Summary

Introduce a **thread-safe in-memory scope registry** that stores per-session scopes, accessed via `ContextVar` for thread-local isolation. All adapters (`semantic_search`, `search_text`, `list_paths`, `file_history`, `blame_range`) retrieve and apply session scope automatically, with explicit parameters overriding scope when present.

### Key Benefits

1. **Functional Scope**: `set_scope` now stores scope that persists across requests within the same session.
2. **Session Isolation**: Concurrent sessions with different scopes don't interfere (each has isolated scope state).
3. **Explicit Precedence**: Explicit adapter parameters override session scope (allows one-off overrides without clearing scope).
4. **Multi-Repo Foundation**: Architecture supports future multi-repository indexing (Phase 3).
5. **Zero Breaking Changes**: Existing queries without `set_scope` continue to work (default = no filters).

## Document Structure

```
codeintel-scope-state-management/
├── README.md                       # This file: overview and navigation
├── proposal.md                     # Why, what changes, impact, success criteria
├── design.md                       # Detailed design: components, decisions, risks
├── tasks.md                        # Exhaustive task breakdown (25 tasks, 60 hours)
├── specs/
│   └── codeintel-scope-management/
│       └── spec.md                 # Capability spec: requirements, contracts, API
└── implementation/
    ├── scope_registry.py           # ScopeRegistry class (thread-safe storage)
    ├── middleware.py               # SessionScopeMiddleware (session ID injection)
    ├── scope_utils.py              # Utilities (merging, filtering)
    └── README.md                   # Implementation notes (TBD)
```

## Quick Links

- **Start Here**: [proposal.md](./proposal.md) for high-level context and rationale
- **Deep Dive**: [design.md](./design.md) for architectural decisions and component details
- **Implementation Plan**: [tasks.md](./tasks.md) for week-by-week task breakdown
- **Requirements**: [specs/codeintel-scope-management/spec.md](./specs/codeintel-scope-management/spec.md) for detailed contracts
- **Code**: [implementation/](./implementation/) for reference implementations

## Key Concepts

### Session Scope

A **session scope** is a set of query constraints (path globs, languages, repositories) that apply to all queries within a session. Instead of passing filters with every query, users call `set_scope` once and subsequent queries inherit those constraints.

**Example**:
```python
# Set scope once
mcp.call_tool("set_scope", {
    "languages": ["python"],
    "include_globs": ["src/**"],
    "exclude_globs": ["**/test_*"]
})

# Subsequent queries auto-apply scope
results = mcp.call_tool("semantic_search", {"query": "data processing"})
# Only Python files in src/ (excluding tests) are searched

# Override scope for one query
results = mcp.call_tool("list_paths", {"include_globs": ["lib/**"]})
# Searches lib/ directory (explicit param overrides scope)
```

### Session Management

Each session is identified by a UUID (auto-generated or client-provided via `X-Session-ID` header). Sessions expire after 1 hour of inactivity to prevent memory leaks. Background task prunes expired sessions every 10 minutes.

**Session Lifecycle**:
1. Client sends request (with or without `X-Session-ID` header)
2. Middleware extracts/generates session ID
3. Client calls `set_scope` to store constraints
4. Subsequent requests with same session ID retrieve and apply scope
5. Session expires after 1 hour of inactivity (or explicitly cleared)

### Scope Application

**Adapters** (functions handling MCP tool calls) retrieve session scope and apply filters:

- **`list_paths`**: Filters directory listing by globs and languages (applied during traversal).
- **`search_text`**: Passes scope paths to ripgrep command (pre-filtering).
- **`semantic_search`**: Filters FAISS results during DuckDB hydration (post-filtering via SQL).
- **`file_history`, `blame_range`**: (Future) Filter Git history by scope paths.

### Explicit Parameter Precedence

When both session scope and explicit adapter parameters are present, **explicit parameters win**. This allows users to override scope for specific queries without clearing it.

**Example**:
```python
# Session scope: Python files in src/
set_scope({"languages": ["python"], "include_globs": ["src/**"]})

# Query 1: Inherits session scope
semantic_search("query")  # Searches Python files in src/

# Query 2: Overrides scope with explicit parameter
list_paths(include_globs=["lib/**"])  # Searches lib/ (ignores session scope)

# Query 3: Session scope still active
search_text("import numpy")  # Searches Python files in src/
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌────────────────────────┐   │
│  │ SessionScope     │────────>│  ApplicationContext    │   │
│  │ Middleware       │         │  ┌──────────────────┐  │   │
│  │                  │         │  │ ScopeRegistry    │  │   │
│  │ - Extract header │         │  │                  │  │   │
│  │ - Generate UUID  │         │  │ - Thread-safe    │  │   │
│  │ - Set ContextVar │         │  │ - LRU expiration │  │   │
│  └──────────────────┘         │  └──────────────────┘  │   │
│           │                   │                        │   │
│           v                   │  ┌──────────────────┐  │   │
│  ┌──────────────────┐         │  │ FAISSManager     │  │   │
│  │ MCP Tool Handler │<────────│  │ DuckDBCatalog    │  │   │
│  │                  │         │  │ VLLMClient       │  │   │
│  │ - Retrieve scope │         │  └──────────────────┘  │   │
│  │ - Merge filters  │         └────────────────────────┘   │
│  │ - Apply to query │                                      │
│  └──────────────────┘                                      │
│           │                                                │
│           v                                                │
│  ┌──────────────────┐                                      │
│  │ Adapter Function │                                      │
│  │                  │                                      │
│  │ - semantic_search│                                      │
│  │ - search_text    │                                      │
│  │ - list_paths     │                                      │
│  └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Request → Response**:
1. **Request arrives** with `X-Session-ID` header (or auto-generate UUID)
2. **Middleware** stores session ID in `ContextVar`
3. **`set_scope` tool** stores scope in registry keyed by session ID
4. **Subsequent tool** retrieves scope via `get_effective_scope(context, session_id)`
5. **Adapter** merges scope with explicit params (explicit overrides)
6. **Apply filters**:
   - `list_paths`: During directory traversal (glob + language filters)
   - `search_text`: Pass to ripgrep command (path filters)
   - `semantic_search`: DuckDB SQL WHERE clauses (path + language filters)
7. **Return results** with applied scope included in response envelope

### Thread Safety

- **ScopeRegistry**: Uses `threading.RLock` for dict operations
- **ContextVar**: Provides thread-local storage (no cross-thread contamination)
- **Immutable Results**: `get_scope` returns copies (prevents caller mutations)

## Implementation Phases

### Phase 2a: Infrastructure (Week 1)
- Implement `ScopeRegistry` with unit tests
- Implement `SessionScopeMiddleware` and register in FastAPI
- Add `scope_registry` to `ApplicationContext`
- Add background pruning task for expired sessions
- **Deliverable**: Registry operational, sessions stored/retrieved

### Phase 2b: Utilities (Week 1)
- Implement `scope_utils.py` functions (merge, filter, language mapping)
- Write comprehensive unit tests for utilities
- **Deliverable**: Scope merging and filtering functions ready for adapter use

### Phase 2c: Adapter Integration (Week 2)
- Update `set_scope` to store scope in registry
- Update `list_paths`, `search_text`, `semantic_search` to apply scope
- **Deliverable**: All adapters honor session scope

### Phase 2d: DuckDB Extensions (Week 3)
- Implement `DuckDBCatalog.query_by_filters` with SQL generation
- Add DuckDB index on `uri` column for performance
- Update `semantic_search` to use `query_by_filters`
- **Deliverable**: Semantic search filters via DuckDB

### Phase 2e: Integration Testing (Week 3)
- Write end-to-end scope integration tests
- Write multi-session isolation tests
- Write performance benchmarks
- **Deliverable**: All integration tests pass, performance within targets

### Phase 2f: Documentation & Rollout (Week 4)
- Update README with scope usage examples
- Create architecture diagrams (sequence, class)
- Update OpenAPI spec with scope API
- Write migration guide for clients
- **Deliverable**: Documentation complete, ready for rollout

## Testing Strategy

### Unit Tests (Per Component)
- **ScopeRegistry**: CRUD operations, expiration, thread safety
- **Middleware**: Session ID extraction/generation, ContextVar setting
- **Scope Utilities**: Merge logic, path filtering, language filtering
- **DuckDB Catalog**: `query_by_filters` with various glob/language combinations

### Integration Tests (End-to-End)
- **Scope Application**: Set scope → call adapters → verify results match scope
- **Session Isolation**: Concurrent requests with different session IDs → no cross-contamination
- **Parameter Precedence**: Explicit params override scope
- **Expiration**: Mock time advance → verify pruning

### Performance Tests
- **Baseline**: Measure search latency without scope
- **Scope Overhead**: Measure latency with scope (target: <5ms delta)
- **Large Result Sets**: Query 100K chunks with language filter (target: <1 second)

### Negative Tests
- **Invalid Session ID**: Graceful fallback (no scope)
- **Unsupported Glob Patterns**: Clear error message
- **Empty Scope**: Behaves like no scope

## Success Criteria

✅ **Functional**:
- Calling `set_scope(languages=["python"])` followed by `semantic_search` returns only Python files
- Concurrent sessions with different scopes receive correctly filtered results
- Explicit adapter parameters override session scope

✅ **Quality**:
- Zero pyright/pyrefly/ruff errors
- 100% unit test coverage for new modules
- All integration tests pass (no flakiness)

✅ **Performance**:
- Scope filtering adds <5ms overhead (measured via Prometheus)
- Memory usage <100MB for 1000 active sessions

✅ **Documentation**:
- README includes runnable examples
- Architecture diagrams show scope lifecycle
- OpenAPI spec documents scope API

## Rollout Plan

### Stage 1: Deploy Infrastructure
- Deploy registry, middleware, utilities
- Monitor for errors (expect zero impact on existing queries)

### Stage 2: Enable Adapters
- Deploy updated adapters (progressive rollout: `list_paths` → `search_text` → `semantic_search`)
- Monitor metrics (`codeintel_scope_filter_duration_seconds`)

### Stage 3: Enable Pruning
- Start background pruning task
- Monitor session count gauge (`codeintel_active_sessions`)

### Stage 4: Rollout Complete
- Publish documentation
- Notify clients of scope availability

## Future Enhancements (Phase 3+)

### Multi-Repository Support
- **Phase 3**: Implement `RepositoryContext` with per-repo indexes
- **Scope Field**: `repos: list[str]` selects which repositories to query
- **Architecture**: `ApplicationContext.repositories: dict[str, RepositoryContext]`
- **Adapters**: Route queries to appropriate repo contexts based on scope

### Branch/Commit Filtering
- **Phase 4**: Filter results by Git branch or commit SHA
- **Scope Fields**: `branches: list[str]`, `commit: str`
- **Requirements**: Chunk metadata includes branch/commit, DuckDB filters by these fields

## Dependencies

- **Phase 1 (Config Lifecycle)**: MUST be merged first (requires `ApplicationContext`)
- **FastAPI**: Existing dependency (middleware system)
- **DuckDB**: Existing dependency (catalog queries)
- **Python ≥3.11**: Required for ContextVar enhancements

## FAQ

### Q: What happens if I don't call `set_scope`?
**A**: Queries work as before (no filtering). Scope is optional—existing clients don't need to change.

### Q: How do I reset scope?
**A**: Call `set_scope` with empty fields: `set_scope({})`. This clears all filters.

### Q: Can I use multiple scopes in one session?
**A**: No, one scope per session. Calling `set_scope` again overwrites the previous scope.

### Q: How do I disable scope for one query?
**A**: Pass explicit parameters: `list_paths(include_globs=["**"])` searches all files regardless of scope.

### Q: Will scope work with multi-repo queries?
**A**: Not yet. Multi-repo support is Phase 3. For now, scope applies to the single indexed repository.

### Q: What if my glob pattern doesn't work?
**A**: We support standard fnmatch globs (`*`, `?`, `[seq]`). Complex patterns (e.g., `{a,b}`) are not supported (fall back to Python filtering). See `LANGUAGE_EXTENSIONS` mapping for supported languages.

## Contact & Support

- **Owner**: CodeIntel Team
- **Slack**: #codeintel-mcp
- **Issues**: GitHub Issues (label: `scope-management`)
- **Docs**: `docs/architecture/scope-management.md` (after implementation)

