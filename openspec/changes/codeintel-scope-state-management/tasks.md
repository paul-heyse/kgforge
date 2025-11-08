# Implementation Tasks

## Phase 2a: Core Infrastructure (Week 1)

### Task 1: Implement ScopeRegistry
**File**: `codeintel_rev/app/scope_registry.py`

**Description**: Create thread-safe in-memory registry for storing session scopes.

**Subtasks**:
1. Define `ScopeRegistry` class with `threading.RLock` for thread safety.
2. Implement `set_scope(session_id: str, scope: ScopeIn) -> None`:
   - Store scope in internal dict: `{session_id: (scope, timestamp)}`.
   - Update timestamp to `time.monotonic()`.
   - Log scope storage with structured logging.
3. Implement `get_scope(session_id: str) -> ScopeIn | None`:
   - Retrieve scope from dict, return None if not found.
   - Update timestamp on access (for LRU tracking).
   - Return copy of scope dict (prevent mutation).
4. Implement `clear_scope(session_id: str) -> None`:
   - Remove entry from dict if exists.
   - Log scope removal.
5. Implement `prune_expired(max_age_seconds: int) -> int`:
   - Iterate dict, calculate age = `time.monotonic() - timestamp`.
   - Remove entries where age > max_age_seconds.
   - Return count of pruned entries.
   - Log pruning stats.
6. Add comprehensive NumPy docstrings with examples.
7. Add type hints for all methods (strict mode compliant).

**Acceptance**:
- pyright reports zero errors.
- All methods have docstrings with Examples section.
- Ruff formatting clean.

**Time Estimate**: 3 hours

---

### Task 2: Implement SessionScopeMiddleware
**File**: `codeintel_rev/app/middleware.py` (NEW)

**Description**: FastAPI middleware for session ID management.

**Subtasks**:
1. Create `middleware.py` with module docstring.
2. Define `session_id_var: ContextVar[str | None]` at module level.
3. Implement `SessionScopeMiddleware` class:
   - Inherit from `starlette.middleware.base.BaseHTTPMiddleware`.
   - Extract `X-Session-ID` header from request.
   - Generate UUID if header missing: `str(uuid.uuid4())`.
   - Set `request.state.session_id = session_id`.
   - Set `session_id_var.set(session_id)` for ContextVar access.
   - Call `await call_next(request)`.
   - Return response (no header modification needed—FastMCP limitation).
4. Add `get_session_id() -> str` helper:
   - Return `session_id_var.get()` or raise `RuntimeError` if not set.
5. Write comprehensive docstrings.
6. Add type hints (strict mode).

**Acceptance**:
- Middleware sets `request.state.session_id` correctly.
- `ContextVar` is set and accessible in nested calls.
- `get_session_id()` raises `RuntimeError` if called outside request context.

**Time Estimate**: 2 hours

---

### Task 3: Register Middleware in FastAPI
**File**: `codeintel_rev/app/main.py`

**Description**: Integrate `SessionScopeMiddleware` into application.

**Subtasks**:
1. Import `SessionScopeMiddleware` from `codeintel_rev.app.middleware`.
2. Add middleware registration in `lifespan` after app creation:
   ```python
   app.add_middleware(SessionScopeMiddleware)
   ```
3. Document middleware registration with inline comment.
4. Verify middleware runs before MCP tool handlers (FastAPI middleware order).

**Acceptance**:
- Middleware is registered in correct order.
- Requests have `request.state.session_id` set.

**Time Estimate**: 30 minutes

---

### Task 4: Add ScopeRegistry to ApplicationContext
**File**: `codeintel_rev/app/config_context.py`

**Description**: Extend `ApplicationContext` to include scope registry.

**Subtasks**:
1. Import `ScopeRegistry` from `codeintel_rev.app.scope_registry`.
2. Add `scope_registry: ScopeRegistry` field to `ApplicationContext` dataclass.
3. Update `ApplicationContext.create()`:
   - Initialize `scope_registry = ScopeRegistry()` before returning context.
   - Log registry creation.
4. Update class docstring to document `scope_registry` field.
5. Update examples in docstring showing registry access.

**Acceptance**:
- `ApplicationContext` has `scope_registry` field.
- Registry is initialized in `create()` method.
- Docstrings updated.

**Time Estimate**: 1 hour

---

### Task 5: Add Background Pruning Task
**File**: `codeintel_rev/app/main.py`

**Description**: Implement background task for pruning expired sessions.

**Subtasks**:
1. Add `prune_expired_sessions` async function:
   ```python
   async def prune_expired_sessions(context: ApplicationContext) -> None:
       while True:
           await asyncio.sleep(600)  # 10 minutes
           pruned = context.scope_registry.prune_expired(max_age_seconds=3600)
           LOGGER.info(f"Pruned {pruned} expired sessions")
   ```
2. Start background task in `lifespan`:
   ```python
   prune_task = asyncio.create_task(prune_expired_sessions(context))
   ```
3. Cancel task on shutdown:
   ```python
   prune_task.cancel()
   await prune_task
   ```
4. Add `SESSION_MAX_AGE_SECONDS` environment variable support (default 3600).
5. Document pruning strategy in inline comments.

**Acceptance**:
- Background task starts on app startup.
- Task prunes expired sessions every 10 minutes.
- Task cancels cleanly on shutdown.

**Time Estimate**: 2 hours

---

### Task 6: Write ScopeRegistry Unit Tests
**File**: `tests/codeintel_rev/test_scope_registry.py` (NEW)

**Description**: Comprehensive unit tests for `ScopeRegistry`.

**Subtasks**:
1. Test `set_scope` and `get_scope`:
   - Set scope → retrieve → assert equal.
   - Set scope twice → verify second overwrites first.
   - Get non-existent session → assert None.
2. Test `clear_scope`:
   - Set scope → clear → get → assert None.
   - Clear non-existent session → no error.
3. Test `prune_expired`:
   - Set scope → mock time advance (2 hours) → prune(max_age=3600) → assert pruned.
   - Set scope → prune immediately → assert not pruned.
4. Test thread safety:
   - Use `threading.Thread` to call `set_scope` from 10 concurrent threads.
   - Verify all scopes stored correctly (no race conditions).
5. Test timestamp updates:
   - Set scope → get scope → verify timestamp updated.
   - Mock time to test LRU behavior.
6. Use `@pytest.mark.parametrize` for edge cases:
   - Empty scope dict.
   - Scope with all fields populated.
   - Scope with only `include_globs`.

**Acceptance**:
- 100% code coverage for `ScopeRegistry`.
- All tests pass with `pytest -q`.
- No flakiness (run 100 times).

**Time Estimate**: 4 hours

---

## Phase 2b: Scope Utilities (Week 1)

### Task 7: Implement Scope Utilities
**File**: `codeintel_rev/mcp_server/scope_utils.py` (NEW)

**Description**: Helper functions for scope retrieval and filtering.

**Subtasks**:
1. Implement `get_effective_scope(context: ApplicationContext, session_id: str | None) -> ScopeIn | None`:
   - If `session_id` is None, return None.
   - Call `context.scope_registry.get_scope(session_id)`.
   - Return result (None if not found).
2. Implement `merge_scope_filters(scope: ScopeIn | None, explicit_params: dict) -> dict`:
   - Start with scope fields as defaults: `{"include_globs": scope.get("include_globs"), ...}`.
   - Override with explicit params: `merged.update({k: v for k, v in explicit_params.items() if v is not None})`.
   - Return merged dict.
3. Implement `apply_path_filters(paths: list[str], include_globs: list[str], exclude_globs: list[str]) -> list[str]`:
   - For each path, check if matches any include glob (using `fnmatch.fnmatch`).
   - Exclude paths matching any exclude glob.
   - Return filtered list.
4. Implement `apply_language_filter(paths: list[str], languages: list[str]) -> list[str]`:
   - Define `LANGUAGE_EXTENSIONS` dict:
     ```python
     LANGUAGE_EXTENSIONS = {
         "python": [".py", ".pyi"],
         "typescript": [".ts", ".tsx"],
         "javascript": [".js", ".jsx", ".mjs", ".cjs"],
         "rust": [".rs"],
         "go": [".go"],
         "java": [".java"],
         "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
         "c": [".c", ".h"],
         "ruby": [".rb"],
         "php": [".php"],
         "swift": [".swift"],
         "kotlin": [".kt", ".kts"],
         "scala": [".scala"],
         "bash": [".sh", ".bash"],
         "powershell": [".ps1"],
         "yaml": [".yaml", ".yml"],
         "json": [".json"],
         "toml": [".toml"],
         "markdown": [".md", ".markdown"],
     }
     ```
   - For each path, check if extension matches any language's extensions.
   - Return filtered list.
5. Implement `path_matches_glob(path: str, pattern: str) -> bool`:
   - Normalize path separators: `path.replace("\\", "/")`.
   - Use `fnmatch.fnmatch(normalized_path, pattern)`.
6. Add comprehensive docstrings with examples for each function.
7. Add type hints (strict mode).

**Acceptance**:
- All functions have docstrings with Examples section.
- Pyright reports zero errors.
- Ruff formatting clean.

**Time Estimate**: 3 hours

---

### Task 8: Write Scope Utilities Unit Tests
**File**: `tests/codeintel_rev/test_scope_utils.py` (NEW)

**Description**: Comprehensive tests for scope utilities.

**Subtasks**:
1. Test `get_effective_scope`:
   - Valid session ID with scope → returns scope.
   - Valid session ID without scope → returns None.
   - None session ID → returns None.
2. Test `merge_scope_filters`:
   - Scope only → returns scope fields.
   - Explicit params only → returns params.
   - Both scope and params → params override scope.
   - Empty scope and empty params → returns empty dict.
3. Test `apply_path_filters`:
   - Include globs: `["**/*.py"]` → filters to Python files only.
   - Exclude globs: `["**/test_*.py"]` → removes test files.
   - Both include and exclude → applies both filters.
   - Empty globs → returns all paths.
4. Test `apply_language_filter`:
   - Language `["python"]` → returns only `.py` and `.pyi` files.
   - Multiple languages `["python", "typescript"]` → returns `.py`, `.pyi`, `.ts`, `.tsx`.
   - Unknown language `["cobol"]` → returns empty list.
5. Test `path_matches_glob`:
   - Simple glob: `*.py` matches `test.py`, not `src/test.py`.
   - Recursive glob: `**/*.py` matches `src/test.py`.
   - Windows paths: handles `\` correctly.
6. Use `@pytest.mark.parametrize` for combinatorial testing.

**Acceptance**:
- 100% code coverage for `scope_utils.py`.
- All tests pass.
- Edge cases covered (empty lists, None values).

**Time Estimate**: 4 hours

---

## Phase 2c: Adapter Integration (Week 2)

### Task 9: Update `set_scope` Adapter
**File**: `codeintel_rev/mcp_server/adapters/files.py`

**Description**: Make `set_scope` store scope in registry.

**Subtasks**:
1. Import `get_session_id` from `codeintel_rev.app.middleware`.
2. Update `set_scope` function:
   - Call `session_id = get_session_id()` to get current session ID.
   - Call `context.scope_registry.set_scope(session_id, scope)`.
   - Return `{"effective_scope": scope, "session_id": session_id, "status": "ok"}`.
3. Update docstring with example showing session ID in response.
4. Add logging: `LOGGER.info("Set scope for session", extra={"session_id": session_id, "scope": scope})`.

**Acceptance**:
- `set_scope` stores scope in registry.
- Response includes session ID.
- Logging captures scope storage event.

**Time Estimate**: 1 hour

---

### Task 10: Update `list_paths` Adapter
**File**: `codeintel_rev/mcp_server/adapters/files.py`

**Description**: Apply scope filters to file listing.

**Subtasks**:
1. Import `get_session_id`, `get_effective_scope`, `merge_scope_filters`, `apply_language_filter` from respective modules.
2. At start of `list_paths`:
   - Call `session_id = get_session_id()`.
   - Call `scope = get_effective_scope(context, session_id)`.
   - Call `merged = merge_scope_filters(scope, {"include_globs": include_globs, "exclude_globs": exclude_globs})`.
   - Extract merged globs: `includes = merged.get("include_globs") or ["**"]`.
   - Extract merged excludes: `excludes = (merged.get("exclude_globs") or []) + default_excludes`.
3. After building `items` list, apply language filter:
   ```python
   if scope and scope.get("languages"):
       filtered_paths = apply_language_filter([item["path"] for item in items], scope["languages"])
       items = [item for item in items if item["path"] in filtered_paths]
   ```
4. Update docstring:
   - Add "Scope Integration" section explaining how session scope is applied.
   - Add example: set scope → list paths → verify filtered results.
5. Update logging to include applied scope.

**Acceptance**:
- `list_paths` applies session scope filters.
- Explicit parameters override scope.
- Docstring documents scope behavior.
- Tests verify filtering (see Task 14).

**Time Estimate**: 2 hours

---

### Task 11: Update `search_text` Adapter
**File**: `codeintel_rev/mcp_server/adapters/text_search.py`

**Description**: Apply scope path filters to text search.

**Subtasks**:
1. Import scope utilities.
2. At start of `search_text`:
   - Get session scope via `get_effective_scope`.
   - Merge scope `include_globs` with explicit `paths` parameter.
   - Use merged paths in `_build_ripgrep_command`.
3. Logic for merging paths:
   ```python
   scope = get_effective_scope(context, get_session_id())
   if paths:
       effective_paths = paths  # Explicit paths override scope
   elif scope and scope.get("include_globs"):
       effective_paths = scope["include_globs"]
   else:
       effective_paths = None  # Search all (ripgrep default)
   ```
4. Update `_build_ripgrep_command` to accept `paths` parameter.
5. Update docstring with scope integration section.
6. Add logging for applied scope.

**Acceptance**:
- `search_text` respects session scope paths.
- Explicit `paths` parameter overrides scope.
- Docstring updated.
- Tests verify filtering (see Task 15).

**Time Estimate**: 2 hours

---

### Task 12: Update `semantic_search` Adapter (Part 1: Scope Retrieval)
**File**: `codeintel_rev/mcp_server/adapters/semantic.py`

**Description**: Retrieve and document scope (defer filtering to Task 13).

**Subtasks**:
1. Import scope utilities.
2. At start of `semantic_search`:
   - Get session scope via `get_effective_scope`.
   - Store scope for later use (after FAISS search).
3. Update return envelope to include applied scope:
   ```python
   return {
       "findings": findings,
       "scope": scope or {},  # Include effective scope in response
       "method": {...},
       # ... rest of envelope
   }
   ```
4. Update docstring:
   - Add "Scope Integration" section.
   - Document that scope filters are applied during DuckDB hydration.
5. Add logging for scope retrieval.

**Acceptance**:
- `semantic_search` retrieves session scope.
- Response includes `scope` field.
- Docstring documents scope behavior.

**Time Estimate**: 1 hour

---

## Phase 2d: DuckDB Catalog Extensions (Week 3)

### Task 13: Implement `DuckDBCatalog.query_by_filters`
**File**: `codeintel_rev/io/duckdb_catalog.py`

**Description**: Add method to filter chunks by path and language.

**Subtasks**:
1. Add method signature:
   ```python
   def query_by_filters(
       self,
       ids: Sequence[int],
       *,
       include_globs: list[str] | None = None,
       exclude_globs: list[str] | None = None,
       languages: list[str] | None = None,
   ) -> list[dict]:
   ```
2. Implement SQL generation:
   - Start with base query: `SELECT c.* FROM chunks AS c WHERE c.id IN (...)`.
   - Add include globs as `AND (c.uri LIKE ? OR c.uri LIKE ? ...)`.
   - Add exclude globs as `AND c.uri NOT LIKE ? AND c.uri NOT LIKE ? ...`.
   - Add language filter as `AND (c.uri LIKE '%.py' OR c.uri LIKE '%.ts' ...)`.
3. Convert globs to SQL `LIKE` patterns:
   - `**/*.py` → `%.py` (simple suffix match).
   - `src/**` → `src/%` (prefix match).
   - Complex globs (e.g., `src/**/test_*.py`) → fall back to Python `fnmatch` post-filtering.
4. Execute query with parameterized SQL.
5. Post-filter results using Python `fnmatch` for complex globs.
6. Add comprehensive docstring with SQL examples.
7. Add type hints (strict mode).

**Acceptance**:
- `query_by_filters` returns correctly filtered chunks.
- SQL generation handles simple globs.
- Complex globs fall back to Python filtering.
- Docstring includes SQL examples.

**Time Estimate**: 4 hours

---

### Task 14: Add DuckDB Index for URI Column
**File**: `codeintel_rev/io/duckdb_catalog.py`

**Description**: Optimize path filtering with index.

**Subtasks**:
1. Update `_ensure_views` to create index:
   ```python
   self.conn.execute("CREATE INDEX IF NOT EXISTS idx_uri ON chunks(uri)")
   ```
2. Document index creation in inline comment.
3. Add logging: `LOGGER.info("Created DuckDB index on uri column")`.

**Acceptance**:
- Index is created on first catalog open.
- Index creation is idempotent (repeated calls don't fail).
- Logging confirms index creation.

**Time Estimate**: 30 minutes

---

### Task 15: Update `semantic_search` to Use `query_by_filters`
**File**: `codeintel_rev/mcp_server/adapters/semantic.py`

**Description**: Apply scope filters during DuckDB hydration.

**Subtasks**:
1. After FAISS search, check if scope has filters:
   ```python
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
   ```
2. Handle case where filtering reduces results below `limit`:
   - Log warning: `LOGGER.warning("Scope filtering reduced results", extra={"requested": limit, "actual": len(chunks)})`.
   - Consider over-fetching from FAISS (request `limit * 2`) to compensate (optional optimization).
3. Update docstring to explain scope filtering happens during hydration.
4. Add logging for filtered chunk count.

**Acceptance**:
- `semantic_search` applies scope filters via DuckDB.
- Results match scope constraints.
- Logging confirms filtering.
- Tests verify filtering (see Task 17).

**Time Estimate**: 2 hours

---

### Task 16: Write DuckDB Catalog Unit Tests
**File**: `tests/codeintel_rev/test_duckdb_catalog.py`

**Description**: Test `query_by_filters` method.

**Subtasks**:
1. Set up test fixtures:
   - Create in-memory DuckDB catalog.
   - Insert test chunks with various URIs and languages.
2. Test include globs:
   - Query with `include_globs=["**/*.py"]` → assert only Python files returned.
   - Query with `include_globs=["src/**"]` → assert only `src/` files returned.
3. Test exclude globs:
   - Query with `exclude_globs=["**/test_*.py"]` → assert no test files returned.
4. Test language filter:
   - Query with `languages=["python"]` → assert only `.py` and `.pyi` files returned.
5. Test combined filters:
   - Query with include, exclude, and languages → assert all filters applied.
6. Test complex globs (fallback to Python filtering):
   - Query with `include_globs=["src/**/test_*.py"]` → assert correct results.
7. Test empty results:
   - Query with filters matching no chunks → assert empty list returned.
8. Test no filters:
   - Query with all filter params None → assert behaves like `query_by_ids`.
9. Use `@pytest.mark.parametrize` for combinatorial testing.

**Acceptance**:
- 100% code coverage for `query_by_filters`.
- All tests pass.
- Edge cases covered (empty filters, no matches, complex globs).

**Time Estimate**: 5 hours

---

## Phase 2e: Integration Testing (Week 3)

### Task 17: Write End-to-End Scope Integration Tests
**File**: `tests/codeintel_rev/test_scope_integration.py` (NEW)

**Description**: Test scope application across all adapters.

**Subtasks**:
1. Test `set_scope` → `list_paths`:
   - Set scope with `include_globs=["**/*.py"]`.
   - Call `list_paths`.
   - Assert only Python files returned.
2. Test `set_scope` → `search_text`:
   - Set scope with `include_globs=["src/**"]`.
   - Call `search_text("query")`.
   - Assert all matches are in `src/` directory.
3. Test `set_scope` → `semantic_search`:
   - Set scope with `languages=["python"]`.
   - Call `semantic_search("query")`.
   - Assert all findings are Python files.
4. Test parameter override:
   - Set scope with `include_globs=["**/*.py"]`.
   - Call `list_paths(include_globs=["**/*.ts"])`.
   - Assert only TypeScript files returned (explicit overrides scope).
5. Test multi-session isolation:
   - Session A: set scope with `languages=["python"]`.
   - Session B: set scope with `languages=["typescript"]`.
   - Call `semantic_search` from both sessions concurrently.
   - Assert Session A gets Python files, Session B gets TypeScript files.
6. Test scope expiration:
   - Set scope.
   - Mock time advance (2 hours).
   - Call `prune_expired`.
   - Call adapter → assert scope no longer applied (searches all files).
7. Test no scope (default behavior):
   - Don't call `set_scope`.
   - Call adapters → assert searches all files (no filtering).
8. Use `pytest.mark.asyncio` for async tests.
9. Use `httpx.AsyncClient` to simulate concurrent sessions.

**Acceptance**:
- All integration tests pass.
- No flakiness (run 100 times).
- Tests cover all critical user journeys.

**Time Estimate**: 6 hours

---

### Task 18: Write Adapter Unit Tests for Scope
**File**: `tests/codeintel_rev/test_semantic_adapter.py`, `test_text_search_adapter.py`, `test_files_adapter.py`

**Description**: Add scope-specific test cases to existing adapter tests.

**Subtasks**:
1. **`test_semantic_adapter.py`**:
   - Add `test_semantic_search_with_scope_filters`:
     - Mock session scope with `languages=["python"]`.
     - Call `semantic_search`.
     - Assert only Python files in results.
   - Add `test_semantic_search_no_scope`:
     - No session scope set.
     - Call `semantic_search`.
     - Assert all files in results.
2. **`test_text_search_adapter.py`**:
   - Add `test_search_text_with_scope_paths`:
     - Mock session scope with `include_globs=["src/**"]`.
     - Call `search_text`.
     - Assert all matches in `src/` directory.
   - Add `test_search_text_explicit_override`:
     - Mock session scope with `include_globs=["src/**"]`.
     - Call `search_text(paths=["lib/"])`.
     - Assert all matches in `lib/` directory (explicit overrides scope).
3. **`test_files_adapter.py`**:
   - Add `test_list_paths_with_scope_globs`:
     - Mock session scope with `include_globs=["**/*.py"]`.
     - Call `list_paths`.
     - Assert only Python files returned.
   - Add `test_list_paths_with_scope_language`:
     - Mock session scope with `languages=["python"]`.
     - Call `list_paths`.
     - Assert only Python files returned.
4. Use `unittest.mock` to mock `get_session_id()` and `context.scope_registry.get_scope()`.

**Acceptance**:
- Each adapter has ≥2 scope-specific tests.
- Tests use mocking to isolate scope behavior.
- All tests pass.

**Time Estimate**: 4 hours

---

## Phase 2f: Documentation & Rollout (Week 4)

### Task 19: Update README with Scope Usage
**File**: `codeintel_rev/README.md`

**Description**: Document scope functionality for users.

**Subtasks**:
1. Add "Scope Management" section:
   - Explain what scope is (query constraints).
   - List supported scope fields (repos, branches, globs, languages).
   - Provide copy-paste examples:
     ```python
     # Set scope to search only Python files
     mcp.call_tool("set_scope", {"languages": ["python"]})
     
     # Subsequent searches respect scope
     results = mcp.call_tool("semantic_search", {"query": "data processing"})
     # Only Python files in results
     
     # Override scope with explicit parameters
     results = mcp.call_tool("list_paths", {"include_globs": ["**/*.ts"]})
     # Only TypeScript files (scope ignored)
     ```
2. Add "Session Management" section:
   - Explain session IDs (auto-generated or client-provided).
   - Document `X-Session-ID` header usage.
   - Explain session expiration (1 hour default).
3. Add "Multi-Repo Support (Future)" section:
   - Document current single-repo limitation.
   - Explain that `repos` field is reserved for Phase 3.
4. Add troubleshooting section:
   - "Scope not applied" → check session ID consistency.
   - "Unexpected results" → verify explicit parameters don't override scope.

**Acceptance**:
- README includes runnable examples.
- Troubleshooting section covers common issues.
- Multi-repo future state documented.

**Time Estimate**: 3 hours

---

### Task 20: Create Architecture Diagram
**File**: `docs/architecture/scope-management.md` (NEW)

**Description**: Visual documentation of scope lifecycle.

**Subtasks**:
1. Create Mermaid sequence diagram:
   ```mermaid
   sequenceDiagram
       participant Client
       participant Middleware
       participant Adapter
       participant Registry
       participant DuckDB
       
       Client->>Middleware: Request (X-Session-ID: abc123)
       Middleware->>Middleware: Extract/generate session ID
       Middleware->>Adapter: Call set_scope(scope)
       Adapter->>Registry: store(session_id, scope)
       Registry-->>Adapter: OK
       Adapter-->>Client: {"session_id": "abc123", "status": "ok"}
       
       Client->>Middleware: Request (X-Session-ID: abc123)
       Middleware->>Adapter: Call semantic_search(query)
       Adapter->>Registry: get(session_id)
       Registry-->>Adapter: scope
       Adapter->>DuckDB: query_by_filters(ids, filters)
       DuckDB-->>Adapter: filtered_chunks
       Adapter-->>Client: {"findings": [...], "scope": {...}}
   ```
2. Add written explanation of each step.
3. Document data flow: request → middleware → adapter → registry → DuckDB → response.
4. Add class diagram showing relationships:
   - `ApplicationContext` → `ScopeRegistry`
   - `Adapter` → `scope_utils` → `ScopeRegistry`
   - `DuckDBCatalog` → SQL filters
5. Document threading model (ContextVar, thread safety).

**Acceptance**:
- Diagrams render correctly in Markdown viewers.
- Documentation explains all components and data flow.

**Time Estimate**: 4 hours

---

### Task 21: Update OpenAPI Spec
**File**: `openapi/codeintel-scope.yaml` (NEW)

**Description**: Document scope API in OpenAPI format.

**Subtasks**:
1. Define `ScopeIn` schema:
   ```yaml
   ScopeIn:
     type: object
     properties:
       repos:
         type: array
         items:
           type: string
         description: Repository names (future: multi-repo support)
       branches:
         type: array
         items:
           type: string
         description: Branch names (future: branch filtering)
       commit:
         type: string
         description: Commit SHA (future: commit filtering)
       include_globs:
         type: array
         items:
           type: string
         description: Path patterns to include (e.g., ["**/*.py"])
       exclude_globs:
         type: array
         items:
           type: string
         description: Path patterns to exclude (e.g., ["**/test_*.py"])
       languages:
         type: array
         items:
           type: string
         description: Programming languages (e.g., ["python", "typescript"])
   ```
2. Define `set_scope` endpoint:
   ```yaml
   /mcp/tools/set_scope:
     post:
       summary: Set query scope
       requestBody:
         content:
           application/json:
             schema:
               $ref: '#/components/schemas/ScopeIn'
       responses:
         '200':
           description: Scope set successfully
           content:
             application/json:
               schema:
                 type: object
                 properties:
                   effective_scope:
                     $ref: '#/components/schemas/ScopeIn'
                   session_id:
                     type: string
                   status:
                     type: string
   ```
3. Document `X-Session-ID` header:
   ```yaml
   parameters:
     - name: X-Session-ID
       in: header
       required: false
       schema:
         type: string
         format: uuid
       description: Session identifier (auto-generated if not provided)
   ```
4. Add examples for each field.
5. Document scope precedence rules (explicit params override scope).

**Acceptance**:
- OpenAPI spec validates against OpenAPI 3.1 schema.
- Spec includes examples and descriptions for all fields.

**Time Estimate**: 3 hours

---

### Task 22: Write Migration Guide
**File**: `docs/migration/scope-management-v2.md` (NEW)

**Description**: Guide for clients upgrading to scope-aware version.

**Subtasks**:
1. Document breaking changes:
   - `AnswerEnvelope` now includes `scope` field (additive, not breaking).
   - `set_scope` response includes `session_id` (additive).
2. Provide upgrade steps:
   - Step 1: Update client to send `X-Session-ID` header (optional but recommended).
   - Step 2: Call `set_scope` to establish query constraints.
   - Step 3: Verify subsequent queries respect scope.
3. Document backward compatibility:
   - Clients not using `set_scope` continue to work (default = no filters).
   - Existing queries without session ID get auto-generated ID.
4. Provide rollback plan:
   - If issues arise, clients can ignore `set_scope` and pass filters explicitly.
5. Add FAQ:
   - "How do I reset scope?" → Call `set_scope` with empty fields.
   - "How do I disable scope for one query?" → Pass explicit parameters.

**Acceptance**:
- Migration guide covers all upgrade scenarios.
- FAQ addresses common questions.

**Time Estimate**: 2 hours

---

## Phase 2g: Performance & Monitoring (Week 4)

### Task 23: Add Prometheus Metrics
**File**: `codeintel_rev/app/scope_registry.py`, `codeintel_rev/mcp_server/adapters/semantic.py`

**Description**: Instrument scope operations with metrics.

**Subtasks**:
1. Add gauge for active sessions:
   ```python
   METRICS.gauge("codeintel_active_sessions", "Number of active sessions")
   ```
   - Update gauge in `set_scope` (increment) and `prune_expired` (decrement).
2. Add counter for scope operations:
   ```python
   METRICS.counter("codeintel_scope_operations_total", "Total scope operations", ["operation"])
   ```
   - Increment for `set_scope`, `get_scope`, `clear_scope`.
3. Add histogram for scope filter overhead:
   ```python
   METRICS.histogram("codeintel_scope_filter_duration_seconds", "Time to apply scope filters")
   ```
   - Measure in `query_by_filters` and `apply_language_filter`.
4. Add logging for metrics initialization.

**Acceptance**:
- Metrics are exposed on `/metrics` endpoint.
- Grafana dashboard can visualize active sessions and operation counts.

**Time Estimate**: 2 hours

---

### Task 24: Benchmark Scope Filtering Performance
**File**: `tests/codeintel_rev/benchmarks/test_scope_performance.py` (NEW)

**Description**: Measure scope filtering overhead.

**Subtasks**:
1. Set up benchmark fixtures:
   - Create DuckDB catalog with 100K chunks.
   - Populate chunks with diverse URIs (various languages, directories).
2. Benchmark `query_by_ids` (baseline):
   - Query 1000 chunks by ID.
   - Measure time (avg over 100 runs).
3. Benchmark `query_by_filters` with language filter:
   - Query 1000 chunks with `languages=["python"]`.
   - Measure time, compare to baseline.
4. Benchmark `query_by_filters` with path globs:
   - Query 1000 chunks with `include_globs=["src/**"]`.
   - Measure time, compare to baseline.
5. Benchmark combined filters:
   - Query with language + path globs.
   - Measure time, compare to baseline.
6. Assert overhead <5ms (accept if within threshold).
7. Use `pytest-benchmark` plugin for reporting.

**Acceptance**:
- Benchmark results show scope filtering overhead <5ms.
- Results documented in benchmark report.

**Time Estimate**: 4 hours

---

### Task 25: Add Health Check for Registry
**File**: `codeintel_rev/app/readiness.py`

**Description**: Monitor scope registry health.

**Subtasks**:
1. Add `scope_registry` check to `ReadinessProbe._run_checks`:
   ```python
   results["scope_registry"] = self._check_scope_registry()
   ```
2. Implement `_check_scope_registry`:
   - Get active session count from registry.
   - If count > 10,000, return `CheckResult(healthy=False, detail="Too many active sessions")`.
   - Otherwise, return `CheckResult(healthy=True)`.
3. Add logging for registry check.

**Acceptance**:
- Health check fails if session count exceeds threshold.
- `/readyz` endpoint reflects registry health.

**Time Estimate**: 1 hour

---

## Summary

**Total Estimated Time**: ~60 hours (3 weeks at 20 hours/week)

**Task Breakdown**:
- Infrastructure: 13 hours
- Utilities: 7 hours
- Adapter Integration: 8 hours
- DuckDB Extensions: 11.5 hours
- Integration Testing: 10 hours
- Documentation: 12 hours
- Performance & Monitoring: 7 hours

**Dependencies**:
- Phase 2a must complete before Phase 2c (adapters depend on registry).
- Phase 2d must complete before Task 15 (semantic search depends on `query_by_filters`).
- Phase 2e can run in parallel with Phase 2f (testing vs documentation).

**Rollout Plan**:
1. Week 1: Complete infrastructure and utilities (Tasks 1-8).
2. Week 2: Integrate adapters (Tasks 9-12).
3. Week 3: Implement DuckDB filtering and integration tests (Tasks 13-18).
4. Week 4: Documentation, performance testing, and rollout (Tasks 19-25).

