# CodeIntel MCP Implementation Plan

## Overview

This document outlines the structured, holistic implementation plan for completing the Tree-sitter MCP integration, fully aligned with AGENTS.md standards and ensuring zero pyright, pyrefly, ruff, or pytest errors without suppression.

## Current State Assessment

### ✅ Already Implemented
- **Language Runtime**: `codeintel/indexer/tscore.py` with manifest-driven language loading
- **MCP Server Skeleton**: `codeintel/mcp_server/server.py` with basic JSON-RPC loop
- **Tool Utilities**: `codeintel/mcp_server/tools.py` with query/symbols/calls/errors helpers
- **CLI Indexer**: `codeintel/indexer/cli.py` with query/symbols commands
- **Language Manifest**: `codeintel/build/languages.json` and `build_languages.py`
- **Python Query**: `codeintel/queries/python.scm` for function definitions and calls

### ❌ Missing Components
- **CLI Façade Integration**: Top-level `codeintel/cli.py` with MCP serve command
- **Security & Limits**: Path sandbox hardening, size caps, rate limiting, timeouts
- **Additional Tools**: `code.listFiles`, `code.getFile`, `code.getOutline`, `code.getAST`
- **Persistent Index**: Optional SQLite store for symbols/references (Step 4)
- **Configuration**: Centralized limits and feature flags (Step 5)
- **Tests**: Comprehensive test suite (Step 6)
- **Documentation**: Agent-ready docs with auto-generated tool reference (Step 7)
- **CI Integration**: Workflow and pre-commit hooks (Step 8)

## Implementation Strategy

### Phase 1: Core Infrastructure (Steps 3-5)

#### Step 3: CLI Façade Integration
**Goal**: Integrate MCP server into standardized CLI façade

**Tasks**:
1. Create `codeintel/cli.py`:
   - Import `cli_operation` from `tools._shared.cli_integration`
   - Import `CliContext`, `EnvelopeBuilder` from `tools._shared.cli_runtime`
   - Create Typer app with `mcp` subcommand group
   - Implement `mcp serve` command using `cli_operation` decorator
   - Export `KGF_REPO_ROOT` environment variable for tools
   - Use `anyio.run(mcp_amain)` to start server
   - Set result summary via `env.set_result()`

2. Update `Makefile`:
   - Add `codeintel-serve` target calling `python -m codeintel.cli mcp serve --repo .`
   - Add `codeintel-index` target (if index subcommand added)

3. Ensure envelope placement:
   - CLI envelopes written to canonical location via façade
   - Command path `["codeintel", "mcp", "serve"]` used for output directory

**Design Decisions**:
- Use existing CLI façade pattern (matches `orchestration`, `download` CLIs)
- Repo root passed via environment variable (`KGF_REPO_ROOT`) for consistency
- Server runs indefinitely until stdin closes (stdio mode)

#### Step 4: Persistent Index (Optional)
**Goal**: Add SQLite-based symbol/reference index for fast cross-file search

**Tasks**:
1. Create `codeintel/index/` directory structure:
   - `schema.sql`: SQLite schema (files, symbols, refs tables)
   - `store.py`: `IndexStore` class with context manager, `ensure_schema()`, `index_incremental()`, `search_symbols()`, `find_references()`
   - `__init__.py`: Public API exports

2. Implement schema:
   - `files` table: path, lang, mtime_ns, size_bytes
   - `symbols` table: path, lang, kind, name, qualname, start_line, end_line, signature, docstring
   - `refs` table: path, lang, kind, src_qualname, dst_qualname, line
   - Indexes on `refs_src` and `refs_dst`

3. Implement indexing:
   - `detect_lang()`: Extension-based language detection
   - `stat_meta()`: File metadata extraction
   - `needs_reindex()`: Incremental check (mtime + size)
   - `replace_file()`: Delete old + insert new symbols/refs
   - `discover_files()`: Recursive file discovery with excludes
   - `index_incremental()`: Main indexing loop

4. Implement search:
   - `search_symbols()`: SQL query with LIKE pattern matching
   - `find_references()`: SQL query for reference lookups

**Design Decisions**:
- SQLite with WAL mode for concurrent reads
- Incremental indexing based on mtime/size (no content hashing needed)
- Qualname format: `{path}::{name}` (can be refined later with module detection)
- Excludes: `.git`, `.venv`, `_build`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `node_modules`

#### Step 5: Security & Resource Limits
**Goal**: Hardened server with sandboxing, caps, rate limiting, and timeouts

**Tasks**:
1. Create `codeintel/config.py`:
   - `ServerLimits` dataclass (frozen=True) with defaults from environment:
     - `max_ast_bytes`: 1 MiB default
     - `max_outline_items`: 2000 default
     - `list_limit_default`: 100 default
     - `list_limit_max`: 1000 default
     - `tool_timeout_s`: 10.0 seconds default
     - `rate_limit_qps`: 5.0 default
     - `rate_limit_burst`: 10 default
     - `enable_ts_query`: False default (opt-in advanced feature)
   - `LIMITS` singleton instance

2. Create `codeintel/mcp_server/ratelimit.py`:
   - `TokenBucket` dataclass with `rate`, `burst`, `tokens`, `last`
   - `acquire(n)` method using `time.monotonic()` for token refill
   - Thread-safe (single-threaded event loop, but defensive)

3. Harden `mcp_server/tools.py`:
   - **Path Sandbox**:
     - `REPO_ROOT` from `os.environ.get("KGF_REPO_ROOT", Path.cwd())`
     - `_resolve_path()`: Reject paths outside repo (use `resolve()` and string prefix check)
     - `_resolve_directory()`: Ensure directory exists
     - `repo_relative()`: Helper to return repo-relative paths only
   - **Size Caps**:
     - `MAX_AST_BYTES` from `LIMITS.max_ast_bytes`
     - `_bounded_limit()`: Enforce `LIMITS.list_limit_max`
     - AST walk budget: 200k nodes max
   - **New Tools**:
     - `list_files()`: Directory scan with excludes, glob filter, limit
     - `get_file()`: Chunked file read (offset, length)
     - `get_outline()`: Hierarchical outline using query captures
     - `get_ast()`: Bounded AST snapshot (JSON or S-expression format)
   - **Problem Details**:
     - Create `SandboxError` exception (inherits `ValueError`)
     - Wrap all errors with Problem Details helpers
     - Use `tools.build_problem_details()` for structured errors

4. Harden `mcp_server/server.py`:
   - **Rate Limiting**:
     - Initialize `TokenBucket` in `__init__`
     - Check `acquire()` before each `tools/call` handler
     - Return Problem Details error on rate limit (429 status)
   - **Timeouts**:
     - Wrap handler calls with `anyio.move_on_after(LIMITS.tool_timeout_s)`
     - Return Problem Details on timeout (504 status)
     - Handle cancellation with Problem Details (499 status)
   - **Feature Flags**:
     - Check `LIMITS.enable_ts_query` before allowing `ts.query` tool
     - Return Problem Details if disabled (403 status)
   - **Error Handling**:
     - Convert all exceptions to Problem Details
     - Use `KGF-CI-*` error codes (e.g., `KGF-CI-RATE`, `KGF-CI-TIMEOUT`, `KGF-CI-SANDBOX`)
   - **New Tool Handlers**:
     - `_tool_list_files()`: Pydantic request model + handler
     - `_tool_get_file()`: Pydantic request model + handler
     - `_tool_get_outline()`: Pydantic request model + handler
     - `_tool_get_ast()`: Pydantic request model + handler
   - **Tool Schemas**:
     - Add schemas to `_tool_schemas()` using Pydantic `model_json_schema()`

**Design Decisions**:
- All paths validated against `REPO_ROOT` before any file operations
- Size limits enforced before parsing (fail fast)
- Rate limiting per-request (simple token bucket, no per-client tracking)
- Timeouts use AnyIO cancellation scopes (clean cancellation)
- Problem Details follow RFC 9457 format with `urn:kgf:problem:codeintel:*` type URIs

### Phase 2: Quality Assurance (Steps 6-7)

#### Step 6: Comprehensive Testing
**Goal**: Full test coverage with unit, integration, and smoke tests

**Tasks**:
1. Create `tests/codeintel/conftest.py`:
   - `repo_fixture`: Synthetic repo with Python, TOML, Markdown files
   - `set_env`: Autouse fixture setting `KGF_REPO_ROOT`, limits, feature flags

2. Create `tests/codeintel/test_tools_sandbox.py`:
   - `test_resolve_path_inside()`: Valid path resolution
   - `test_resolve_path_outside_raises()`: Sandbox violation rejection
   - `test_resolve_path_symlink_escape()`: Symlink traversal prevention

3. Create `tests/codeintel/test_tools_ast_limits.py`:
   - `test_get_ast_respects_size_limit()`: Large file rejection
   - `test_get_ast_bounded_traversal()`: Node budget enforcement

4. Create `tests/codeintel/test_tools_outline.py`:
   - `test_outline_simple()`: Basic outline extraction
   - `test_outline_hierarchical()`: Nested class/function structure

5. Create `tests/codeintel/test_server_roundtrip.py`:
   - `test_tools_list()`: JSON-RPC `tools/list` round-trip
   - `test_get_outline_call()`: JSON-RPC `tools/call` with `code.getOutline`
   - Use subprocess for true stdio behavior

6. Create `tests/codeintel/test_rate_limit_timeout.py`:
   - `test_rate_limit()`: Token bucket enforcement
   - `test_timeout()`: Handler timeout behavior

7. Create `tests/codeintel/test_cli_entrypoints.py`:
   - `test_cli_serve_help()`: Typer help text generation
   - `test_cli_serve_starts()`: Server startup (with timeout)

**Design Decisions**:
- Use pytest fixtures for test isolation
- Subprocess tests for true stdio behavior (not mocked)
- Parametrized tests for edge cases
- Mark integration tests with `@pytest.mark.integration`

#### Step 7: Documentation
**Goal**: Agent-ready documentation with auto-generated tool reference

**Tasks**:
1. Create `docs/modules/codeintel/index.md`:
   - Overview of CodeIntel MCP server
   - Quick links to tools, limits, config
   - Usage examples

2. Create `docs/modules/codeintel/quickstart_mcp.md`:
   - How to run server (`python -m codeintel.cli mcp serve`)
   - ChatGPT MCP setup instructions
   - First calls to try (`tools/list`, `code.getOutline`, `ts.query`)

3. Create `docs/modules/codeintel/tools.md`:
   - Auto-generated from server schemas (see generator below)
   - Tool descriptions, parameters, examples

4. Create `docs/modules/codeintel/limits.md`:
   - All `CODEINTEL_*` environment variables
   - Default values and override instructions
   - Security implications

5. Create `docs/modules/codeintel/config.md`:
   - `KGF_REPO_ROOT` configuration
   - Excludes patterns
   - Feature flags (`CODEINTEL_ENABLE_TS_QUERY`)

6. Create `tools/mkdocs_suite/docs/_scripts/gen_codeintel_mcp_docs.py`:
   - Import `MCPServer` and call `_tool_schemas()`
   - Generate Markdown from schemas
   - Write to `docs/modules/codeintel/tools.md`

7. Update `mkdocs.yml`:
   - Add CodeIntel section to nav
   - Link all new doc pages

8. Add Makefile target:
   - `docs-codeintel`: Run generator script
   - CI drift check: `git diff --exit-code docs/modules/codeintel/tools.md`

**Design Decisions**:
- Documentation generated from code (single source of truth)
- Drift checks ensure docs stay in sync
- Examples are copy-ready and runnable

### Phase 3: CI & Validation (Steps 8-9)

#### Step 8: CI Integration
**Goal**: Automated quality gates and drift checks

**Tasks**:
1. Create `.github/workflows/codeintel.yml`:
   - Trigger on `codeintel/**`, `tools/_shared/**`, `docs/modules/codeintel/**`
   - Steps:
     - Checkout
     - Setup Python 3.13
     - Install dependencies (`pip install -e .[dev]`)
     - Build TS manifest (`python -m codeintel.build_languages`)
     - Lint (`ruff check codeintel`)
     - Typecheck (`pyright codeintel`, `pyrefly check`)
     - Tests (`pytest -q tests/codeintel`)
     - Generate docs (`make docs-codeintel`)
     - Drift check (`git diff --exit-code docs/modules/codeintel/tools.md`)

2. Update `.pre-commit-config.yaml`:
   - Add hook: `forbid-direct-tree-sitter-imports` (grep check)
   - Add hook: `forbid-unsafe-open` (grep check for raw `open()` in `mcp_server`)

**Design Decisions**:
- CI runs same checks as local development
- Pre-commit hooks prevent common mistakes
- Drift checks catch documentation inconsistencies

#### Step 9: Quality Gate Validation
**Goal**: Ensure zero errors across all quality gates

**Tasks**:
1. Run full quality gate suite:
   ```bash
   uv run ruff format && uv run ruff check --fix
   uv run pyright --warnings --pythonversion=3.13
   uv run pyrefly check
   uv run pytest -q tests/codeintel
   make artifacts && git diff --exit-code
   python tools/check_new_suppressions.py codeintel
   python tools/check_imports.py
   uv run pip-audit
   ```

2. Fix any errors:
   - No suppressions allowed (per AGENTS.md)
   - Structural solutions only
   - Type annotations required for all public APIs
   - NumPy-style docstrings for all public symbols

3. Validate Problem Details:
   - All errors return RFC 9457 Problem Details
   - Error codes follow `KGF-CI-*` pattern
   - Schema validation passes

**Design Decisions**:
- Zero-error mandate enforced
- All fixes must be structural (no type ignores, no suppressions)
- Documentation must be complete and accurate

## Code Quality Standards

### Type Safety
- **Postponed Annotations**: All modules start with `from __future__ import annotations`
- **Full Type Coverage**: All public APIs fully typed (no `Any` in signatures)
- **PEP 695 Generics**: Use modern type parameter syntax where applicable
- **Protocol/TypedDict**: Prefer over `Any` for structural typing

### Documentation
- **NumPy Style**: All public functions/classes/modules have NumPy docstrings
- **Required Sections**: Summary, Parameters, Returns, Raises, Examples, Notes
- **Runnable Examples**: Examples in docstrings must execute (doctest/xdoctest)

### Error Handling
- **Problem Details**: All errors return RFC 9457 Problem Details
- **Exception Taxonomy**: Use `KgFoundryError` hierarchy where applicable
- **Error Codes**: Follow `KGF-CI-*` pattern for codeintel errors
- **Cause Chains**: Always `raise ... from e` to preserve exception chains

### Security
- **Path Sandbox**: All paths validated against `REPO_ROOT`
- **Size Limits**: Enforced before parsing (fail fast)
- **Rate Limiting**: Token bucket per-request
- **Timeouts**: All async operations have timeouts

### Testing
- **Coverage**: Unit tests for all tools, integration tests for server
- **Parametrization**: Edge cases covered with `@pytest.mark.parametrize`
- **Fixtures**: Isolated test fixtures for repo and environment
- **Markers**: Use `@pytest.mark.integration` for integration tests

## File Structure

```
codeintel/
├── __init__.py
├── cli.py                    # NEW: Top-level CLI with MCP serve
├── config.py                 # NEW: ServerLimits configuration
├── build_languages.py
├── build/
│   └── languages.json
├── index/                    # NEW: Optional persistent index
│   ├── __init__.py
│   ├── schema.sql
│   └── store.py
├── indexer/
│   ├── __init__.py
│   ├── cli.py
│   ├── cli_context.py
│   └── tscore.py
├── mcp_server/
│   ├── __init__.py
│   ├── server.py            # MODIFY: Add rate limit, timeout, new tools
│   ├── tools.py             # MODIFY: Harden sandbox, add new tools
│   ├── ratelimit.py         # NEW: Token bucket implementation
│   └── http_bridge.py
└── queries/
    └── python.scm

tests/
└── codeintel/               # NEW: Comprehensive test suite
    ├── conftest.py
    ├── test_tools_sandbox.py
    ├── test_tools_ast_limits.py
    ├── test_tools_outline.py
    ├── test_server_roundtrip.py
    ├── test_rate_limit_timeout.py
    └── test_cli_entrypoints.py

docs/modules/codeintel/      # NEW: Documentation
├── index.md
├── quickstart_mcp.md
├── tools.md                 # AUTO-GENERATED
├── limits.md
└── config.md

tools/mkdocs_suite/docs/_scripts/
└── gen_codeintel_mcp_docs.py  # NEW: Doc generator

.github/workflows/
└── codeintel.yml            # NEW: CI workflow
```

## Acceptance Criteria

### Functional
- [ ] `python -m codeintel.cli mcp serve --repo .` starts server successfully
- [ ] `tools/list` returns all tool schemas
- [ ] `tools/call code.listFiles` returns bounded file list
- [ ] `tools/call code.getOutline` returns hierarchical outline
- [ ] `tools/call code.getAST` returns bounded AST
- [ ] Path sandbox rejects `../` traversal
- [ ] Rate limiting enforces QPS limits
- [ ] Timeouts cancel long-running operations
- [ ] Problem Details returned for all errors

### Quality Gates
- [ ] `ruff format && ruff check --fix` passes with zero errors
- [ ] `pyright --warnings --pythonversion=3.13` passes with zero errors
- [ ] `pyrefly check` passes with zero errors
- [ ] `pytest -q tests/codeintel` passes with 100% coverage
- [ ] `make artifacts && git diff --exit-code` passes (no drift)
- [ ] `python tools/check_new_suppressions.py codeintel` passes (no suppressions)
- [ ] `python tools/check_imports.py` passes (architectural boundaries)
- [ ] `pip-audit` passes (no vulnerabilities)

### Documentation
- [ ] All public APIs have NumPy docstrings
- [ ] `docs/modules/codeintel/tools.md` auto-generated and up-to-date
- [ ] Quickstart guide includes ChatGPT MCP setup
- [ ] All environment variables documented

### CI/CD
- [ ] `.github/workflows/codeintel.yml` runs on relevant file changes
- [ ] Pre-commit hooks prevent unsafe patterns
- [ ] Documentation drift checks catch inconsistencies

## Implementation Order

1. **Step 3**: CLI façade integration (enables manual testing)
2. **Step 5**: Security & limits (critical for production readiness)
3. **Step 4**: Persistent index (optional, can be deferred)
4. **Step 6**: Tests (validates implementation)
5. **Step 7**: Documentation (completes user-facing features)
6. **Step 8**: CI integration (automates quality gates)
7. **Step 9**: Final validation (ensures zero errors)

## Notes

- **No Suppressions**: All errors must be fixed structurally, never suppressed
- **Type Safety First**: Prefer type narrowing with `cast()` + `isinstance` over `type: ignore`
- **Problem Details**: All errors follow RFC 9457 format with `urn:kgf:problem:codeintel:*` types
- **Documentation**: Examples must be runnable and copy-ready
- **Testing**: Parametrize edge cases, use fixtures for isolation

