## Why

The CodeIntel MCP implementation currently loads configuration from environment variables repeatedly in each adapter function (calling `load_settings()` multiple times per request). This violates AGENTS.md design principles in three critical ways:

1. **Redundant Environment Parsing**: Each adapter call re-parses environment variables instead of using a single shared configuration instance
2. **Inconsistent Context Access**: Mixed patterns between `load_settings()` in adapters vs `get_service_context()` in semantic search
3. **Deferred FAISS Loading**: Lazy initialization causes first-request latency and lacks proper startup health checks

This design review addresses these deficiencies with a structural, holistic solution that:
- Eliminates all redundant configuration loading through explicit dependency injection
- Centralizes configuration lifecycle management in FastAPI application startup
- Implements fail-fast behavior with comprehensive health checks
- Provides extensibility for future multi-repository support while maintaining single-repo simplicity

## What Changes

- **ADDED**: `codeintel_rev/app/config_context.py` — centralized application context with explicit configuration lifecycle
  - `ApplicationContext` dataclass holding settings, paths, and long-lived clients (vLLM, FAISS)
  - `ResolvedPaths` dataclass with canonicalized absolute paths (eliminates path resolution duplication)
  - `resolve_application_paths()` function for path validation with structured error handling
  - Configuration loaded once during FastAPI startup, injected into all request handlers

- **ADDED**: `codeintel_rev/app/readiness.py` — comprehensive readiness probe system
  - `ReadinessProbe` class managing health checks for all dependencies
  - Validates FAISS index, DuckDB catalog, vLLM service, and filesystem resources
  - Integrated into FastAPI lifespan with fail-fast behavior on startup

- **MODIFIED**: `codeintel_rev/app/main.py` — refactored application lifespan
  - Configuration initialization moved from lazy to explicit startup phase
  - `ApplicationContext` stored in `app.state` for request-level access
  - Optional FAISS pre-loading controlled by `FAISS_PRELOAD` environment variable
  - Structured error handling with ConfigurationError failing application startup

- **MODIFIED**: All adapter functions (4 adapters) — dependency injection pattern
  - `codeintel_rev/mcp_server/adapters/files.py` — accepts `ApplicationContext` parameter
  - `codeintel_rev/mcp_server/adapters/text_search.py` — accepts `ApplicationContext` parameter
  - `codeintel_rev/mcp_server/adapters/history.py` — accepts `ApplicationContext` parameter
  - `codeintel_rev/mcp_server/adapters/semantic.py` — refactored to use injected context
  - All adapters eliminate `load_settings()` calls, use `context.paths.repo_root` and `context.settings`

- **REMOVED**: `codeintel_rev/mcp_server/service_context.py` — functionality consolidated into `ApplicationContext`
  - `ServiceContext` class merged into `ApplicationContext`
  - `get_service_context()` singleton removed in favor of FastAPI state management
  - `reset_service_context()` functionality replaced by application lifecycle management

- **MODIFIED**: `codeintel_rev/mcp_server/server.py` — MCP tool wrappers with context injection
  - All `@mcp.tool()` functions extract `ApplicationContext` from FastAPI request state
  - New `_get_context(request: Request)` helper for consistent context retrieval
  - Context passed explicitly to adapter functions (no global state)

- **MODIFIED**: `codeintel_rev/config/settings.py` — added `faiss_preload` configuration
  - `IndexConfig` dataclass gains `faiss_preload: bool` field (default: False)
  - `load_settings()` reads `FAISS_PRELOAD` environment variable

- **ADDED**: Comprehensive test suite for configuration lifecycle
  - `tests/codeintel_rev/test_config_context.py` — unit tests for `ApplicationContext` and path resolution
  - `tests/codeintel_rev/test_app_lifespan.py` — integration tests for FastAPI startup/shutdown
  - `tests/codeintel_rev/adapters/test_files_adapter.py` — adapter tests with mocked context

- **ADDED**: `codeintel_rev/docs/CONFIGURATION.md` — configuration management documentation
  - Describes centralized configuration lifecycle
  - Documents FAISS pre-loading options (eager vs lazy)
  - Provides best practices for development, production, and Kubernetes deployments

## Impact

- **Specs**: New capability `codeintel-configuration` describing centralized configuration lifecycle management
- **Affected code**: 
  - Core: `codeintel_rev/app/` (new config_context.py, readiness.py, modified main.py)
  - Adapters: `codeintel_rev/mcp_server/adapters/` (all 4 adapters refactored)
  - Server: `codeintel_rev/mcp_server/server.py` (MCP tool wrappers)
  - Config: `codeintel_rev/config/settings.py` (added faiss_preload)
  - Tests: New test modules for configuration, lifespan, and adapters
- **Data contracts**: No schema changes; internal refactoring only
- **Breaking changes**: None for external API; internal adapter signatures change (breaking for internal callers only)
- **Rollout**: 
  - Phase 1: Create `ApplicationContext` and `ReadinessProbe` (backward compatible)
  - Phase 2: Refactor adapters sequentially (incremental, testable)
  - Phase 3: Update MCP server integration (final cutover)
  - Phase 4: Remove deprecated `ServiceContext` (cleanup)
  - Optional: Enable `FAISS_PRELOAD=1` in production for consistent response times

## Dependencies

- Requires existing FastAPI application structure in `codeintel_rev/app/main.py`
- Depends on current adapter interfaces (will be modified but maintain external compatibility)
- Builds on existing error taxonomy (`kgfoundry_common.errors.ConfigurationError`)
- Leverages FastAPI lifespan events (already present)
- No new external dependencies required

