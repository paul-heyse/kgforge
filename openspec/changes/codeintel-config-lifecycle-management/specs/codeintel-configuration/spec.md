## ADDED Requirements

### Requirement: Centralized Configuration Lifecycle

The CodeIntel MCP application SHALL manage configuration through a centralized lifecycle that loads settings exactly once during application startup and shares them across all request handlers via explicit dependency injection.

**Rationale**: Eliminates redundant environment variable parsing (currently 4+ calls per request), ensures configuration consistency, and provides a single source of truth for application-wide settings.

#### Scenario: Configuration loaded once at startup

- **GIVEN** the FastAPI application is starting
- **WHEN** the `lifespan()` function executes
- **THEN** `ApplicationContext.create()` is called exactly once
- **AND** settings are loaded from environment variables exactly once
- **AND** the resulting `ApplicationContext` is stored in `app.state.context`
- **AND** all subsequent requests use the same `ApplicationContext` instance

#### Scenario: Configuration error prevents startup

- **GIVEN** the `REPO_ROOT` environment variable points to a non-existent directory
- **WHEN** the FastAPI application attempts to start
- **THEN** `resolve_application_paths()` raises `ConfigurationError`
- **AND** the application fails to start
- **AND** a structured error message is logged with RFC 9457 Problem Details
- **AND** the error message includes the invalid path and remediation guidance

#### Scenario: Configuration immutability enforced

- **GIVEN** an `ApplicationContext` has been created
- **WHEN** code attempts to modify `context.settings.paths.repo_root`
- **THEN** a `FrozenInstanceError` is raised
- **AND** the original configuration remains unchanged

### Requirement: Explicit Dependency Injection

All adapter functions SHALL accept `ApplicationContext` as an explicit parameter and SHALL NOT call `load_settings()` or access global configuration state.

**Rationale**: Makes dependencies explicit (easy to trace, test, mock), eliminates hidden global state, follows AGENTS.MD dependency injection principle.

#### Scenario: Adapter receives context parameter

- **GIVEN** a client requests `/mcp/tools/list_paths`
- **WHEN** the MCP server invokes the `list_paths` tool wrapper
- **THEN** the wrapper extracts `ApplicationContext` from `request.app.state.context`
- **AND** the wrapper calls `files_adapter.list_paths(context, ...)`
- **AND** the adapter function uses `context.paths.repo_root` for path resolution
- **AND** the adapter function does NOT call `load_settings()`

#### Scenario: Adapter function is testable with mocked context

- **GIVEN** a unit test for `files_adapter.list_paths()`
- **WHEN** the test creates a mock `ApplicationContext` with test paths
- **AND** the test calls `list_paths(mock_context, path="src")`
- **THEN** the function executes using the mocked context
- **AND** no environment variables are read
- **AND** no global state is accessed

#### Scenario: All adapters use consistent pattern

- **GIVEN** the four MCP adapters (files, text_search, history, semantic)
- **WHEN** each adapter function is inspected
- **THEN** every adapter function has `context: ApplicationContext` as first parameter
- **AND** every adapter function uses `context.paths.*` for path access
- **AND** every adapter function uses `context.settings.*` for configuration access
- **AND** no adapter function calls `load_settings()` or `get_service_context()`

### Requirement: Resolved Paths Dataclass

The application SHALL provide a `ResolvedPaths` dataclass containing canonicalized absolute paths for all filesystem resources, eliminating path resolution duplication across adapters.

**Rationale**: Centralizes path resolution logic (currently duplicated in 4+ places), validates paths at startup, ensures consistency, reduces bugs from different resolution strategies.

#### Scenario: Paths resolved at startup

- **GIVEN** settings contain relative paths like `data/faiss/code.ivfpq.faiss`
- **WHEN** `resolve_application_paths(settings)` is called
- **THEN** all paths are converted to absolute paths
- **AND** relative paths are resolved against `repo_root`
- **AND** the returned `ResolvedPaths` contains only absolute Path objects
- **AND** `ResolvedPaths` is immutable (frozen=True)

#### Scenario: Invalid path raises structured error

- **GIVEN** `REPO_ROOT` environment variable is `/nonexistent/path`
- **WHEN** `resolve_application_paths(settings)` is called
- **THEN** a `ConfigurationError` is raised
- **AND** the error message includes the invalid path
- **AND** the error message includes the environment variable name
- **AND** the error conforms to RFC 9457 Problem Details format

### Requirement: Comprehensive Readiness Checks

The application SHALL provide a `ReadinessProbe` class that validates all critical resources (FAISS index, DuckDB catalog, vLLM service, filesystem) and exposes structured health check results via the `/readyz` endpoint.

**Rationale**: Enables fail-fast behavior at startup, provides Kubernetes readiness probe integration, validates that resources are accessible before serving traffic.

#### Scenario: All resources healthy

- **GIVEN** all required resources exist (FAISS index, DuckDB catalog, repo root)
- **AND** the vLLM service is reachable at configured URL
- **WHEN** `ReadinessProbe.refresh()` is called
- **THEN** all `CheckResult` instances have `healthy=True`
- **AND** the `/readyz` endpoint returns HTTP 200
- **AND** the response payload shows `ready: true`

#### Scenario: Missing FAISS index detected

- **GIVEN** the FAISS index file does not exist at `paths.faiss_index`
- **WHEN** `ReadinessProbe.refresh()` is called
- **THEN** the `faiss_index` check has `healthy=False`
- **AND** the check detail explains "FAISS index not found at {path}"
- **AND** the `/readyz` endpoint returns HTTP 200 (always returns 200, payload indicates status)
- **AND** the response payload shows `ready: false`

#### Scenario: vLLM service unreachable

- **GIVEN** the vLLM service URL is configured but the service is not running
- **WHEN** `ReadinessProbe._check_vllm_connection()` is called
- **THEN** an HTTP request is made to `{vllm_url}/health` with 2-second timeout
- **AND** the request fails with `httpx.HTTPError`
- **AND** a `CheckResult` with `healthy=False` is returned
- **AND** the detail field contains "vLLM service unreachable: {error}"

### Requirement: Optional FAISS Pre-loading

The application SHALL support optional FAISS index pre-loading during startup controlled by the `FAISS_PRELOAD` environment variable to eliminate first-request latency in production while maintaining fast startup for development.

**Rationale**: Production deployments benefit from consistent response times (no 2-10 second first-request delay); development benefits from fast iteration (no waiting for FAISS loading on every restart).

#### Scenario: Lazy loading (default behavior)

- **GIVEN** `FAISS_PRELOAD` environment variable is unset or `0`
- **WHEN** the application starts
- **THEN** the FAISS index is NOT loaded during startup
- **AND** startup completes quickly (< 1 second)
- **AND** the first semantic search request loads the FAISS index
- **AND** the FAISS index loading time (2-10 seconds) is added to first request latency

#### Scenario: Eager loading (production behavior)

- **GIVEN** `FAISS_PRELOAD=1` environment variable is set
- **WHEN** the application starts
- **THEN** `faiss_manager.load_cpu_index()` is called during startup
- **AND** `faiss_manager.clone_to_gpu()` is attempted during startup
- **AND** startup takes 2-10 seconds longer (FAISS loading time)
- **AND** the first semantic search request has consistent latency (no loading delay)
- **AND** structured logging shows "Pre-loading FAISS index during startup"

#### Scenario: GPU initialization failure is graceful

- **GIVEN** `FAISS_PRELOAD=1` and CUDA is not available
- **WHEN** the application starts and attempts GPU clone
- **THEN** `faiss_manager.clone_to_gpu()` returns False
- **AND** the application continues startup (does not fail)
- **AND** structured logging shows "FAISS GPU acceleration unavailable: {reason}"
- **AND** semantic search falls back to CPU index

### Requirement: Fail-Fast Startup Behavior

The application SHALL fail to start with a clear error message if any critical resource is missing or invalid, preventing deployment of broken configuration.

**Rationale**: Prevents "works locally, breaks in production" scenarios, provides immediate feedback on misconfiguration, aligns with 12-factor app principle of failing fast.

#### Scenario: Missing FAISS index prevents startup (when pre-loading enabled)

- **GIVEN** `FAISS_PRELOAD=1` environment variable is set
- **AND** the FAISS index file does not exist
- **WHEN** the application attempts to start
- **THEN** `faiss_manager.load_cpu_index()` raises `FileNotFoundError`
- **AND** the `lifespan()` function catches the exception
- **AND** structured logging shows "FAISS index pre-load failed: {error}"
- **AND** the application continues startup but logs degraded mode
- **AND** readiness probe shows FAISS check as unhealthy

#### Scenario: Invalid repo root prevents startup

- **GIVEN** `REPO_ROOT=/nonexistent/path` environment variable
- **WHEN** the application attempts to start
- **THEN** `resolve_application_paths()` raises `ConfigurationError`
- **AND** the `lifespan()` function catches the exception
- **AND** structured logging shows "Application startup failed due to configuration error"
- **AND** the exception is re-raised
- **AND** FastAPI fails to start
- **AND** the process exits with non-zero status code

#### Scenario: Configuration error provides remediation guidance

- **GIVEN** `REPO_ROOT` environment variable is missing or invalid
- **WHEN** `resolve_application_paths()` raises `ConfigurationError`
- **THEN** the error message includes "Repository root does not exist: {path}"
- **AND** the error context includes `{"repo_root": "{path}", "source": "REPO_ROOT env var"}`
- **AND** the error conforms to RFC 9457 Problem Details format
- **AND** the error message suggests setting valid `REPO_ROOT` environment variable

### Requirement: Backward Compatibility for External API

The MCP tool interfaces exposed to clients SHALL remain unchanged, with refactoring limited to internal implementation details.

**Rationale**: External clients depend on MCP tool signatures; breaking changes require client updates and coordination. Internal refactoring should not affect client code.

#### Scenario: MCP tool signatures unchanged for clients

- **GIVEN** a client calls `/mcp/tools/list_paths` with parameters `{"path": "src", "max_results": 100}`
- **WHEN** the tool is invoked
- **THEN** the tool signature from client perspective is identical to pre-refactor
- **AND** the tool returns the same response format
- **AND** the tool behavior is functionally identical

#### Scenario: Internal adapter signatures change (not breaking)

- **GIVEN** the `files_adapter.list_paths()` function signature now includes `context` parameter
- **WHEN** the MCP tool wrapper calls the adapter
- **THEN** the wrapper handles context extraction and injection
- **AND** internal callers (like other adapters) must be updated
- **AND** external MCP clients are not affected
- **AND** this is documented as an internal-only breaking change

### Requirement: Zero Global State

The application SHALL eliminate all global singletons and `@lru_cache` decorators for configuration access, using FastAPI application state instead.

**Rationale**: Global state violates AGENTS.MD principles, makes testing difficult, prevents parallel test execution, hides dependencies.

#### Scenario: No global ServiceContext singleton

- **GIVEN** the codebase after refactoring
- **WHEN** searching for `@lru_cache` decorators related to configuration
- **THEN** no `get_service_context()` function exists
- **AND** no `ServiceContext` class with `@lru_cache` exists
- **AND** `codeintel_rev/mcp_server/service_context.py` has been deleted

#### Scenario: Configuration stored in FastAPI state

- **GIVEN** the application has started successfully
- **WHEN** a request handler accesses `request.app.state.context`
- **THEN** an `ApplicationContext` instance is returned
- **AND** the instance is the same for all requests (singleton per application instance)
- **AND** the instance is not stored in global variables or module-level cache

#### Scenario: Tests can mock context without cache clearing

- **GIVEN** a unit test for an adapter function
- **WHEN** the test creates a mock `ApplicationContext`
- **AND** the test calls the adapter function with the mock context
- **THEN** the test does not need to clear any caches
- **AND** the test does not need to monkeypatch global variables
- **AND** multiple tests can run in parallel without interference

### Requirement: Structured Logging for Configuration Lifecycle

The application SHALL emit structured log messages for all configuration lifecycle events (load, validate, initialize, shutdown) with appropriate context fields.

**Rationale**: Enables observability of configuration loading, debugging of startup issues, monitoring of degraded modes (e.g., GPU unavailable).

#### Scenario: Configuration loading logged

- **GIVEN** the application is starting
- **WHEN** `ApplicationContext.create()` is called
- **THEN** a structured log message is emitted: "Loading application configuration from environment"
- **AND** when successful, a log message includes: "Application context created"
- **AND** the log extra fields include `{"repo_root": "{path}", "faiss_index": "{path}", "vllm_url": "{url}"}`

#### Scenario: FAISS pre-loading logged

- **GIVEN** `FAISS_PRELOAD=1` environment variable is set
- **WHEN** FAISS index is loaded during startup
- **THEN** a log message is emitted: "Pre-loading FAISS index during startup"
- **AND** on success, a log message includes: "FAISS CPU index loaded successfully"
- **AND** if GPU enabled, a log message includes: "FAISS GPU acceleration enabled"
- **AND** if GPU unavailable, a log message includes: "FAISS GPU acceleration unavailable: {reason}"

#### Scenario: Configuration error logged with context

- **GIVEN** the repository root does not exist
- **WHEN** `resolve_application_paths()` raises `ConfigurationError`
- **THEN** a structured log message is emitted at ERROR level
- **AND** the message is: "Application startup failed due to configuration error"
- **AND** the log extra fields include `{"error_code": "CONFIG_ERROR", "context": {"repo_root": "{path}"}}`
- **AND** full exception traceback is included via `exc_info=True`

