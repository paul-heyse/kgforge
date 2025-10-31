## ADDED Requirements

### Requirement: Pathlib Standardization Across Workflows
All filesystem interactions under `src/` SHALL use `pathlib.Path` helpers for directory management and file IO, avoiding `os.path.*`, bare `open()`, and `os.makedirs()`.

- Helper utilities SHALL live in `kgfoundry_common.fs` (or equivalent) and be reused across download, orchestration, embeddings, registry, and search modules.
- Directory creation SHALL call `Path.mkdir(parents=True, exist_ok=True)`; temporary paths SHALL use `TemporaryDirectory` wrappers.
- File reads/writes SHALL be performed through `Path.read_text`, `Path.iterdir`, or `Path.open()` with explicit encoding.
- A codemod under `tools/codemods/pathlib_fix.py` SHALL be executed before manual cleanup in each module cluster, and its log SHALL be attached to the PR.

#### Scenario: Index writer uses Path helpers
- **GIVEN** the FAISS, BM25, SPLADE, and fixture index writers
- **WHEN** they persist indices or metadata
- **THEN** each writer calls `Path.mkdir(parents=True, exist_ok=True)` before writing and uses `Path.open()` for serialization

#### Scenario: Artifact orchestrator avoids os.path
- **GIVEN** the documentation/download orchestration flows
- **WHEN** they compute output paths
- **THEN** the implementation uses `Path` arithmetic ( `/` operator ) and contains no calls to `os.path.join`, `os.path.dirname`, or `os.path.exists`


### Requirement: Exception Taxonomy and Problem Details
The platform SHALL expose a typed exception hierarchy in `kgfoundry_common.errors` and SHALL map exceptions to RFC 9457 Problem Details responses for HTTP surfaces.

- Public exceptions SHALL inherit from a common `KgFoundryError` base with structured fields (`code`, `http_status`, `log_level`).
- All exceptions raised inside try/except blocks SHALL be re-raised with `raise ... from exc` preserving causality.
- HTTP endpoints (e.g., `search_api/app.py`) SHALL return Problem Details JSON bodies with `type`, `title`, `status`, `detail`, and `instance` fields, plus `errors` when additional context exists.
- Problem Details payloads SHALL include a stable `type` URI and `code` sourced from `kgfoundry_common.errors.codes`.
- A canonical Problem Details example SHALL live under `schema/examples/problem_details/search-missing-index.json` and be referenced in documentation tests.
- A codemod under `tools/codemods/blind_except_fix.py` SHALL be executed before manual exception cleanup; any remaining TODOs MUST reference a tracking ticket.

#### Scenario: Blind excepts eliminated
- **GIVEN** existing `try/except Exception` blocks in search adapters, GPU utilities, and download jobs
- **WHEN** error handling is executed
- **THEN** the catch blocks log via module loggers, wrap the original exception with a typed subclass, and propagate (no swallow-and-pass)

#### Scenario: HTTP error surfaces Problem Details
- **GIVEN** a missing dense index in `search_api/app.py`
- **WHEN** the endpoint is invoked
- **THEN** the response status is 503 and the JSON body matches the documented Problem Details schema (validated in tests against the example file)


### Requirement: Secure Serialization and Persistence
The system SHALL avoid unsafe serialization/primitives (`pickle`, unparameterized SQL) and SHALL validate file inputs before use.

- Embedding and search components SHALL serialize lightweight metadata using JSON (or MessagePack) with explicit schemas instead of `pickle`.
- SQL interactions (DuckDB, SQLite) SHALL use parameterized statements or query builders that prevent injection.
- External inputs (fixture archives, downloaded assets) SHALL be validated using checksums or schema validation prior to ingestion.

#### Scenario: Safe index metadata persistence
- **GIVEN** BM25/SPLADE metadata writers
- **WHEN** metadata is written
- **THEN** the code serializes via a typed JSON encoder and the resulting file validates against a schema enforced in tests

#### Scenario: Parameterized SQL queries
- **GIVEN** registry modules using DuckDB/SQLite
- **WHEN** they execute queries derived from user input
- **THEN** statements are built with placeholders/bind variables and mypy enforces typed return objects


### Requirement: Typed JSON Contracts
Cross-boundary payloads SHALL be defined by JSON Schema 2020-12 (OpenAPI 3.2 for HTTP) and mirrored by fully typed Pydantic models.

- Schemas SHALL cover: agent session state, MCP envelopes, registry records, search requests/responses, and analytics emitted from observability flows.
- Each schema SHALL include examples and version metadata; breaking changes SHALL bump the schema version.
- Pydantic models SHALL specify `ConfigDict(extra="forbid")` and export round-trip validation tests.
- Every schema SHALL define a unique `$id` and `x-version`/`x-compatibility-notes`; all example files SHALL live under `schema/examples/**`.

#### Scenario: Schema validation for agent session state
- **GIVEN** `kgfoundry/agent_catalog/session.py`
- **WHEN** session payloads are serialized/deserialized
- **THEN** values validate against `schema/agent_catalog/session.v1.json` via unit tests and mypy observes no implicit `Any`

#### Scenario: Search API OpenAPI contract enforced
- **GIVEN** `search_api/app.py`
- **WHEN** OpenAPI generation runs
- **THEN** the produced spec aligns with `schema/search_api/openapi.json` and pytest verifies request/response examples


### Requirement: Vector Search Protocol Compliance
Vector search components SHALL implement typed protocols ensuring compatibility with FAISS modules and numpy typing.

- Define `FaissModuleProtocol` and related index/result protocols under `kgfoundry/search_api/types.py` (or equivalent) using PEP 544.
- `_SimpleFaissModule` and GPU adapters SHALL satisfy these protocols without `# type: ignore`.
- Numpy arrays SHALL be typed with `numpy.typing.NDArray` and concrete dtypes.
- Public import paths SHALL be consolidated under the `kgfoundry.*` namespace; duplicate top-level packages SHALL be removed or shimmed.

#### Scenario: Protocol satisfaction verified by mypy
- **GIVEN** mypy strict mode
- **WHEN** checking `_SimpleFaissModule`
- **THEN** it implements all required protocol members with matching signatures (no unused ignores)

#### Scenario: Typed vector results
- **GIVEN** FAISS adapter response builders
- **WHEN** they produce search results
- **THEN** return types are `Sequence[VectorSearchResult]` (typed dataclass/TypedDict) without `Any`


### Requirement: Parquet IO Type Safety
Parquet ingestion utilities SHALL expose typed facades to pyarrow/duckdb that avoid implicit `Any` propagation.

- Public functions SHALL declare concrete return types (`pa.Table`, `pd.DataFrame`) and parameter types using generics where applicable.
- Imports from pyarrow/duckdb SHALL be covered by stubs or local type annotations to prevent `no-any-unimported` errors.
- Tests SHALL assert that helper functions enforce schema validation and raise typed exceptions on mismatch.

#### Scenario: Typed table extraction
- **GIVEN** `kgfoundry_common.parquet_io.read_table`
- **WHEN** it returns data
- **THEN** the type is `pa.Table` with schema matching the declared JSON Schema, and mypy no longer reports `Any`

#### Scenario: DataFrame conversion retains typing
- **GIVEN** conversion helpers to pandas
- **WHEN** they execute
- **THEN** the return type is `pd.DataFrame` with `SchemaField` metadata captured in tests


### Requirement: Typed Configuration Settings
Application configuration SHALL be defined via `pydantic_settings.BaseSettings` (or equivalent) classes with strict typing and validation.

- Required environment variables SHALL be explicitly declared with default `None` and `Field(..., description=...)` to enforce presence.
- Loading configuration SHALL fail fast with a typed `ConfigurationError` when mandatory values are absent or malformed.
- Nested configuration (e.g., search weights, registry URIs) SHALL be represented as typed models to eliminate dynamic dict usage.

#### Scenario: Missing env fails fast
- **GIVEN** an unset required variable (e.g., `KGFOUNDRY_SEARCH_API_URL`)
- **WHEN** `RuntimeSettings()` is instantiated
- **THEN** a `ConfigurationError` is raised with Problem Details metadata logged

#### Scenario: Override via env works
- **GIVEN** default settings
- **WHEN** overrides are provided via environment or `.env` files
- **THEN** the typed model reflects the overrides and mypy sees no dynamic `dict`


### Requirement: Structured Observability Envelope
Each major workflow SHALL emit structured logs, Prometheus metrics, and optional OpenTelemetry traces with consistent correlation identifiers.

- Module-level loggers SHALL be defined with `NullHandler` to prevent duplicate handlers in libraries.
- Logs SHALL include `operation`, `status`, `duration_ms`, and `correlation_id` fields via structured logging helpers.
- Metrics SHALL expose counters/gauges (`kgfoundry_requests_total`, `kgfoundry_request_errors_total`) tagged by operation and status.
- Trace spans (when enabled) SHALL capture errors with status `ERROR` and attach exception metadata.
- Consumers SHALL obtain loggers exclusively via `kgfoundry_common.logging.get_logger(__name__)`, which injects the required structured context.

#### Scenario: Error path emits log + metric + trace
- **GIVEN** a failing search request
- **WHEN** the error propagates through the service layer
- **THEN** a structured error log is emitted, `kgfoundry_request_errors_total` increments with `status="error"`, and the trace span records the exception

#### Scenario: Libraries install NullHandler
- **GIVEN** importing `kgfoundry_common.logging`
- **WHEN** the module is imported by downstream applications
- **THEN** no duplicate handlers are attached and structured logging helpers are available


### Requirement: Public API Hygiene
Package `__init__` surfaces SHALL explicitly define sorted `__all__`, expose only typed symbols, and include Problem Details examples for error contracts.

- Each public module SHALL provide a NumPy-style docstring whose first sentence is a one-line summary (inherited from other change tracks as needed).
- `__all__` SHALL be alphabetized; any dynamic registry SHALL be replaced by explicit imports with typed objects.
- Problem Details sample JSON SHALL be referenced from module docstrings/tests for at least one error path.
- Documentation SHALL point to the canonical schema/example paths so Agent Portal links remain stable.

#### Scenario: Namespace module exports typed symbols
- **GIVEN** `_namespace_proxy.py` and top-level packages
- **WHEN** they expose public APIs
- **THEN** exports are enumerated via sorted `__all__`, rely on typed imports, and mypy reports no `Any`

#### Scenario: Problem Details example documented
- **GIVEN** `kgfoundry/search_client`
- **WHEN** users inspect the module docstring or reference docs
- **THEN** they see a copy-ready Problem Details example that matches the schema and is exercised by doctests


### Requirement: Type Checker and Lint Gate Enforcement
Tooling configuration SHALL enforce zero-tolerance for new Ruff and mypy errors within `src/**`.

- `mypy.ini` SHALL avoid non-per-module flags inside per-module sections; strict settings SHALL apply to agent catalog, search, registry, and observability packages.
- CI SHALL fail fast on any new `Any` introduction or ignored Ruff violations.
- Developer documentation SHALL describe the gating commands required before merge (mirroring AGENTS.md).
- Ruff configuration SHALL enable `flake8-type-checking` strict mode for heavy imports, and mypy SHALL load the `pydantic` and `sqlalchemy` plugins.

#### Scenario: Clean baseline enforcement
- **GIVEN** the updated configurations
- **WHEN** `uv run ruff check src` and `uv run mypy src` run on main
- **THEN** both commands exit 0 with no pending suppressions

#### Scenario: Regression caught in CI
- **GIVEN** a PR reintroduces a blind exception or `Any`
- **WHEN** CI executes
- **THEN** Ruff or mypy fails, blocking merge until the issue is resolved


### Requirement: Layered Import Contracts and Suppression Guard
The build system SHALL enforce architectural boundaries and prevent unmanaged suppressions.

- An `import-linter.cfg` contract SHALL forbid domain modules from importing adapters/I/O layers (and other agreed boundaries). All changes MUST satisfy the contract.
- A script (`tools/check_new_suppressions.py`) SHALL fail the build when new `# type: ignore` or `noqa` directives appear without an accompanying `TICKET:` tag.
- CI SHALL run both checks as part of acceptance gates; local runs SHALL be documented in the tasks checklist.

#### Scenario: Import contract enforced
- **WHEN** `import-linter --config importlinter.cfg` executes on the branch
- **THEN** it exits 0, proving no layering violations were introduced

#### Scenario: Suppressions require tracking
- **WHEN** `python tools/check_new_suppressions.py src` executes
- **THEN** it exits 0 unless every suppression includes `TICKET:`; otherwise the build fails with actionable output


### Requirement: Testing Strategy & Doctest Parity
All new or modified tests SHALL follow a table-driven approach with explicit happy, edge, and failure cases; doctest/xdoctest examples SHALL execute without modification and map back to scenarios in this spec.

- Tests touching a requirement SHALL reference the requirement name or scenario in a docstring/comment for traceability.
- Doctest examples SHALL import the same symbols exposed via public APIs and SHALL reference schema/Problem Details examples when applicable.
- Performance-sensitive code SHALL include benchmark or deterministic timing tests to protect budgets defined elsewhere in this spec.

#### Scenario: Parametrized regression coverage
- **GIVEN** a new helper or adapter introduced by this change
- **WHEN** pytest runs
- **THEN** there is at least one `@pytest.mark.parametrize`-decorated test covering success, edge, and failure conditions tied to the corresponding requirement

#### Scenario: Doctest execution
- **GIVEN** documentation/examples updated by this change
- **WHEN** `pytest --doctest-modules` executes
- **THEN** all doctests pass without modification and produce the expected Problem Details or schema-backed outputs


### Requirement: Concurrency & Context Propagation
Async or concurrent workflows SHALL preserve context (correlation IDs, timeouts) using `contextvars` and SHALL avoid blocking operations on the event loop.

- Async functions SHALL document await semantics, supported timeouts, and cancellation behavior in their docstrings.
- Blocking I/O SHALL execute in thread/process pools to prevent event-loop starvation.
- Correlation IDs and structured logging context SHALL propagate through async tasks and background workers.

#### Scenario: ContextVar propagation
- **GIVEN** an async search request with a seeded correlation ID
- **WHEN** the request awaits downstream adapters
- **THEN** logs, metrics, and Problem Details emitted from inside the adapter include the same correlation ID

#### Scenario: Timeout enforcement
- **GIVEN** a configured timeout for vector search
- **WHEN** the operation exceeds the budget
- **THEN** a typed timeout exception is raised (with `raise ... from exc`), a Problem Details response is emitted, and blocking threads are cancelled or released


### Requirement: Performance Budgets & Monitoring
The system SHALL define and enforce performance budgets for search/index workflows, using monotonic timing and benchmark tests to detect regressions.

- Budgets (e.g., hybrid search p95 latency < 300 ms, index serialization < configurable threshold) SHALL be documented in module docstrings and design notes.
- Benchmark or deterministic timing tests SHALL exist for critical paths; failures SHALL block merges until addressed or budgets updated with rationale.
- Timing measurements SHALL use `time.monotonic()` (or async equivalents) and structured logging for observability.

#### Scenario: Search latency budget enforced
- **GIVEN** the hybrid search service running against fixture data
- **WHEN** the benchmark test executes
- **THEN** the recorded duration stays within the documented budget; otherwise the test fails and emits a mitigation note

#### Scenario: Regression detection
- **GIVEN** a change that slows index writes
- **WHEN** benchmarks run in CI
- **THEN** they fail with a descriptive message identifying the regression and referencing the documented budget


### Requirement: Documentation & Discoverability
Public documentation and examples SHALL remain copy-ready, reference canonical schemas, and ensure Agent Portal links continue to function.

- Docstrings SHALL include cross-links or references to the relevant schema/Problem Details JSON when applicable.
- `make artifacts` SHALL produce updated docs without warnings; new examples SHALL appear in the Agent Portal with working deep links.
- PRs SHALL note documentation changes and include before/after references when user-facing behavior shifts.

#### Scenario: Schema-linked docstrings
- **GIVEN** a public API returning Problem Details
- **WHEN** a user reads its docstring
- **THEN** the docstring references the example JSON path and the example validates against the schema via doctest

#### Scenario: Agent Portal link integrity
- **GIVEN** new catalog or schema entries
- **WHEN** the Agent Portal is rebuilt
- **THEN** deep links open in the editor/GitHub as configured without 404s or stale anchors


### Requirement: Packaging & Distribution Integrity
The project SHALL retain reproducible packaging metadata and build artifacts after the refactor.

- `pyproject.toml` metadata SHALL remain valid (PEP 621) with accurate dependencies/extras; optional features SHALL use extras with environment markers (e.g., `gpu`).
- Building wheels (`python -m build`) in a clean environment SHALL succeed; installing the package in a fresh virtualenv SHALL succeed without missing dependencies.
- Any new dependencies SHALL be justified and documented, keeping footprint minimal.
- GPU-specific dependencies SHALL only be installed when the `gpu` extra is requested.

#### Scenario: Wheel build succeeds
- **WHEN** `python -m build` runs after the changes
- **THEN** both sdist and wheel build successfully and include updated metadata reflecting new modules/schemas

#### Scenario: Clean install smoke test
- **GIVEN** a clean virtual environment
- **WHEN** `pip install dist/kgfoundry-*.whl` executes
- **THEN** installation completes without errors and `python -m kgfoundry` (or relevant CLI) starts with typed settings validation


### Requirement: Security & Supply Chain Hardening
Security posture SHALL improve by removing unsafe constructs, scanning dependencies, and validating untrusted inputs.

- The code SHALL avoid `eval`, `exec`, unsafe YAML loaders, or untrusted `pickle` usage.
- External inputs SHALL be validated against schemas, checked for path traversal, and sanitized before processing.
- Dependency scans (`pip-audit` or equivalent) SHALL report no unresolved high-severity vulnerabilities; results SHALL be recorded.

#### Scenario: Input sanitization enforced
- **GIVEN** a user-provided path or payload
- **WHEN** the system processes it
- **THEN** validation catches traversal or schema violations and raises a typed error before reaching the filesystem or database

#### Scenario: Vulnerability scan clean
- **WHEN** `pip-audit` runs on the dependency graph
- **THEN** it exits 0 (or documented exceptions with mitigation) and the results are attached to the PR


### Requirement: Idempotency & Retry Semantics
Externally triggered operations (HTTP/CLI/queues) SHALL be idempotent or explicitly document retry semantics, ensuring repeated invocations converge.

- Idempotent endpoints SHALL handle duplicate requests without side effects; non-idempotent operations SHALL document retry/backoff guidance.
- Circuit-breaker or retry logic SHALL surface typed exceptions and Problem Details when exhausted.

#### Scenario: Repeatable index build
- **GIVEN** an index build command invoked twice with the same inputs
- **WHEN** the second invocation runs
- **THEN** it detects existing artifacts, no-ops or refreshes safely, and returns a Problem Details warning if manual intervention is required

#### Scenario: Retry semantics documented
- **GIVEN** a transient dependency failure
- **WHEN** retry logic exhausts attempts
- **THEN** the response includes Problem Details indicating retry guidance and the operation remains consistent afterward


### Requirement: File, Time, and Number Hygiene
The codebase SHALL use timezone-aware datetimes, monotonic clocks for durations, and appropriate numeric types for domain values.

- Timestamps SHALL use `datetime.now(tz=datetime.UTC)` (or equivalent) and be serialized with timezone offsets.
- Duration measurements SHALL rely on `time.monotonic()`; monetary or precise decimal values SHALL use `decimal.Decimal`.
- Ruff DTZ rules SHALL pass without suppressions; docstrings SHALL clarify units/timezone expectations.

#### Scenario: Timezone-aware timestamps
- **GIVEN** a new log entry or persisted timestamp
- **WHEN** inspected via tests
- **THEN** it includes timezone information and round-trips correctly through serialization/deserialization

#### Scenario: Monotonic durations
- **WHEN** duration metrics are emitted for operations
- **THEN** they derive from monotonic clocks, preventing negative or skewed durations even if system time changes


### Requirement: Phased Delivery & Early Wins
The team SHALL deliver “first wins” within seven days to de-risk adoption.

- Initial deliveries SHALL include: pathlib helpers + codemod (R1), exception taxonomy + Problem Details registry (R2), one schema/model pair with validation (R4), runtime settings fail-fast (R7), and CI automation (import-linter + suppression guard + PR summary).
- Each early-win change SHALL leave the tree at the clean baseline and provide documentation for downstream adoption.

#### Scenario: First week milestones met
- **WHEN** the seventh day ends after project start
- **THEN** the artifacts listed above are merged, tested, and referenced in follow-up tasks, enabling Phase 2 adoption to proceed smoothly


