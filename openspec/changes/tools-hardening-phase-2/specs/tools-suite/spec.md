## ADDED Requirements
### Requirement: Secure Tooling Execution Discipline
All tooling subprocess calls SHALL use vetted helpers that enforce allow-listed binaries, sanitized environments, structured logging, and Problem Details responses.

#### Scenario: Subprocess execution is sanitized
- **GIVEN** any module under `tools/` needs to invoke an external executable
- **WHEN** it performs the invocation
- **THEN** it calls `tools._shared.proc.run_tool` (or a documented wrapper) with an absolute path, timeout, redacted environment, and raises a typed `ToolExecutionError` with Problem Details on failure

#### Scenario: Ruff security and complexity checks pass
- **GIVEN** Ruff security rules (`S603`, `S607`) and complexity guards (`C901`, `PLR0912`) run on `tools/`
- **WHEN** the suite executes
- **THEN** no violations occur because handlers are refactored for clarity and subprocess usage is centralized

### Requirement: External Tooling Integration Parity
Import-linter orchestration and Agent Catalog builders SHALL match the latest public APIs, using typed request/response models and structured error envelopes.

#### Scenario: Import-linter wrapper reflects Report API
- **GIVEN** `tools/check_imports.py`
- **WHEN** import-linter reports contracts
- **THEN** the script uses `Report.status`, iterates typed violations, emits structured logs, and exits with Problem Details JSON when contracts fail

#### Scenario: Agent Catalog search uses typed requests
- **GIVEN** `tools/docs/build_agent_catalog.py` (or derivative tooling)
- **WHEN** it issues a catalog search
- **THEN** it constructs a typed `CatalogSearchRequest`, passes it to `search_catalog(request=...)`, handles `CatalogSearchError`, and surfaces Problem Details plus regression fixtures

### Requirement: Typed Tooling Payload Models
Documentation and navmap tooling SHALL operate on explicit dataclasses/TypedDicts validated against JSON Schemas, eliminating `Any` leakage.

#### Scenario: Docstring cache round-trips through schema
- **GIVEN** docstring builder cache persistence
- **WHEN** cached payloads are serialized
- **THEN** they instantiate typed models, validate against `schema/tools/docstring_cache.json`, and tests assert parity for happy/edge/error cases

#### Scenario: Docs analytics emit typed envelopes
- **GIVEN** `tools/docs/build_agent_analytics.py`
- **WHEN** it produces analytics JSON
- **THEN** the result matches a typed model, validates against `schema/tools/doc_analytics.json`, and returns Problem Details on validation failure

### Requirement: LibCST Codemod Typing Guarantees
Codemod utilities SHALL rely on typed LibCST interfaces backed by local stubs so mypy/pyrefly report no missing attributes or `Any` leakage.

#### Scenario: LibCST stubs enable typing
- **GIVEN** codemod modules under `tools/codemods/`
- **WHEN** mypy and pyrefly run
- **THEN** LibCST node references resolve via local stubs (or bundled `py.typed`), yielding zero `attr-defined`/`name-defined` violations

#### Scenario: Codemod tests exercise typed transformations
- **GIVEN** pytest codemod suites
- **WHEN** they execute
- **THEN** they construct typed LibCST trees, apply transformers, and assert output without disabling type checking via `Any`

### Requirement: Tooling Observability Signals
All tooling operations SHALL emit structured logs with correlation IDs, Prometheus metrics, and OpenTelemetry spans.

#### Scenario: Failure path is observable
- **GIVEN** a subprocess timeout occurs inside `run_tool`
- **WHEN** `ToolExecutionError` is raised
- **THEN** an error log with `correlation_id` is recorded, `tool_failures_total{tool=...,reason="timeout"}` increments, and an OTEL span captures error status with attributes (`tool`, `exit_code`, `duration_ms`)

### Requirement: Typed 12-Factor Settings
Tools SHALL read configuration via typed environment settings and fail fast with Problem Details when required variables are missing.

#### Scenario: Missing required env fails fast
- **GIVEN** `AGENT_CATALOG_URL` is required
- **WHEN** a tool starts without that environment variable
- **THEN** it raises `SettingsError` from the typed settings loader and emits a Problem Details payload describing the missing variable

### Requirement: Tooling Security Hygiene
Tooling SHALL avoid unsafe evaluation, use safe parsers, validate untrusted input, and maintain a clean dependency bill of materials.

#### Scenario: YAML parsing is safe
- **GIVEN** a tool must parse YAML input
- **WHEN** the file is loaded
- **THEN** it uses `yaml.safe_load`, rejects unknown tags, and surfaces a Problem Details error instead of executing arbitrary payloads

### Requirement: Packaged Tools
The tools suite SHALL build distributable artifacts with optional extras and run cleanly after installation.

#### Scenario: Clean install
- **WHEN** `pip install .[tools]` executes inside a fresh virtual environment
- **THEN** the installation succeeds, entry points resolve, and CLIs run without import errors or missing metadata

### Requirement: Idempotent CLI Operations
Tooling CLIs SHALL be idempotent, document retry semantics, and avoid duplicate side effects on repeated runs.

#### Scenario: Double run converges
- **GIVEN** a CLI that mutates artifacts
- **WHEN** it is executed twice with identical inputs
- **THEN** the second run performs no additional side effects and reports convergence in logs/metrics

### Requirement: Performance Budgets
Hot tooling paths SHALL define local latency and memory budgets backed by automated checks.

#### Scenario: Budget respected
- **GIVEN** the agent catalog builder operates on a representative small dataset
- **WHEN** performance micro-bench tests execute
- **THEN** recorded p95 latency stays under 2 seconds and peak RSS remains below 400 MiB

### Requirement: Docs Are Runnable and Linked
Documentation artifacts SHALL include runnable examples and cross-links tying specs, schemas, and code anchors together.

#### Scenario: Examples execute
- **WHEN** documentation artifacts are regenerated via `make artifacts`
- **THEN** doctest/xdoctest runs on published examples succeed and generated Agent Portal links resolve to the referenced code anchors

### Requirement: Versioning & Deprecation
Public tooling interfaces SHALL follow SemVer language, emit single-use deprecation warnings, and state removal timelines.

#### Scenario: Deprecated flag warns
- **GIVEN** a deprecated CLI flag remains available for one release
- **WHEN** a user invokes the flag
- **THEN** exactly one structured warning logs the deprecation, specifies the removal version, and recommends the replacement flag

### Requirement: Time & Path Hygiene
Tooling SHALL standardize on `pathlib`, timezone-aware datetimes, and monotonic time sources for durations.

#### Scenario: Duration is monotonic
- **GIVEN** a tool measures execution time
- **WHEN** it records start/stop timestamps
- **THEN** it uses `time.monotonic()` for the delta and includes the elapsed milliseconds in structured logs with timezone-aware wall-clock context where emitted

### Requirement: Tool HTTP Surfaces Align with OpenAPI
Any tooling component that exposes or validates HTTP payloads SHALL define an OpenAPI 3.2 document and lint it in CI.

#### Scenario: OpenAPI spec lints clean
- **GIVEN** a tooling HTTP surface (e.g., Agent Portal preview endpoint)
- **WHEN** `spectral lint` runs against its OpenAPI 3.2 specification
- **THEN** the linter passes and the spec references the latest Problem Details schemas

