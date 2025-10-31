## ADDED Requirements
### Requirement: Shared Tools Infrastructure Hardening
The shared tooling infrastructure SHALL provide structured logging, Problem Details generation, and secure subprocess orchestration for all `tools/` modules.

#### Scenario: Structured logging replaces prints
- **GIVEN** any library module under `tools/`
- **WHEN** it emits diagnostic information
- **THEN** it uses `tools._shared.logging.get_logger()` to produce structured logs with contextual fields instead of `print`

#### Scenario: Subprocess execution is secured
- **GIVEN** a tool needs to invoke an external executable
- **WHEN** it calls `run_tool`
- **THEN** the executable path is absolute, environment is sanitized, a timeout is enforced, and failures yield a Problem Details payload

#### Scenario: All subprocess calls route via `run_tool`
- **GIVEN** any tooling module under `tools/`
- **WHEN** it needs to invoke a subprocess
- **THEN** the call goes through `tools._shared.proc.run_tool` with a timeout and sanitized environment

### Requirement: Docstring Builder Typed Pipeline
The docstring builder SHALL operate on typed models, enforce schema validation, and expose observability metrics while maintaining compatibility flags for rollout.

#### Scenario: Typed pipeline validates DocFacts
- **GIVEN** `DOCSTRINGS_TYPED_IR=1` and harvested symbols
- **WHEN** the builder generates DocFacts
- **THEN** payloads validate against `schema_docfacts.json`, emitting `SchemaViolationError` with Problem Details if validation fails

#### Scenario: CLI emits versioned JSON envelope
- **GIVEN** the builder runs with `--json`
- **WHEN** it completes (success or failure)
- **THEN** the output matches `schema/tools/docstring_builder_cli.json` and includes metrics updates and structured logs

### Requirement: Documentation Pipelines Schema Compliance
Documentation generators SHALL produce typed outputs validated against JSON Schemas and surface structured errors for failure modes.

#### Scenario: Agent analytics validates against schema
- **GIVEN** `tools/docs/build_agent_analytics.py`
- **WHEN** it emits machine-readable analytics
- **THEN** the payload validates against `schema/tools/doc_analytics.json` and failures raise `DocumentationBuildError` with Problem Details

#### Scenario: Graph builder secures subprocess usage
- **GIVEN** graph rendering requires calling Graphviz
- **WHEN** the tool runs
- **THEN** it invokes Graphviz via `run_tool`, enforces timeouts, and logs structured success/failure events

### Requirement: Navmap Pipeline Reliability
Navmap utilities SHALL use typed models, schema validation, and structured error handling for build, check, migrate, and repair operations.

#### Scenario: Navmap document round-trip succeeds
- **GIVEN** a generated navmap document
- **WHEN** it is validated
- **THEN** it conforms to `schema/tools/navmap_document.json`, with mismatches raising `NavmapError` containing Problem Details

#### Scenario: Repair script reports errors consistently
- **GIVEN** `tools/navmap/repair_navmaps.py`
- **WHEN** a repair fails
- **THEN** the script logs an error event, increments a Prometheus counter, and exits with Problem Details JSON when `--json` is requested

### Requirement: CLI Observability and Testing
All tooling CLIs SHALL expose structured observability, table-driven tests, and sample Problem Details fixtures ensuring reproducibility.

#### Scenario: CLI failure produces observability signals
- **GIVEN** any tooling CLI encounters an error
- **WHEN** it exits non-zero
- **THEN** it writes Problem Details JSON (when machine mode enabled), logs a structured error with correlation ID, and records Prometheus metrics

#### Scenario: CLI emits base envelope schema
- **GIVEN** any tooling CLI runs with `--json`
- **WHEN** it completes (success or failure)
- **THEN** the output validates against `schema/tools/cli_envelope.json` and includes a Problem Details payload on failure

#### Scenario: Tests cover edge and failure cases
- **GIVEN** the pytest suite under `tests/tools`
- **WHEN** it runs in CI
- **THEN** it executes table-driven tests covering happy path, edge, and failure modes for shared infrastructure, docstring builder, docs pipelines, navmap utilities, and CLI adapters

