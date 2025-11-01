## MODIFIED Requirements
### Requirement: Typed Tooling Payload Models
Documentation, navmap, and CLI tooling SHALL operate on explicit msgspec/dataclass structures validated against JSON Schemas, eliminating `Any` leakage and enforcing time/path hygiene.

#### Scenario: Docstring cache round-trips through schema
- **GIVEN** docstring builder cache persistence
- **WHEN** cached payloads are serialized
- **THEN** they instantiate typed models, validate against `schema/tools/docstring_cache.json`, and tests assert parity for happy/edge/error cases

#### Scenario: Docs analytics emit typed envelopes
- **GIVEN** `tools/docs/build_agent_analytics.py`
- **WHEN** it produces analytics JSON
- **THEN** the result matches a typed model, validates against `schema/tools/doc_analytics.json`, and returns Problem Details on validation failure

#### Scenario: Msgspec structs type-check without Any
- **GIVEN** msgspec-backed helpers for CLI envelopes, navmap documents, docstring caches, analytics, and `sitecustomize`
- **WHEN** `mypy --config-file mypy.ini` runs against `tools/_shared`, `tools/docstring_builder`, `tools/docs`, `tools/navmap`, and `sitecustomize.py`
- **THEN** no `Any`-based diagnostics remain because helpers expose typed constructors, converters, and schema validators

#### Scenario: Legacy payloads migrate safely
- **GIVEN** previously stored cache or navmap payloads that follow the legacy structure
- **WHEN** the updated tooling deserializes them
- **THEN** migration helpers accept the old version, upgrade it to the new struct, and record a regression test demonstrating the behaviour

#### Scenario: CLI envelopes expose typed orchestrators
- **GIVEN** `tools/_shared/cli.py` and `tools/navmap/repair_navmaps.py`
- **WHEN** tests instantiate CLI envelopes or repair navmaps
- **THEN** they do so via typed factories (`CliEnvelope`, `CliEnvelopeBuilder`), use `pathlib.Path` for filesystem references, and assert that schema validation rejects naive-datetime payloads

#### Scenario: Navmap models enforce timezone-aware timestamps
- **GIVEN** `tools/navmap/document_models.py`
- **WHEN** navmap documents are created or mutated
- **THEN** typed fields require timezone-aware `datetime`, enforce deterministic ID generation, and round-trip via JSON Schema using parametrized tests

#### Scenario: Schema helpers emit versioned JSON Schema
- **GIVEN** `tools/_shared/schema.render_schema`
- **WHEN** it is invoked for a tooling model (for example `tools._shared.cli.CliEnvelope`)
- **THEN** the generated document conforms to JSON Schema 2020-12, includes the correct `$id`/version metadata, and matches the canonical file under `schema/tools/`

## ADDED Requirements
### Requirement: Layered Tooling Architecture
The tools package SHALL expose explicit, documented public APIs and maintain separation between domain logic and adapter layers enforced by import-linter rules.

#### Scenario: Public exports are explicit and documented
- **GIVEN** `tools/__init__.py` and `tools/docs/__init__.py`
- **WHEN** a consumer inspects the package or `pydoc`
- **THEN** exports match a curated `__all__`, every public symbol has a PEP 257 docstring referencing the exception taxonomy, and `schema/examples/tools/problem_details/tool-execution-error.json` documents a representative error payload

#### Scenario: Docstring builder CLI separates orchestration
- **GIVEN** `tools/docstring_builder/cli.py`
- **WHEN** the CLI is invoked in unit tests
- **THEN** thin adapter functions delegate to domain-layer orchestrators (`tools/docstring_builder/orchestrator.py`), enabling `pytest` to exercise logic without filesystem side effects and reducing cyclomatic complexity below Ruff limits

#### Scenario: Import-linter enforces layering
- **GIVEN** `tools/make_importlinter.py`
- **WHEN** `python tools/make_importlinter.py --check` executes
- **THEN** it emits package-specific layer contracts (for docstring builder, docs, and navmap) that prevent adapter modules from importing private domain internals, and CI fails when the contracts are violated

### Requirement: Instrumented Tool Execution
Tool subprocess orchestration SHALL provide structured logging, Prometheus metrics, OpenTelemetry traces, and idempotent retries with RFC 9457 Problem Details.

#### Scenario: Hardened subprocess wrapper emits correlation IDs
- **GIVEN** `tools/_shared/proc.run_tool`
- **WHEN** a subprocess is executed in tests with a seeded `contextvars.ContextVar`
- **THEN** structured logs include the correlation ID, timeout, cwd, and sanitized command, and a timeout raises `ToolExecutionError` with preserved cause via `raise … from` semantics

#### Scenario: Metrics and traces fire on success and failure
- **GIVEN** `tools/_shared/metrics.observe_tool_run`
- **WHEN** unit tests simulate success and failure outcomes (using a stub counter/histogram)
- **THEN** Prometheus counters/histograms increment with the correct operation/correlation labels and OpenTelemetry spans record error status, proven by parametrized tests

#### Scenario: Retry semantics documented and tested
- **GIVEN** tooling operations that support retries (e.g., navmap repair, docs generation)
- **WHEN** retryable errors occur in tests
- **THEN** the operation is idempotent, emits a Problem Details payload with retry guidance, and subsequent invocation succeeds without side effects

#### Scenario: `run_tool_with_retry` governs idempotent subprocesses
- **GIVEN** `tools._shared.proc.run_tool_with_retry`
- **WHEN** orchestrators wrap retryable commands (for example navmap drift repair)
- **THEN** retries honour configured backoff, record attempts in structured telemetry, and surface `ToolExecutionError` with preserved causes when exhaustion occurs

### Requirement: Hardened Tooling Configuration and Inputs
Tooling configuration and external inputs SHALL be validated via typed settings, sanitized filesystem access, and supply-chain checks.

#### Scenario: Settings module validates environment requirements
- **GIVEN** `tools/_shared/settings.py`
- **WHEN** namespace-specific settings (e.g., `DocbuilderSettings`) load with missing or malformed environment variables during tests
- **THEN** they fail fast with a `SettingsError` carrying Problem Details that enumerate the invalid fields and remediation guidance

#### Scenario: Path and URL inputs are sanitized
- **GIVEN** modules that consume user paths/URLs (`tools/navmap/build_navmap.py`, `tools/docs/build_graphs.py`, CLI adapters)
- **WHEN** tests provide malicious inputs (`../../etc/passwd`, `file://` URLs)
- **THEN** central helpers (`require_workspace_file`, `validate_allowed_url`) reject them with Problem Details, and pathlib-based utilities are used everywhere (no `os.path` regressions)

#### Scenario: Dependency and secret scanning gates pass
- **GIVEN** dependency updates or new optional extras
- **WHEN** `uv run pip-audit` and repository secret scanning tools execute in CI
- **THEN** they succeed without new findings, or changes ship with time-bound suppressions documented in the PR

