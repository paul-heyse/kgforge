## ADDED Requirements
### Requirement: Keyword-Only Configured APIs
Public APIs SHALL accept keyword-only parameters or typed configuration objects (frozen dataclasses/TypedDict) instead of boolean positional arguments. Configuration objects MUST live in `tools/*/config*.py`, `docs/toolchain/config.py`, or `src/orchestration/config.py`, include NumPy-style docstrings, and satisfy Ruff `FBT00x` rules.

#### Scenario: Ruff enforces keyword-only usage
- **GIVEN** the updated modules under `docs/`, `tools/`, and runtime packages
- **WHEN** `uv run ruff check --select FBT` executes
- **THEN** no violations are reported, and attempting to call legacy functions with positional booleans raises `TypeError` with a deprecation message documented in tests

#### Scenario: Typed configs validated
- **GIVEN** a config dataclass such as `DocstringBuildConfig` defined in `tools/docstring_builder/config_models.py`
- **WHEN** invalid combinations (e.g., `emit_diff=True` while `enable_plugins=False`) are provided
- **THEN** a `ConfigurationError` is raised, `build_configuration_problem` renders the RFC 9457 payload defined in `schema/examples/problem_details/public-api-invalid-config.json`, and pytest asserts the JSON structure

#### Scenario: Legacy wrappers warn once
- **GIVEN** a consumer still calling the legacy positional API (e.g., `tools.docstring_builder.orchestrator.run_legacy(cache, False, True)`)
- **WHEN** the code executes
- **THEN** a single `DeprecationWarning` is emitted with guidance to construct the config object, and telemetry counters/logs record the legacy usage for cleanup tracking

### Requirement: Cache Access via Documented Interfaces
Consumers SHALL interact with caches and internal state through public Protocols or façade functions. Private attributes such as `_cache`, `_collect_module`, `_ParameterKind` MUST issue `DeprecationWarning` when accessed and are scheduled for removal in Phase 2.

#### Scenario: Private attribute access blocked
- **GIVEN** a module previously accessing `BuilderCache._cache`
- **WHEN** it runs after the migration
- **THEN** it obtains a `DocstringBuilderCache` via `get_docstring_cache()`, and attempting to access `_cache` raises `AttributeError` with remediation hint; pytest verifies both the accessor and the failure path

#### Scenario: Protocol compliance tested
- **GIVEN** cache Protocols defined in `tools.docstring_builder.cache.interfaces`
- **WHEN** pytest executes interface compliance tests
- **THEN** implementations satisfy protocol contracts and provide docstrings describing supported operations

### Requirement: Problem Details for Configuration Errors
All new configuration validation errors SHALL produce RFC 9457 Problem Details responses, documented via schema examples and surfaced in CLI/log outputs.

#### Scenario: CLI emits Problem Details on invalid config
- **GIVEN** a CLI invoked with invalid options (e.g., `tools-navmap-repair --force --dry-run`)
- **WHEN** the command runs
- **THEN** it logs the Problem Details JSON (fields `type`, `title`, `detail`, `instance`, `invalid_parameters`), exits with status code `2`, and integration tests validate the JSON against the schema example

#### Scenario: Docstrings reference Problem Details
- **GIVEN** updated module docstrings for public APIs
- **WHEN** docstring inspection runs via `tests/docs/test_docstrings.py`
- **THEN** documentation references the Problem Details example, includes runnable usage demonstrating validation errors, and doctest/xdoctest pass

### Requirement: Observability & Telemetry for Migration
The system SHALL emit structured logs and counters when legacy positional APIs or private caches are used so maintainers can track outstanding migrations.

#### Scenario: Telemetry counter increments
- **GIVEN** instrumentation registering `kgfoundry_public_api_legacy_usage_total`
- **WHEN** a legacy wrapper executes
- **THEN** the counter increments with labels `subsystem`, `api`, and the log line contains `warning="deprecated-public-api"`

#### Scenario: Dashboards track zero usage before shim removal
- **GIVEN** the observability dashboard fed by the telemetry counter
- **WHEN** the release readiness review happens
- **THEN** the dashboard shows zero legacy usage for two consecutive weeks before Phase 2 removes the shims


