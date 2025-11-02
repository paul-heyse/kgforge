## ADDED Requirements
### Requirement: Structured Logging Backbone
The system SHALL provide typed structured logging helpers that propagate contextvars, attach a `NullHandler` to every library logger, and emit JSON-friendly payloads for downstream ingestion.

#### Scenario: Helper enriches log with context
- **GIVEN** a request context with `request_id="req-123"` and `operation="config.load"`
- **WHEN** `log_success` is invoked via `get_logger(__name__)`
- **THEN** the emitted record contains the request metadata, status=`"ok"`, and no `Any` types appear in type-checking results

#### Scenario: NullHandler prevents configuration leaks
- **GIVEN** a consumer importing `kgfoundry_common.logging`
- **WHEN** they obtain a logger without configuring handlers
- **THEN** no duplicate or `No handler could be found` warnings occur because a `NullHandler` is already attached

#### Scenario: Structured log failure surfaces Problem Details link
- **GIVEN** a caught domain exception wrapped via `log_failure`
- **WHEN** the helper emits the record
- **THEN** the payload includes `error_type`, `problem_type`, and `detail` keys matching the Problem Details schema

### Requirement: Typed Configuration & Documentation
The system SHALL load configuration through `pydantic_settings.BaseSettings`, enforce typed environment overrides, and document every field with runnable examples.

#### Scenario: Invalid environment variable raises typed error
- **GIVEN** an environment variable with an invalid value (e.g., `LOG_LEVEL=verbose`)
- **WHEN** `load_config()` executes
- **THEN** it raises a `ConfigurationError` derived from `ValueError`, chained from the original validation error, and logs the failure via `log_failure`

#### Scenario: Docs reflect configuration contract
- **GIVEN** the generated documentation artifacts
- **WHEN** `make artifacts` completes
- **THEN** the configuration reference page lists each field, default, type, and env variable, with examples that copy-paste into `.env`

#### Scenario: Doctest proves helper usage
- **GIVEN** the docstring examples for `AppSettings`
- **WHEN** doctests run during pytest
- **THEN** the example demonstrating env override passes without additional fixtures

### Requirement: Problem Details & Serialization Consistency
The system SHALL raise Problem Details exceptions with preserved causes, avoid inline f-strings in exception constructors, and rely on monotonic timing for duration metrics.

#### Scenario: Exception chaining preserved
- **GIVEN** a serialization failure triggered by malformed JSON
- **WHEN** `ProblemDetailsError` is raised via helper functions
- **THEN** `__cause__` references the original `json.JSONDecodeError`, and the emitted Problem Details payload matches `schema/examples/problem_details/serialization-error.json`

#### Scenario: Monotonic timing recorded
- **GIVEN** a serialization benchmark using the timing utility
- **WHEN** durations are measured
- **THEN** the recorded value originates from `time.monotonic()` and is included as `duration_ms` in structured logs

#### Scenario: Tests enforce safe string formatting
- **GIVEN** the updated tests for error modules
- **WHEN** ruff and pytest execute
- **THEN** they confirm no exception message is constructed via inline f-string; messages are assigned before raising and validated via assertions

