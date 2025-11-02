## Context
- `kgfoundry_common.logging.LoggerAdapter` subclasses `logging.LoggerAdapter` without proper generics, exposes `Any` kwargs, and lacks structured helper functions; several modules instantiate loggers without `NullHandler`.
- Structured logging expectations (operation, status, duration, correlation IDs) are inconsistently implemented; some modules rely on free-form strings and f-strings at error sites.
- Configuration utilities mix manual `os.getenv` calls and partially documented defaults, leading to untyped surfaces and weak validation.
- `problem_details`, `errors`, and `serialization` contain raw f-strings inside exception constructors, omit `raise ... from e`, and capture wall-clock timing rather than `time.monotonic()`.
- Missing or incomplete docstrings violate PEP 257, and doctest/xdoctest coverage is absent for common helpers.

## Goals / Non-Goals
- **Goals**
  - Deliver typed structured logging helpers with consistent fields, contextvar-backed correlation IDs, and automatic `NullHandler` registration.
  - Migrate configuration to `pydantic_settings.BaseSettings`, ensuring typed env overrides, docstrings, and failure-on-invalid behavior.
  - Standardize Problem Details/error raising to use typed exceptions, preserved causes, and monotonic timing.
  - Fill docstring gaps, add doctest-ready examples, and ensure tests cover new behaviors.
- **Non-Goals**
  - Introducing new telemetry backends or metrics exporters.
  - Rewriting downstream service log ingestion pipelines.
  - Changing serialization formats beyond the safety/typing enhancements.

## Decisions
- Create `StructuredLoggerAdapter` with `type logging.LoggerAdapter[logging.Logger]` syntax via `typing` generics, exposing methods `bind`, `log_success`, `log_failure`, `log_io`, each returning typed `LogRecord` friendly dicts.
- Provide module-level helper `get_logger(name: str, *, extra: Mapping[str, object] | None = None) -> StructuredLoggerAdapter` that ensures a `NullHandler` is attached exactly once.
- Introduce `ContextVars` (e.g., `request_id_var`, `correlation_id_var`) to propagate identifiers; logging helpers fetch them automatically and accept overrides.
- Implement `AppSettings(BaseSettings)` within `kgfoundry_common.config`, enumerating all env fields with explicit types, default strategies, and docstrings; expose `load_config()` returning a frozen settings instance and raising `ConfigurationError` on invalid input.
- Update `problem_details` helpers to build payloads via dataclasses/TypedDicts, always `raise ProblemDetailsError from err` when wrapping underlying exceptions, and ensure tests verify payload shapes.
- Replace ad-hoc string formatting with `logging.LoggerAdapter` structured calls and safe `.format` or f-string assignments stored before raising; avoid constructing messages inline inside exceptions.
- Add doctest examples for `get_logger`, `log_success`, configuration loader, and Problem Details builder; integrate them into pytest.

## Alternatives
- Use `structlog` or another third-party structured logging solution — rejected to minimize dependencies and keep existing stdlib integrations.
- Keep manual env parsing with incremental validation — rejected because it cannot provide full type coverage or auto-doc generation.
- Introduce global singletons for config/logging — rejected in favor of dependency injection and testability.

## Risks / Trade-offs
- Structured logging helper adoption might require downstream code changes.
  - Mitigation: Provide compatibility shim functions returning legacy dictionaries, document migration guide.
- `pydantic_settings` instantiation may increase startup time slightly.
  - Mitigation: Cache settings instance; benchmark to confirm negligible impact.
- Enforcing `raise ... from e` could reveal hidden bugs where downstream code assumes bare exceptions.
  - Mitigation: Communicate changes, update tests to assert chained exceptions, and provide helper functions to extract original causes.

## Migration
- Ship helper adoption in tandem with configuration changes to minimize split-brain states.
- Provide release notes and examples demonstrating new helper usage; include sample Problem Details JSON in docs.
- Maintain deprecated wrappers (marked with `warnings.warn`) for one release cycle; track usage via logging to ensure safe removal later.
- After deployment, monitor logs for field-mismatch warnings and update documentation accordingly.

