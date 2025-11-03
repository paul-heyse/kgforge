## ADDED Requirements
### Requirement: Logging context hygiene
Logging, CLI, and docstring-builder contexts SHALL use frozen dataclasses or TypedDicts
with explicit required/optional partitions, surfaced through safe accessor helpers, and
emit Problem Details payloads containing correlation IDs on error paths.

#### Scenario: Immutable logging context with safe accessors
- **GIVEN** the hardened `LogContextExtra`
- **WHEN** logging helpers augment the context with `status`, `operation`, and correlation
  ID
- **THEN** they obtain a new instance (no in-place mutation), optional keys are accessed via
  helper methods, and type checkers report no `TypedDict` access errors

#### Scenario: ErrorReport problem details emission
- **GIVEN** a docstring-builder failure producing an `ErrorReport`
- **WHEN** the pipeline emits Problem Details
- **THEN** the payload matches the canonical example (status, detail, correlation ID), and
  tests assert the emitted JSON equals the stored schema example

#### Scenario: CLI result optional fields handled safely
- **GIVEN** a CLI command returning `CliResult`
- **WHEN** optional fields (e.g., `artifact_path`, `warning`) are absent
- **THEN** consumers use `.get` or accessor helpers without raising `KeyError`, and tests
  confirm success/error variants handle optional fields correctly

#### Scenario: Doctest coverage for accessor workflow
- **GIVEN** doctest execution over logging/docstring builder helper modules
- **WHEN** the example constructs contexts, updates them via helper methods, and emits a
  Problem Details payload
- **THEN** doctest passes, demonstrating the prescribed workflow and observability fields

