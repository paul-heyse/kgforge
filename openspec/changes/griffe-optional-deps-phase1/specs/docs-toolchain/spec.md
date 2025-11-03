## ADDED Requirements
### Requirement: Optional dependency guards and guidance
Documentation tooling SHALL gate optional dependencies (Griffe, AutoAPI, Sphinx) behind
typed guard helpers that emit structured logs, metrics, and Problem Details with
correlation IDs when dependencies are absent. CLI utilities SHALL surface remediation
instructions and exit gracefully.

#### Scenario: Guarded import emits Problem Details
- **GIVEN** an environment without Griffe installed
- **WHEN** `safe_import_griffe()` runs
- **THEN** it raises `OptionalDependencyError` carrying an RFC 9457 Problem Details payload
  with remediation instructions and correlation ID, and observability metrics/logs record
  the failure

#### Scenario: CLI smoke test verifies graceful degradation
- **GIVEN** the documentation CLI executed in an environment missing Griffe extras
- **WHEN** the CLI runs
- **THEN** it prints the structured Problem Details payload, exits non-zero, and tests
  confirm logs/metrics capture the failure while providing install guidance

#### Scenario: Documentation/example alignment
- **GIVEN** doctest execution over updated tooling docstrings
- **WHEN** the example demonstrates detecting missing optional dependencies and instructing
  users to install extras
- **THEN** doctest passes, showing the exact command and Problem Details output

