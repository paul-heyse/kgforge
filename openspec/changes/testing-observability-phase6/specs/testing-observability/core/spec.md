## ADDED Requirements
### Requirement: Table-Driven Test Coverage
The system SHALL provide parametrized pytest suites for search options, schema round-trips, orchestration flows, and Prometheus wiring, mapping scenarios to spec requirements.

#### Scenario: Search options fixtures cover success/failure
- **GIVEN** the table-driven search options tests
- **WHEN** pytest runs
- **THEN** it executes cases for default facets, custom candidate pool, missing embedding model, and invalid facets, with assertions covering Problem Details outputs

#### Scenario: Orchestration CLI idempotency verified
- **GIVEN** the indexing CLI tests
- **WHEN** the same command runs twice with identical inputs
- **THEN** no duplicate outputs occur, structured logs indicate idempotent behavior, and Prometheus error counters remain consistent

### Requirement: Doctest Coverage for Public APIs
The system SHALL include doctest/xdoctest snippets in public docstrings (search helpers, FAISS usage, configuration) and run them as part of pytest.

#### Scenario: Search helper docstring runs
- **GIVEN** the docstring example for `build_search_options`
- **WHEN** doctests run
- **THEN** the example executes without setup beyond provided fixtures and validates Problem Details handling

#### Scenario: FAISS usage docstring validated
- **GIVEN** FAISS adapter docstrings
- **WHEN** doctests execute (skipping if FAISS unavailable)
- **THEN** they demonstrate typical index creation and teardown behavior without errors

### Requirement: Observability Assertions
The system SHALL verify that failure paths emit structured logs, Prometheus metrics, and OpenTelemetry spans containing required fields (operation, status, error type, correlation ID).

#### Scenario: Structured log assertion
- **GIVEN** a failing search operation test
- **WHEN** the test captures logs
- **THEN** the record includes `operation`, `status="error"`, `error_type`, and `correlation_id`, matching observability guidelines

#### Scenario: Prometheus and trace validation
- **GIVEN** a failure scenario in orchestration
- **WHEN** metrics and spans are captured
- **THEN** the Prometheus counter increments exactly once and the trace span is marked with error status and attributes documenting the failure

### Requirement: Idempotency & Retry Tests
The system SHALL include tests demonstrating CLI and HTTP search behaviors are idempotent and honor documented retry semantics.

#### Scenario: HTTP retry produces single effective call
- **GIVEN** a mocked backend that tracks call count
- **WHEN** the client retries a request after a transient Problem Details error
- **THEN** only one successful mutation occurs, and the test asserts the retry logic respects idempotency tokens

#### Scenario: CLI re-run without duplicate output
- **GIVEN** the CLI indexing command invoked twice
- **WHEN** outputs are compared
- **THEN** the second run completes without modifying artifacts and logs an idempotent status field

