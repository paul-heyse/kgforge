## ADDED Requirements
### Requirement: Public API Hygiene
Search packages SHALL expose explicit public APIs via `__all__`, follow PEP 8 names, provide PEP 257 one‑line docstrings, and publish fully annotated signatures.

#### Scenario: Exports declared via __all__
- **GIVEN** a public module under `search_api` or `agent_catalog`
- **WHEN** it is imported
- **THEN** it defines `__all__` listing public symbols and hides internal helpers

#### Scenario: Docstrings present and typed
- **GIVEN** a public function/class
- **WHEN** docs are generated
- **THEN** the symbol has a PEP 257 one‑sentence summary and fully annotated parameters/returns

### Requirement: Typed Search Interfaces
Agent catalog and search services SHALL operate on typed Protocols and TypedDicts, eliminating `Any` from public interfaces.

#### Scenario: FAISS adapter implements Protocol
- **GIVEN** `FaissIndexProtocol`
- **WHEN** the adapter is instantiated
- **THEN** mypy confirms it implements the Protocol and all public methods are fully typed

#### Scenario: Search response matches schema
- **GIVEN** a successful search
- **WHEN** the response is generated
- **THEN** it validates against `schema/search/search_response.json`

### Requirement: Secure SQL & Data Access
All SQL statements SHALL use parameterized queries with sanitized inputs; string-based SQL concatenation is prohibited.

#### Scenario: Parameterized DuckDB queries
- **GIVEN** user input provided to a DuckDB query
- **WHEN** the statement executes
- **THEN** placeholders and parameter bindings are used, preventing SQL injection and satisfying S608 checks

#### Scenario: SQL injection attempt rejected
- **GIVEN** malicious input containing SQL control characters
- **WHEN** a search endpoint receives it
- **THEN** the system rejects the input with a Problem Details response and logs the attempt

### Requirement: Schema-backed HTTP/CLI/MCP Responses
All search HTTP responses, CLI outputs, and MCP messages SHALL validate against published JSON Schemas and emit RFC 9457 Problem Details on failure.

#### Scenario: HTTP failure returns Problem Details
- **GIVEN** a service error during search
- **WHEN** the FastAPI endpoint responds
- **THEN** it returns a Problem Details payload that validates against the canonical schema and logs the correlation ID

#### Scenario: CLI `--json` emits envelope
- **GIVEN** the CLI executes with `--json`
- **WHEN** it completes (success or error)
- **THEN** the output conforms to `schema/tools/cli_envelope.json` and `schema/search/catalog_cli.json`

### Requirement: Structured Logging & Metrics
Search operations SHALL emit structured logs with correlation IDs and metrics capturing counts, latency, and errors.

#### Scenario: Metrics recorded per search
- **GIVEN** a search request
- **WHEN** it completes
- **THEN** counters increment (`search_requests_total`), histograms observe duration (`search_duration_seconds`), and logs include `correlation_id`, `operation`, and `status`

#### Scenario: Metrics available without Prometheus
- **GIVEN** environments lacking Prometheus
- **WHEN** metrics helpers are used
- **THEN** stub implementations avoid AttributeError and maintain consistent logging

### Requirement: Observability & Timeout Controls
All outgoing operations (HTTP calls, DuckDB queries, FAISS operations) SHALL enforce timeouts and log slow paths.

#### Scenario: Timeout documented and enforced
- **GIVEN** a long-running DuckDB query
- **WHEN** it exceeds the configured timeout
- **THEN** the system aborts, logs a warning with correlation ID, increments `search_errors_total`, and returns Problem Details

### Requirement: Concurrency & Context Correctness
Async code SHALL propagate correlation IDs via `contextvars`, avoid blocking calls, and document/enforce timeouts and cancellations.

#### Scenario: Correlation ID middleware propagates context
- **GIVEN** a FastAPI request with `X-Correlation-ID`
- **WHEN** the request is processed
- **THEN** the correlation ID is set in a `ContextVar` and appears in structured logs for downstream operations

#### Scenario: Blocking operations isolated
- **GIVEN** an async endpoint calling FAISS or DuckDB
- **WHEN** the operation runs
- **THEN** it executes in a threadpool or non‑blocking path, respecting configured timeouts and cancellation

### Requirement: Documentation & Tests
### Requirement: Doctests & Packaging
Examples SHALL execute via doctest/xdoctest and packaging SHALL succeed with optional extras.

#### Scenario: Doctests execute in CI
- **GIVEN** docstrings/examples in public modules
- **WHEN** `pytest -q` runs
- **THEN** doctests/xdoctests execute without failure

#### Scenario: Wheel builds and installs with extras
- **GIVEN** a clean virtual environment
- **WHEN** `pip wheel .` and `pip install .[faiss,duckdb,splade]` run
- **THEN** both commands succeed and package metadata is correct

### Requirement: Layering & Import Boundaries
Search packages SHALL satisfy import‑linter contracts preventing upward/cross‑layer imports.

#### Scenario: Import contracts pass
- **GIVEN** `importlinter.cfg`
- **WHEN** contracts run in CI
- **THEN** `search_api` does not import app layers above, and `registry` does not depend on `search_api`

### Requirement: Idempotency & Error Retries
Externally triggered operations SHALL be idempotent where feasible and have documented retry semantics.

#### Scenario: Idempotent repeated search
- **GIVEN** the same query and deterministic state
- **WHEN** it runs repeatedly
- **THEN** responses are stable within tolerance and side effects do not accumulate
Tests SHALL cover happy paths, edge cases, and failure modes; doctests/xdoctests SHALL execute successfully; documentation SHALL reference schemas and metrics.

#### Scenario: Table-driven tests cover injection attempts
- **GIVEN** `tests/search_api/test_endpoints.py`
- **WHEN** the suite runs
- **THEN** it includes parametrized cases for valid input, invalid schema, and SQL injection attempts, all passing

#### Scenario: Doctests execute
- **GIVEN** docstrings/examples in public modules
- **WHEN** `pytest -q` runs
- **THEN** doctests/xdoctests execute without failure

### Requirement: Rollout Safeguards
Typed search pathways SHALL be gated behind feature flags with a documented rollout plan.

#### Scenario: Feature flag disables typed pipeline
- **GIVEN** `AGENT_SEARCH_TYPED=0`
- **WHEN** the system runs
- **THEN** legacy behavior is used while emitting a WARN log about deprecation, enabling quick rollback

