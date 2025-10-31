## ADDED Requirements
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

### Requirement: Documentation & Tests
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

