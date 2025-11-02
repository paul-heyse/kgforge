## Context
- Table-driven coverage is missing for new `SearchOptions` helpers, schema validation, and orchestration workflows.
- Docstrings lack runnable examples demonstrating search usage and FAISS operations.
- Observability helpers exist but tests do not assert structured log payloads, Prometheus metrics, or trace spans on failure paths.
- CLI/HTTP clients lack automated tests proving idempotency and retry behavior.

## Goals / Non-Goals
- **Goals**
  - Create comprehensive parametrized pytest suites for search options, catalog schemas, orchestration indexing, and Prometheus wiring.
  - Add doctest/xdoctest snippets to public APIs (search helpers, FAISS usage, configuration).
  - Validate structured logs, metrics, and traces emitted for both success and failure paths.
  - Confirm idempotency/retry semantics for CLI and HTTP search flows with explicit tests.
- **Non-Goals**
  - Replace integration testing with external services.
  - Introduce new telemetry systems beyond existing logging/Prometheus/OTel utilities.
  - Modify business logic outside what is necessary to expose telemetry or test hooks.

## Decisions
- Use pytest parametrization with fixture modules `tests/agent_catalog/conftest.py`, `tests/orchestration/conftest.py`, etc., sharing factory functions for options and payloads.
- Add test utilities capturing logs (via `caplog`), Prometheus metrics (via per-test `CollectorRegistry`), and OTEL spans (using in-memory exporters).
- Incorporate doctests by enabling `pytest --doctest-modules` for designated packages; ensure docstrings include minimal runnable examples.
- Implement idempotency fixtures resetting state (e.g., reinitializing in-memory SQLite or index directories between retries) to confirm repeated commands produce identical results.
- Update Problem Details helpers/tests to assert emitted JSON matches schema (load from `schema/examples/problem_details/*.json`).

## Alternatives
- Use snapshot testing for telemetry outputs — deferred; prefer structured asserts for field-level verification.
- Rely solely on doctest — rejected; doctests complement but do not replace table-driven pytest coverage.

## Risks / Trade-offs
- Additional tests may increase CI duration.
  - Mitigation: Scope fixtures carefully, use xdist if necessary, and profile test runs.
- Doctests may be brittle if docstrings change frequently.
  - Mitigation: Keep examples concise and covered by standard fixtures; run doctests locally before merges.
- Telemetry capture might fail in environments lacking exporters.
  - Mitigation: Provide optional dependencies and skip tests when exporters unavailable (with markers).

## Migration
- Introduce shared fixtures and utilities first.
- Add table-driven tests module by module (agent catalog, orchestration, search API, common logging).
- Insert doctests into docstrings as tests reach parity.
- Enable doctest plugin in pytest configuration; update CI settings accordingly.
- Document new testing expectations in `docs/contributing/testing.md`.

