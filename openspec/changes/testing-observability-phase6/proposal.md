## Why
Recent refactors add structured helpers and stricter typing, but our tests and observability coverage lag behind. We need comprehensive table-driven tests, doctests, and telemetry validation to prove the system behaves correctly under varied inputs and failure modes.

## What Changes
- [x] **ADDED**: Parametrized pytest suites covering new `SearchOptions` variants, schema validation round-trips, CLI orchestration flows, and Prometheus metrics wiring.
- [x] **ADDED**: Doctest/xdoctest snippets embedded in public docstrings (search helpers, FAISS usage, configuration examples).
- [x] **MODIFIED**: Logging/metrics code paths instrumented to emit structured logs, Prometheus counters/histograms, and OpenTelemetry spans on both success and failure.
- [x] **ADDED**: Tests validating idempotency and retry semantics for CLI/HTTP search behaviors, including Problem Details payload assertions.
- [ ] **REMOVED**: Legacy tests superseded by table-driven suites and unused examples.

## Impact
- **Affected specs (capabilities):** `testing-observability/core`
- **Affected code paths:** `tests/agent_catalog/**`, `tests/orchestration/**`, `tests/search_api/**`, `tests/kgfoundry_common/**`, observability utilities (`kgfoundry_common/logging.py`, `prometheus.py`, tracing helpers)
- **Data contracts:** Problem Details JSON examples, Prometheus metric names, documentation examples
- **Rollout plan:** Introduce tests incrementally, ensure telemetry dependencies available in CI, document new observability expectations.

## Acceptance
- [ ] Parametrized pytest suites cover all helper variants and failure scenarios, with fixtures mapping to spec requirements.
- [ ] Doctests/xdoctests run as part of pytest and pass without extra configuration.
- [ ] Structured logs, Prometheus metrics, and OpenTelemetry spans are emitted for failure paths; tests assert presence and field contents.
- [ ] CLI/HTTP idempotency/retry tests pass, verifying Problem Details payload stability and no duplicate side effects.

## Out of Scope
- Implementing new telemetry backends beyond log/metrics/traces.
- Rewriting business logic beyond test scaffolding requirements.
- Integration tests for external services (focus is on in-repo behaviors).

## Risks / Mitigations
- **Risk:** Test suite may become slow due to telemetry setup.
  - **Mitigation:** Use fixtures with scoped setup/teardown, leverage in-memory Prometheus registries and dummy OTEL exporters.
- **Risk:** Capturing logs/metrics may require additional dependencies.
  - **Mitigation:** Use stdlib logging handlers and Prometheus test registries already in repo; avoid heavy OTEL exporters.
- **Risk:** Idempotency tests might require CLI rework.
  - **Mitigation:** Stub external effects within tests, add helper functions for resetting state, and document assumptions.

## Alternatives Considered
- Rely on manual QA for observability — rejected due to inconsistency and lack of automation.
- Defer doctests — rejected; docstrings must remain runnable examples per standards.

