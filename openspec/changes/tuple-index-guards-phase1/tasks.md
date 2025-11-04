## 1. Implementation
- [x] 1.1 Audit tuple access sites
  - [x] Inventory `parts[0]` or equivalent indexing in `src/search_api/faiss_gpu.py`,
        `src/kgfoundry_common/prometheus.py`, and related helpers.
  - [x] Document required context metadata (operation name, correlation ID, metrics).
- [x] 1.2 Introduce guard helpers
  - [x] Implement utilities that validate sequence length, emit structured logs/metrics,
        and raise Problem Details errors when empty.
  - [x] Ensure helpers support dependency injection for logging/metrics to ease testing.
- [x] 1.3 Apply guards to FAISS/Prometheus code paths
  - [x] Replace direct tuple indexing with guard helpers in FAISS GPU module.
  - [x] Update Prometheus helper functions to use guards before accessing tuple elements.
- [x] 1.4 Observability instrumentation
  - [x] Verify guards log `status="error"`, include `operation` and `correlation_id`, and
        increment failure metrics.
  - [x] Add/refresh Problem Details examples documenting the new error payload.
- [x] 1.5 Regression coverage
  - [x] Add parametrized pytest cases covering empty sequences, minimal valid tuples, and
        verifying metrics/log output via fixtures.
  - [x] Add doctest/xdoctest snippets demonstrating guard usage and failure payloads.
- [x] 1.6 Documentation updates
  - [x] Update relevant docs to describe guard helpers, observability expectations, and
        Problem Details references.

## 2. Testing & Quality Gates
- [x] 2.1 `uv run pytest -q tests/kgfoundry_common/test_sequence_guards.py`
- [x] 2.2 `uv run pytest --doctest-modules src/kgfoundry_common/sequence_guards.py`
- [x] 2.3 `uv run ruff format && uv run ruff check --fix`
- [x] 2.4 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.5 `uv run pyrefly check`
- [x] 2.6 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.7 Schema validation: All Problem Details payloads conform to RFC 9457

