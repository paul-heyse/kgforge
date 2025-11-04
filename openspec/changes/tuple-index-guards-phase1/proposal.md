## Why
Several FAISS and Prometheus helper modules index tuple elements (`parts[0]`) without
defensive checks. Static analysis (Pyright `tuple[()]` errors) and runtime failures reveal
these paths raise `IndexError` when upstream code returns empty tuples. The absence of
guards also prevents structured Problem Details or observability metrics from capturing
the failure context. We need to harden these modules with explicit sequence checks,
Problem Details emission, and regression tests that verify empty-input handling.

## What Changes
- [ ] **MODIFIED**: Update FAISS GPU helpers and Prometheus instrumentation to guard
  tuple/sequence access with explicit checks, raising domain-specific Problem Details when
  inputs are empty.
- [ ] **ADDED**: Structured logging and metrics that capture the failure (operation,
  status, correlation ID) before raising errors.
- [ ] **ADDED**: Parametrized pytest suites covering empty inputs, minimal valid inputs,
  and error scenarios to confirm guards and observability hooks.
- [ ] **MODIFIED**: Documentation and doctests demonstrating the guarded behavior and
  Problem Details payloads.

## Impact
- **Capability:** `observability/core`
- **Code paths:** `src/search_api/faiss_gpu.py`, `src/kgfoundry_common/prometheus.py`,
  `src/kgfoundry_common/problem_details.py`, related tests under `tests/search_api/` and
  `tests/kgfoundry_common/`.
- **Contracts:** Emit RFC 9457 Problem Details with correlation IDs for empty-input
  failures; observability metrics/logging must reflect the guarded behavior.
- **Delivery:** Implement on branch `openspec/tuple-index-guards-phase1`, exercising the
  full quality-gate loop (`ruff`, `pyright`, `pyrefly`, `pyright`, `pytest`, `make artifacts`).

## Acceptance
- [ ] All tuple indexing hot paths include explicit sequence checks that raise
  Problem Details errors with structured logging/metrics.
- [ ] Tests cover empty inputs, ensuring guards fire and observability hooks record the
  failure.
- [ ] Quality gates (Ruff, Pyright, Pyrefly, MyPy, pytest, make artifacts) pass without
  suppressions.
- [ ] Documentation and doctests illustrate guarded FAISS/Prometheus usage and emitted
  Problem Details payloads.

## Out of Scope
- Redesigning FAISS adapters beyond tuple guards.
- Introducing new metrics exporters or observability pipelines.
- Altering schema alignment or plugin registry logic tracked in other changes.

## Risks / Mitigations
- **Risk:** Guards could change error semantics for existing callers.
  - **Mitigation:** Provide regression tests ensuring new errors map to existing exception
    taxonomy and document migration notes.
- **Risk:** Additional logging/metrics might add overhead.
  - **Mitigation:** Keep instrumentation lightweight and reuse existing observability
    helpers.
- **Risk:** Tests may require heavy FAISS setup.
  - **Mitigation:** Stub or mock sequences to simulate empty inputs; keep tests unit-level.

