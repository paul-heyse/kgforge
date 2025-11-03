## Context
- Static analysis reports (Pyright `tuple[()]` errors) identify tuple indexing in
  `search_api/faiss_gpu.py` and `kgfoundry_common/prometheus.py` that assumes tuple length
  ≥ 1. When upstream pipelines supply empty sequences, these paths crash with `IndexError`
  instead of returning structured Problem Details.
- Observability guidelines in `AGENTS.md` require structured logs, Prometheus metrics, and
  Problem Details with correlation IDs for failure paths. Current code skips these hooks
  when it fails before logging.

## Goals / Non-Goals
- **Goals**
  1. Add precondition guards before tuple indexing, raising domain-specific
     `ProblemDetailsError` (or equivalent) when sequences are empty.
  2. Ensure observability instrumentation records the failure via structured logging and
     metrics increments.
  3. Provide parametrized regression tests covering empty inputs, minimal valid tuples,
     and verifying emitted Problem Details/metrics.
  4. Document the guarded behavior with runnable examples (doctests) that illustrate the
     error payload.
- **Non-Goals**
  - Rewriting FAISS pipelines or Prometheus abstraction layers beyond guard logic.
  - Introducing new telemetry exporters.
  - Modifying schema alignment or plugin registries handled by other initiatives.

## Decisions
- Introduce helper function(s) (e.g., `first_or_problem(parts, context)`) that encapsulate
  length checks and Problem Details creation, ensuring consistent error messaging.
- Leverage existing exception taxonomy (e.g., `VectorSearchError`, `PrometheusError`) with
  `raise ... from e` semantics to preserve cause chains.
- Update observability instrumentation to log with `operation`, `status="error"`, and
  `correlation_id`; increment relevant Prometheus counters before raising.
- Expand pytest suites with table-driven tests for `faiss_gpu` and Prometheus helpers,
  using mocks/stubs to simulate empty sequences. Ensure doctests run for helper
  functions.

## Alternatives Considered
- **Allow IndexError to propagate** — rejected; fails observability requirements and leaves
  unstructured errors.
- **Return default values for empty tuples** — rejected; masks upstream issues and breaks
  contract semantics.
- **Introduce broad try/except** — rejected; targeted guards offer clearer diagnostics and
  avoid swallowing unrelated errors.

## Risks & Mitigations
- **Behavior change for callers** — mitigated by documenting new Problem Details behavior
  and ensuring status codes align with existing taxonomy.
- **Duplicate instrumentation** — mitigated by centralizing guard logic to avoid repeated
  logging/metrics code.
- **Test brittleness** — mitigated via fixtures/mocks rather than full FAISS builds.

## Migration Plan
1. Audit tuple indexing call sites in FAISS GPU module and Prometheus helpers; identify
   contexts and expected metadata.
2. Implement guard helpers that validate sequences, emit logs/metrics, and raise Problem
   Details errors with correlation IDs.
3. Replace direct indexing with guard helpers across affected modules.
4. Add regression tests (pytest parametrized) covering empty/minimal inputs and verifying
   emitted logs/metrics/Problem Details.
5. Update doctests and documentation to describe guarded behavior.
6. Run full quality gates and regenerate artifacts.

