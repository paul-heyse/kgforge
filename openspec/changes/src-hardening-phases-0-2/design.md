# Phase 2 Design Note — src-hardening-phases-0-2

## Context
- Phase 0 baseline captured Ruff, pyrefly, mypy, pytest, architecture drift (placeholder until pytestarch tests exist), and pip-audit diagnostics (docstring issues intentionally deferred).
- Remaining blockers span typed API gaps (agent catalog client, parquet IO, observability exports), schema validation, security lint warnings, import contract tooling, and CI gates (pip-audit, pytest naming conflict).
- Phase 2 focuses on typed refactors, schema enforcement, observability/config hardening, and security fixes while keeping existing behaviour stable. Docstring remediation will occur in a later phase.

## Goals / Non-Goals
- **Goals**
  - Remove `Any` flows in catalog/search/registry stacks using TypedDict/Protocol abstractions.
  - Enforce JSON Schema 2020-12/OpenAPI 3.2 as source of truth with examples/round-trip tests.
  - Harden observability helpers, metrics, and configuration fail-fast behaviour; ensure async context propagation and idempotency.
  - Close outstanding Ruff, pyrefly, mypy, and pytest failures unrelated to docstrings.
  - Make pytestarch-based layering tests and pip-audit gates runnable with actionable output.
- **Non-Goals**
  - Docstring indentation/coverage fixes (targeted for future docstring-focused phase).
  - Large functional changes or feature work outside catalog/search/registry/observability scope.

## Decisions
- Introduce typed link/template helpers in `kgfoundry/agent_catalog/client.py` and restore curated exception exports in `kgfoundry_common.errors` to satisfy pyrefly/mypy.
- Wrap `pyarrow` access via typed facades (`ParquetChunkWriter`, schema validators) and update orchestration flows to pass typed payloads.
- Expand `kgfoundry_common/observability` with `MetricsRegistry`, `record_operation`, and Prometheus/OpenTelemetry unit tests; rename duplicate test modules to resolve pytest import mismatch.
- Pin `import-linter` to a compatible version or adapt `tools/check_imports.py` for v2 syntax so import contracts run reliably; adjust config as needed.
- Run `pip-audit` against a built wheel or requirements file, recording results and suppression policy instead of auditing the editable install directly.
- Systematically replace unsafe `pickle`, add HTTP timeouts, and sanitize subprocess/network usage per Ruff security lint guidance.
- Adopt pytestarch suites (e.g., `tests/architecture/test_layers.py`) to enforce layering; integrate results into CI instead of import-linter.

## Work Plan
1. **Typed API cleanup**
   - Refactor agent catalog client link builders and namespace exports.
   - Enforce Protocol/TypedDict usage across catalog/search/registry modules; remove `Any` and justify remaining `# type: ignore`.
2. **Schema & serialization**
   - Author/refresh schemas under `schema/search/` with examples/version notes; add Spectral lint + runtime validation.
   - Update orchestration parquet flows and tests to use typed conversions.
3. **Observability & config**
   - Finalize metrics registry and record-operation utilities; add Prometheus/Otel tests and rename conflicting test modules.
   - Add config fail-fast tests (missing env → `SettingsError` + Problem Details) and async context propagation checks.
4. **Security & reliability**
   - Fix Ruff security warnings (`S403`, `S404`, `S113`, `S104`), path traversal guards, and input validation tests.
   - Ensure idempotency tests cover catalog/registry mutations and retries.
5. **Tooling gates**
   - Resolve architecture tests to emit machine-readable reports (pytest JSON or JUnit) and capture violations in execution logs.
   - Establish repeatable pip-audit flow (wheel audit or requirements lock); document outcomes.
   - Update telemetry dashboards and capture baseline metrics for search and schema validation.

## Testing Strategy
- Extend pytest with table-driven scenarios covering happy/edge/failure/injection/idempotency cases; map tests to spec scenarios.
- Ensure doctest/xdoctest runs for new schema examples (docstrings unchanged otherwise).
- Add unit tests for metrics/observability, config fail-fast, async context, and parquet schema round-trips.
- Run benchmarks for FAISS/BM25/SPLADE with seeded data; record budgets and regression thresholds.
- Re-run full gate suite (Ruff, pyrefly, mypy, pytest + architecture suites, pip-audit) and archive outputs in execution note.

## Risks / Mitigations
- **Scope creep**: adhere to phased checklist; defer docstring/Phase 3 tasks until Phase 2 acceptance met.
- **Tooling churn**: document dependency pins (msgspec, pytestarch) and audit flows to prevent recurring environment drift.
- **Telemetry regressions**: maintain compatibility shims and flag-controlled rollouts; monitor dashboards before flipping defaults.
- **Complexity regressions**: track Ruff complexity warnings and refactor early to avoid large diff late in the phase.
