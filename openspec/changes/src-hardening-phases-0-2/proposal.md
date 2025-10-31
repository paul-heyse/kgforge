## Why
Ruff, pyrefly, and mypy still surface hundreds of diagnostics across the `src/`
tree. The failures are concentrated in three areas: (1) missing guardrails that
let regressions ship unnoticed, (2) placeholder docstrings and undefined
exception contracts that block documentation and schema parity, and
(3) pervasive `Any` usage, insecure serialization, and schema gaps within the
agent catalog, search API, and registry pipelines. Without a coordinated plan
we cannot restore type cleanliness, publish reliable docs, or harden the stack
against security regressions.

## What Changes
- [ ] **ADDED**: Phase 0 requirement to run deterministic quality gates, capture docstring coverage and telemetry baselines, and publish the four-item design note before implementation begins.
- [ ] **ADDED**: Phase 2 requirement to deliver typed, documented APIs, enforce JSON Schema/OpenAPI contracts, expand parametrized/doctor-tested suites, harden logging/config/security boundaries, and eliminate `Any` usage.
- [ ] **ADDED**: Phase 3 requirement to validate packaging, documentation, observability, and rollout artefacts, flipping feature flags once telemetry proves stable.
- [ ] **MODIFIED**: `schema/`, `docs/`, observability tooling, packaging metadata, and handbook guidance to support schemas, examples, benchmarks, and monitoring introduced in Phases 2 and 3.
- [ ] **BREAKING**: None; CLI and HTTP interfaces remain backward compatible while optional typed envelopes are guarded by feature flags.

## Impact
- **Affected specs:** `src-hardening-roadmap` (new capability) coordinating quality gates, API clarity, type-safety, and rollout governance.
- **Affected code paths:**
  - Project-wide tooling (`scripts/bootstrap.sh`, `tools/check_imports.py`, import-linter contracts, observability configuration).
  - Documentation/schema generators (`tools/**`, `docs/**`, `schema/**`, Agent Portal artefacts).
  - Runtime modules: `kgfoundry/agent_catalog/**`, `search_api/**`, `registry/**`, `vectorstore_faiss/**`, `kgfoundry_common/**` (serialization, settings, telemetry).
  - Packaging metadata (`pyproject.toml`, extras) and changelog/versioning workflows.
- **Rollout:**
  - Phase 0 gates progress on baseline diagnostics, design notes, and telemetry dashboards.
  - Phase 2 delivers typed pathways behind `AGENT_SEARCH_TYPED` / `SEARCH_API_TYPED`, with compatibility shims and scripted tests/benchmarks.
  - Phase 3 executes full quality/packaging gates, verifies telemetry, documents migration & rollback steps, and archives execution artefacts.
- **Risks & mitigations:**
  - **Scope creep:** Spec-defined tasks grouped by principle; import-linter + OpenSpec validation stop cross-phase leakage.
  - **CI drift:** Baseline commands mandated in Phase 0; Phase 3 reruns complete gate suite before flag flips.
  - **Performance regressions:** Benchmarks and budgets recorded in Phase 2; telemetry monitored during rollout.

## Acceptance
- **Phase 0 baseline:** Recorded outputs for `uv run ruff format && uv run ruff check src`, `uv run pyrefly check --show-suppressed`, `uv run mypy --config-file mypy.ini`, `uv run pytest -q --maxfail=1`, doctest coverage snapshot, `python tools/check_imports.py`, `uv run pip-audit --strict`, `openspec validate src-hardening-phases-0-2`, and docstring coverage metrics; four-item design note maps requirements to planned tests.
- **Phase 2 delivery:**
  - Public APIs are fully typed with PEP 257/NumPy docstrings; Problem Details samples committed and referenced.
  - JSON Schema 2020-12/OpenAPI 3.2 artefacts validate against meta-schemas and Spectral; round-trip tests prove parity.
  - Pytest suites include parametrized happy/edge/failure/retry/injection scenarios; doctest/xdoctest runs clean.
  - pyrefly + mypy report zero diagnostics for targeted modules; no unexplained `# type: ignore`.
  - Structured logs, metrics, and traces emit correlation IDs; settings flow through typed env configuration; async boundaries document timeout/cancellation behaviour.
  - Security guardrails: no unsafe `pickle`/`eval`, inputs validated, `pip-audit --strict` clean, secret scan clean, path traversal prevented.
  - Modularity enforced via updated import-linter contracts; functions refactored to single responsibility; idempotency tests demonstrate retry semantics.
  - Performance benchmarks meet documented budgets or have approved mitigations.
- **Phase 3 rollout:**
  - Full quality, packaging, and docs gates rerun post-refactor (`pip wheel .`, clean `pip install .[faiss,duckdb,splade,gpu]`, `make artifacts && git diff --exit-code`, updated Agent Portal examples).
  - CHANGELOG uses SemVer language, deprecations carry warnings + migration paths, and docs cross-link schemas/examples.
  - Telemetry dashboards confirm failing scenarios emit log + metric + trace with correlation IDs and respect budgets.
  - Feature flags flip only after staging burn-in; rollback steps documented; execution note archives command outputs, telemetry snapshots, benchmark baselines, and residual risks.

