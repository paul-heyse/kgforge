## Context
Pyrefly currently reports 97 suppressed diagnostics across the codebase.  The majority sit in optional dependency boundaries (FAISS/cuVS GPU adapters, Prometheus metrics, FastAPI decorators), documentation tooling, and high-throughput math routines where NumPy’s typing is imprecise.  The suppressions prevent us from catching regressions automatically and encourage ad-hoc casting or `Any` fallbacks that erode confidence in our typed interfaces.

## Goals / Non-Goals
- **Goals**
  - Eliminate all remaining `pyrefly` suppressions and justify any residual `type: ignore[...]` pragmas.
  - Provide typed wrappers/stubs so optional dependencies degrade gracefully without sacrificing static guarantees.
  - Ensure high-traffic code paths (doc generators, FAISS search, tooling observability) remain performance-neutral after refactor.
- **Non-Goals**
  - Redesign FAISS or FastAPI APIs beyond typing hygiene.
  - Introduce new runtime metrics or tracing behaviour (observability signatures remain unchanged).
  - Modify CI gating beyond adding a suppression manifest script.

## Decisions
1. **Shared Prometheus facade** – introduce `tools/_shared/prometheus.py` that returns typed metrics or no-op stubs. All tooling observability modules consume this facade (already implemented during initial workstream).
2. **FAISS/cuVS typed proxies** – reuse `search_api.types` protocols and ensure adapters import them unconditionally. Runtime modules guard optional dependencies via `None` checks instead of object fallbacks.
3. **FastAPI decorator wrappers** – define helper functions or enrich stubs so middleware and dependency decorators keep annotations intact (eliminating `type: ignore[misc]`).
4. **Suppression governance** – ship a lint command (`scripts/check_pyrefly_suppressions.py`) plus pre-commit hook that fails on new `type: ignore` usage without a ticket reference.

## Migration Plan
1. **Observability sweep**
   - Ensure all modules producing metrics/logs/traces import the shared `tools/_shared/prometheus.py` helpers.
   - Update module docstrings/examples, confirm structured logging fields, and document optional dependency fallbacks.
   - Run the static-check trio (`ruff`, `pyrefly`, `pyright`) across `tools/_shared`, docstring builder, docs tooling, and navmap packages after each batch.
2. **FAISS/cuVS adapters**
   - Expand FAISS/libcuvs stubs to cover GPU helpers and serialization entry points.
   - Introduce NumPy typing utilities and refactor adapters to consume them while guarding optional dependencies.
   - Validate CPU/GPU/cuVS matrix through the static-check trio; update documentation/Problem Details samples accordingly.
3. **FastAPI & settings**
   - Enhance FastAPI/Starlette stubs and add typed wrappers for middleware/dependency registration.
   - Refactor app/error modules plus settings to leverage the wrappers and emit consistent Problem Details.
   - Re-run `ruff`/`pyrefly`/`pyright` for `src/search_api` and `kgfoundry_common` after each milestone.
4. **Agent catalog math**
   - Implement shared NumPy helpers, refactor catalog modules to consume them, and ensure schema references remain accurate.
   - Provide doctest-driven examples for ranking/idempotency scenarios and verify static checks over the catalog package.
5. **Tooling scripts & docs**
   - Create typed facades for DuckDB/config handling inside `docs/_scripts/shared.py` and refactor dependent scripts/CLIs.
   - Ensure CLI entry points expose typed `main` functions, structured logging, and safe path handling.
   - Execute static checks (`ruff`, `pyrefly`, `pyright`) targeting docs and site scripts; refresh contributor docs.
6. **Audit & guardrails**
   - Implement the suppression manifest script, integrate it into pre-commit/CI, and update contributor guidelines.
   - Run a final static-check sweep to confirm no suppressions remain and document escalation procedures for future exceptions.

## Risks / Mitigations
- **Optional dependency drift** – removing fallbacks may break environments lacking Prometheus/FAISS.  Mitigate by preserving runtime guards (returning `None` registries, checking for `None` module handles) and covering fallback paths with tests.
- **NumPy typing gaps** – even with casts, new releases might change typing behaviour. Mitigate via helper modules and targeted tests that assert array shapes/dtypes.
- **Large diff surface** – typed wrappers touch several modules; stage workstreams and run targeted gates per cluster to keep changes reviewable.

## Testing Strategy
- Static analysis as the primary safety net: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run pyright --warnings --pythonversion=3.13` scoped to each workstream and globally before merge.
- Doctest/xdoctest snippets embedded in modules where examples add value (observability helpers, FAISS wrappers, NumPy utilities).
- Optional dependency rotations: run pyrefly/pyright in CPU-only environments and with GPU extras to validate fallback paths.
- Manual smoke runs (as needed) for doc builders/navmap repair/search API GPU flows, but no new pytest suites are introduced under this change.

