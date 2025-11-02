## Why
Despite recent hardening, 97 `pyrefly` suppressions remain scattered across the repository.  They mask optional dependency gaps (FAISS/cuVS, Prometheus, FastAPI), NumPy shape ambiguities, and missing stubs for documentation tooling.  As long as these suppressions persist, static analysis cannot provide the “fail fast” signal we expect from the typing pipeline, leaving performance-sensitive search paths, doc automation, and GPU adapters susceptible to silent regressions.

## What Changes
- Deliver a typed Prometheus facade across the repo, migrate every observability module to it, and document structured logging/metrics/tracing expectations.
- Harden FAISS/cuVS adapters and NumPy-heavy logic with shared typing utilities, richer stubs, and optional dependency wrappers that keep GPU fallbacks safe.
- Strengthen FastAPI and settings surfaces via enhanced stubs, typed middleware/dependency helpers, and explicit Problem Details handling.
- Introduce reusable NumPy typing helpers and refactor the agent catalog to lean on them while preserving performance budgets and schema contracts.
- Modernise docs/tooling scripts with typed facades for DuckDB/config loading, structured logging, and security-conscious path handling; update contributor documentation accordingly.
- Add governance guardrails (suppression manifest script, contributor guidance, CI enforcement) so new suppressions cannot land without justification.

## Impact
- **Affected specs:** `code-quality`
- **Affected code:** `tools/_shared/**`, `tools/**/observability.py`, `src/search_api/faiss_adapter.py`, `src/vectorstore_faiss/gpu.py`, `src/kgfoundry_common/**`, `docs/_scripts/**`, `site/_scripts/**`, `stubs/**`, build scripts, and governance automation.
- **Rollout:** Execute the detailed task plan incrementally, validating each cluster with ` uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini`. Coordinate with search, tooling, and docs teams for GPU/dependency-specific checks; no behavioural regressions are expected.

