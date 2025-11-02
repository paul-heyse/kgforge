## 1. Implementation
- [ ] **1.1 Observability facade consolidation**
  - [x] Introduce `tools/_shared/prometheus.py` with typed counter/histogram builders and update tooling observability modules (`tools/_shared/metrics.py`, `tools/docstring_builder/observability.py`, `tools/docs/observability.py`, `tools/navmap/observability.py`).
  - [x] Audit every remaining Prometheus consumer (e.g., `tools/_shared/proc.py`, `tools/generate_pr_summary.py`, `tools/navmap/build_navmap.py`, `src/kgfoundry_common/observability.py`) and migrate them to the shared builders; ensure each migration produces a module-level logger with a `NullHandler`, structured log extras (`operation`, `status`, `duration_ms`, `correlation_id`), and Ops-friendly docstrings (principles 1, 5, 9).
  - [x] Update module docstrings to include copy-ready usage examples plus notes on optional dependency fallbacks; link to the relevant spec scenarios and Problem Details exemplars so operators can diagnose missing metrics (principles 1, 5, 9, 13).
  - [x] Introduce `tools/_shared/observability_facade.md` (or equivalent developer note) summarising the contract: typed APIs, when to emit metrics, how to attach tracing spans, and how to extend the helper without reintroducing suppressions (principles 1, 7, 9, 14).
  - [x] Ensure every public helper exposes a fully annotated signature (PEP 695 generics where applicable) and that internal helpers remain module-private (`_` prefixed). Run `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini tools/_shared tools/docstring_builder tools/docs tools/navmap` after each cluster of changes (principles 1, 4).
  - [x] Add doctest/xdoctest snippets demonstrating: (a) metric creation under Prometheus, (b) metric creation without Prometheus (no-op), and (c) structured logging with correlation IDs; verify via `python -m xdoctest` or equivalent lightweight doctest harness (principles 3, 9, 13).
  - [x] Record fallback behaviour in the CHANGELOG or release notes so downstream users know that observability APIs are stable and SemVer-compliant (principles 11, 14).
- [ ] **1.2 FAISS & cuVS adapter hardening**
  - [x] Rewrite `search_api/faiss_adapter.py` and `vectorstore_faiss/gpu.py` to consume typed protocols from `search_api.types`, replacing object fallbacks and NumPy `type: ignore` casts.
  - [ ] Extend `stubs/faiss/**` and `stubs/libcuvs/**` to cover GPU parameter spaces, multi-GPU cloning functions, serialization helpers, and constants (e.g., `GpuIndexFlatIP`, `index_cpu_to_gpu_multiple`), ensuring docstrings reflect upstream behaviour and highlight optional extras (principles 1, 11).
  - [ ] Create `kgfoundry_common/numpy_typing.py` housing reusable vector helpers (`FloatMatrix`, `IntVector`, `normalize_l2`, `topk_indices`, `safe_argpartition`) with exhaustive annotations, docstrings, and doctest examples demonstrating edge cases (zero vectors, NaNs). Cross-reference these helpers from both FAISS modules and the agent catalog (principles 1, 4, 12, 16).
  - [ ] Encapsulate GPU initialisation and cuVS toggles in a small adapter (e.g., `search_api/faiss_gpu.py`) that returns typed resources and options. Document idempotency, error rewrites (Problem Details), and fallback semantics (principles 5, 8, 9, 15).
  - [ ] Validate behaviour without adding pytest suites by running `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini src/search_api/faiss_adapter.py src/vectorstore_faiss/gpu.py` under three environments: base (CPU), GPU extras (`uv sync --extra gpu`), and GPU extras with cuVS intentionally missing (principles 4, 9, 12).
  - [ ] Update documentation (`docs/explanations/FAISS_*`) and Problem Details samples (under `schema/examples/search/problem_details/`) to reflect the new error surfaces; ensure examples reference the typed helpers and note backward-compatibility guarantees (principles 2, 5, 13).
- [ ] **1.3 FastAPI & settings typing**
  - [ ] Enhance `stubs/fastapi/**` and `stubs/starlette/**` to model `FastAPI`, middleware decorators, `Depends`, `Header`, `Security`, and response types (e.g., `JSONResponse`, `StreamingResponse`). Include docstrings and ensure the stubs cover attribute access we perform (principles 1, 4).
  - [ ] Add typed wrappers (`typed_middleware`, `typed_exception_handler`, `typed_dependency`) in `src/search_api/fastapi_helpers.py` (or similar) that preserve type information while delegating to FastAPI runtime constructs. Each helper should emit structured logs, accept correlation IDs, and enforce timeouts explicitly (principles 5, 7, 8, 9).
  - [ ] Refactor `src/search_api/app.py`, `kgfoundry_common/errors/http.py`, and related modules to consume the wrappers, maintain layered architecture (domain → adapters → HTTP), and surface Problem Details JSON from a single, typed location (principles 1, 5, 7).
  - [ ] Refresh settings modules (`kgfoundry_common/settings.py`, `tools/_shared/settings.py`) to use `pydantic_settings.BaseSettings` with field-level annotations, environment variable metadata, and docstrings referencing configuration schemas. Provide doctest snippets that demonstrate loading, overriding, and failure-fast behaviour when required env vars are missing (principles 1, 6, 13).
  - [ ] Execute `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check src/search_api/app.py kgfoundry_common/errors/http.py`, and `uv run mypy --config-file mypy.ini src/search_api kgfoundry_common` after each milestone to ensure suppressions are not reintroduced (principles 4, 5).
  - [ ] Update OpenAPI schema validations (via existing tooling) to confirm error envelopes still align with RFC 9457 and that new/updated endpoints remain backward compatible (principles 2, 5, 13).
- [ ] **1.4 Agent catalog & NumPy helpers**
  - [ ] Implement `kgfoundry_common/numpy_typing.py` (see 1.2) and ensure agent catalog modules import from it exclusively. Provide docstrings with usage notes, performance expectations, and idempotency guarantees (principles 1, 12, 15).
  - [ ] Refactor `kgfoundry/agent_catalog/search.py`, `kgfoundry/agent_catalog/build.py`, and related files to separate pure scoring logic from I/O. Expose public APIs via explicit `__all__`, ensure modules return typed Problem Details on errors, and reference JSON schemas for cross-boundary data (principles 1, 2, 7).
  - [ ] Provide doctest snippets demonstrating vector normalisation, top-K ranking, and deterministic fallbacks when distances tie. Ensure the doctests run as part of `python -m xdoctest` or a similar workflow (principles 3, 12, 13).
  - [ ] Run `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini src/kgfoundry/agent_catalog` to validate the refactor. Document performance budgets (e.g., acceptable latency/memory) and confirm they hold via small “developer note” measurements (principles 4, 12).
  - [ ] Update schema references (JSON Schema 2020-12) and dataset docs to indicate any backward/forward compatibility constraints; coordinate with portal/navigation artifacts (`make artifacts`) to ensure documentation links remain intact (principles 2, 13).
- [ ] **1.5 Tooling & doc scripts cleanup**
  - [x] Verify mkdocs/griffe generators already type-check without suppressions; address remaining casts in DuckDB/parquet helpers.
  - [ ] Introduce typed facades in `docs/_scripts/shared.py` for: (a) loading configuration (`DocGenSettings` dataclass), (b) safe DuckDB queries using parameterised paths and `pathlib`, (c) JSON serialization with explicit schemas. Include docstrings, `__all__`, and doctest snippets for each helper (principles 1, 2, 7, 10, 16).
  - [ ] Refactor `docs/_scripts/build_symbol_index.py`, `docs/_scripts/symbol_delta.py`, `docs/_scripts/models.py`, and `docs/_scripts/validation.py` to consume the facades, emit structured logs, and remove any ad-hoc `sys.path` manipulations; ensure all scripts are import-safe and idempotent (principles 5, 7, 15).
  - [ ] Ensure CLI entry points (including `site/_scripts/mkdocs_gen_api.py`) expose typed `main(argv: Sequence[str] | None) -> int` functions with full docstrings and Problem Details references for error cases; confirm via `uv run pyrefly check docs/_scripts site/_scripts` and `uv run mypy --config-file mypy.ini docs/_scripts site/_scripts` (principles 1, 4, 5).
  - [ ] Update developer documentation explaining how to extend doc tooling safely (typed interfaces, no raw DuckDB SQL, use of `pathlib`, security guards against path traversal); cross-link from `openspec/AGENTS.md` (principles 10, 13, 14).
- [ ] **1.6 Governance**
  - [ ] Implement `scripts/check_pyrefly_suppressions.py` that scans for `pyrefly: ignore` or `# type: ignore` pragmas and fails unless the line includes a ticket reference or documented rationale. Hook it into pre-commit and CI (`lint` stage) ensuring the command prints actionable remediation guidance (principles 4, 14).
  - [ ] Update `openspec/AGENTS.md`, `docs/contributing/quality.md`, and the internal engineering handbook to document: (a) the new typed wrappers, (b) how to request temporary suppressions, (c) conventions for optional deps/stubs, and (d) required verification commands (ruff/pyrefly/mypy). Include code snippets that run as doctests (principles 1, 13).
  - [ ] Review CI workflows to guarantee the suppression check runs alongside `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini`; document the escalation path if a suppression is truly unavoidable (principles 4, 14).

## 2. Testing
- [ ] 2.1 Run `uv run ruff format && uv run ruff check --fix` on touched paths to ensure style compliance and import hygiene.
- [ ] 2.2 Run `uv run pyrefly check` scoped to each module cluster during refactor and globally before merge.
- [ ] 2.3 Run `uv run mypy --config-file mypy.ini` for affected packages (`tools`, `src/search_api`, `src/vectorstore_faiss`, `src/kgfoundry_common`).
- [ ] 2.4 Exercise optional dependency fallbacks by running `uv run pyrefly check`/`uv run mypy` in CPU-only environments and with `uv sync --extra gpu` where relevant.

## 3. Docs & Artifacts
- [ ] 3.1 Update developer documentation (`openspec/AGENTS.md`, `docs/contributing/quality.md`) describing typed wrappers and suppression policy.
- [ ] 3.2 Regenerate API docs and navmap artifacts (`make artifacts`) to confirm no drift.
- [ ] 3.3 `openspec validate pyrefly-suppression-bust --strict`.

## 4. Rollout
- [ ] 4.1 Communicate the suppression removal plan to affected teams (search, tooling, docs) with any required migration steps.
- [ ] 4.2 Monitor CI (pyrefly/mypy) for regressions after merge; open follow-up tickets for any third-party stub contributions upstream.

