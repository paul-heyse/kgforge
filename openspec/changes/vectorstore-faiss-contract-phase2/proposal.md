## Why
FAISS orchestration currently fails static analysis: Ruff flags 137 violations across vectorstore and CLI modules, while pyrefly and mypy report missing attributes (`FloatArray`, `VecArray`) and constructor mismatches in the namespace bridges. The `index_faiss` CLI calls methods that do not exist on the runtime adapter, leaving GPU/CPU parity untested and observability gaps (no metrics, unstructured logs, undocumented timeouts). Without a hardened contract the CLI cannot guarantee idempotent, typed index builds or deliver reliable telemetry.

## What Changes
- [x] **ADDED**: `vectorstore/faiss` capability spec codifying typed aliases, factory orchestration guarantees, observability requirements, and regression coverage expectations.
- [x] **MODIFIED**: `vectorstore_faiss` and `search_api.faiss_adapter` runtimes plus their `.pyi` stubs so `FloatArray`, `IntArray`, `StrArray`, and `VecArray` are first-class, fully documented exports with matching signatures across namespace bridges.
- [x] **MODIFIED**: `orchestration/cli.py:index_faiss` now consumes a dependency-injected `FaissVectorstoreFactory`, enforces explicit timeouts/idempotency, and emits structured logs with correlation IDs, accelerator labels, and status fields.
- [x] **ADDED**: Schema-first persistence layer (`schema/vectorstore/faiss-index-manifest.schema.json`) alongside runtime emit/validate helpers and RFC 9457 Problem Details example (`schema/examples/problem_details/faiss-index-build-timeout.json`).
- [x] **ADDED**: Comprehensive regression suite covering CPU/GPU parity, timeout and persistence failures, manifest round-trips, structured logging, and Prometheus counter/histogram instrumentation.
- [ ] **REMOVED**: Legacy helpers and unchecked pickle flows superseded by the factory, schema, and safe serialization wrappers.

## Impact
- **Capability surface:** new `vectorstore/faiss` spec governing typed aliases, factories, observability, and regression requirements.
- **Runtime/stub scope:** `src/vectorstore_faiss/gpu.py`, `src/kgfoundry/vectorstore_faiss/gpu.py`, `src/search_api/faiss_adapter.py`, `src/kgfoundry/search_api/faiss_adapter.py`, plus their corresponding stubs under `stubs/vectorstore_faiss` and `stubs/search_api`.
- **Orchestration & common libs:** `src/orchestration/cli.py`, `src/kgfoundry_common/logging.py`, and `src/kgfoundry_common/prometheus.py` for logging adapters and metric helpers; additional helpers may live under a new `src/search_api/vectorstore_factory.py` module.
- **Schemas & artifacts:** `schema/vectorstore/faiss-index-manifest.schema.json`, `schema/examples/problem_details/faiss-index-build-timeout.json`, and any downstream docs/artifacts that reference the manifest or error payloads.
- **Operational rollout:** ship as a single change guarded by configuration toggles (accelerator selection, timeout budgets). Validate CPU-only and GPU-enabled environments in staging before enabling observability dashboards; publish migration notes detailing schema versions and new env vars.

- [ ] Ruff (`uv run ruff format && uv run ruff check --fix`) reports zero violations across `vectorstore_faiss`, `search_api`, and `orchestration` modules.
- [ ] Pyrefly and mypy complete cleanly for the affected packages with no new suppressions; namespace bridges expose resolved aliases to both analyzers.
- [ ] `pytest -q` executes the parametrized regression suite, demonstrating CPU/GPU parity, timeout/error handling, manifest validation, metrics emission, and Problem Details output.
- [ ] Manifest schema validates against the 2020-12 meta-schema and Problem Details example renders via CLI/factory errors; docstrings include runnable examples documenting timeouts and idempotency.
- [ ] Prometheus counters (`kgfoundry_index_build_total`) and histograms (`kgfoundry_index_build_duration_seconds`) surface in smoke tests (`uv run python -m tests.observability.test_prometheus --faiss-index`) with expected labels populated.

## Out of Scope
- Replacing FAISS with an alternative vector database or changing BM25 indexing flows
- Shipping new deployment tooling for GPUs beyond documenting required drivers
- Refactoring unrelated Ruff violations outside the vectorstore/search/orchestration surface

## Risks / Mitigations
- **Risk:** GPU-dependent tests may flake on environments lacking drivers.  
  **Mitigation:** provide CPU fallbacks and mark GPU tests with `@pytest.mark.gpu`, skipping when hardware unavailable.
- **Risk:** Schema changes could break legacy index consumers.  
  **Mitigation:** version the manifest schema, provide conversion helpers, and validate backward compatibility in regression tests.
- **Risk:** Structured logging and metrics could add overhead to index builds.  
  **Mitigation:** emit logs at INFO with bounded payloads, use async-safe Prometheus client, and benchmark for regression budget.

## Alternatives Considered
- Reinstating broad `# type: ignore` directives on namespace bridges — rejected because it masks real API drift and keeps pyrefly/mypy red.
- Embedding FAISS calls directly in the CLI rather than factories — rejected; factories enable testing, DI, and observability injection per design principles.
- Treating CLI observability as a separate phase — rejected to ensure telemetry, schema, and type safety land together, preventing future divergence.

