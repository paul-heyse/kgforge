# Design

## Context

Phase 1 upgrades replaced msgspec structs with Pydantic models but left the FAISS vectorstore surface divergent. Namespace bridges (`kgfoundry.vectorstore_faiss`, `kgfoundry.search_api.faiss_adapter`) expose symbols that no longer exist (`FloatArray`, `VecArray`), triggering pyrefly/pyright failures. The orchestration CLI still instantiates a legacy `FaissGpuIndex` with nonexistent methods, so Ruff flags dead code and complexity while index builds rely on untyped dictionaries, ad-hoc logging, and no Prometheus metrics. GPU vs CPU parity is untested, persistence lacks a schema, and error handling cannot emit Problem Details payloads consistent with our taxonomy.

## Goals

- Restore a unified, typed FAISS contract that passes Ruff, pyrefly, and pyright without suppressions.
- Provide dependency-injected factories that build both CPU and GPU adapters while documenting timeout and idempotency semantics.
- Instrument CLI index builds with structured logging, Prometheus counters/histograms, and Problem Details error payloads.
- Define JSON Schema for persisted FAISS manifests and add regression tests covering CPU/GPU parity, failure modes, and round-trip persistence.

## Non-Goals

- Replacing FAISS with a different vector database or altering BM25 indexing flows.
- Shipping GPU driver management or container orchestration; we assume environments provide required acceleration libraries.
- Addressing unrelated Ruff violations outside vectorstore/search/orchestration modules.

## Decisions

1. **Typed array aliases and stubs:** Implement `FloatArray`, `IntArray`, `StrArray`, and `VecArray` type aliases backed by `numpy.typing.NDArray` and mirror them in runtime modules and `.pyi` stubs to satisfy bridge imports and static analyzers.
2. **Vectorstore factory abstraction:** Introduce `FaissVectorstoreFactory` (or similar) responsible for constructing `FaissAdapter` instances with explicit configuration objects (timeouts, GPU flags, metrics providers). The CLI acquires adapters exclusively through this factory to enable dependency injection in tests.
3. **Structured observability:** Standardise logging through `kgfoundry_common.logging` helpers, adding `operation`, `status`, `accelerator`, and correlation IDs. Instrument Prometheus counters (`kgfoundry_index_build_total`) and histograms (`kgfoundry_index_build_duration_seconds`) inside the factory to capture success/failure signals.
4. **Schema-first persistence:** Persist index metadata (paths, version, embedding dimension, checksum) alongside ID maps and register a JSON Schema under `schema/vectorstore/faiss-index-manifest.schema.json`. Provide helpers to emit/validate manifests and enforce verification during CLI saves.
5. **Problem Details integration:** Define a canonical Problem Details example (`schema/examples/problem_details/faiss-index-build-timeout.json`) and raise `IndexBuildError`/`VectorSearchError` with `raise ... from e` semantics referencing the schema. CLI error handling renders the payload for human-readable output.
6. **Parametrized regression tests:** Add pytest suites that parameterize accelerator modes, simulate persistence failures, and assert schema validation plus metrics increments. GPU-specific tests skip gracefully when hardware is unavailable.

## Detailed Plan

### 1. Type surface remediation

1. Inventory the current exports in `src/vectorstore_faiss/gpu.py`, `src/kgfoundry/vectorstore_faiss/gpu.py`, `src/search_api/faiss_adapter.py`, and `src/search_api/faiss_gpu.py`, capturing every pyrefly/pyright error linked to missing aliases.
2. Implement `FloatArray`, `IntArray`, `StrArray`, and `VecArray` in `src/vectorstore_faiss/gpu.py` using `numpy.typing.NDArray`, add NumPy-style docstrings, and publish them via `__all__`.
3. Synchronise namespace bridges by updating `src/kgfoundry/vectorstore_faiss/gpu.py` (`__all__`, docstrings, `namespace_attach`) and confirming Ruff compliance.
4. Remove the blanket `# pyright: ignore-errors` from `src/search_api/faiss_adapter.py`, replace raw `NDArray` annotations with the typed aliases, and add precise return/exception annotations across helper methods.
5. Refresh protocol definitions in `src/search_api/types.py` so method signatures use the new aliases and naming satisfies Ruff’s PEP 8 checks.
6. Align stub packages (`stubs/vectorstore_faiss/gpu.pyi`, `stubs/search_api/faiss_adapter.pyi`) with the runtime exports, ensuring docstrings and `__all__` mirror implementation state.

### 2. Factory and CLI integration

1. Introduce `FaissAdapterSettings`/`FaissBuildContext` data models in a new `src/search_api/vectorstore_factory.py`, capturing paths, factory string, metric, nprobe, accelerator mode, timeout, and optional Prometheus registry.
2. Implement `FaissVectorstoreFactory` with operations `build_index`, `load_or_build`, `save_manifest`, and `close`, delegating GPU detection to `vectorstore_faiss.gpu.FaissGpuIndex` and raising typed exceptions with preserved causes.
3. Refactor `orchestration/cli.py:index_faiss` to construct settings from CLI options, request adapters from the factory, and wrap execution in correlation-aware logging contexts.
4. Enforce timeout/idempotency semantics via `time.monotonic()` checks and atomic file writes (`Path.replace`), documenting the behaviour in CLI docstrings and help text.

### 3. Observability wiring

1. Update `kgfoundry_common.logging.LoggerAdapter` usage (or extend it if needed) to emit structured fields: `operation`, `accelerator`, `status`, `index_path`, and correlation IDs.
2. Register `kgfoundry_index_build_total` (Counter) and `kgfoundry_index_build_duration_seconds` (Histogram) in `kgfoundry_common.prometheus`, allowing dependency injection of registries for deterministic tests.
3. Emit metrics at factory boundaries for start, success, timeout, and error paths; ensure labels cover backend, accelerator, and status.
4. Capture observability guidance in docstrings/examples so operators understand expected telemetry.

### 4. Schema & error payloads

1. Author `schema/vectorstore/faiss-index-manifest.schema.json` (draft 2020-12) listing required metadata (`version`, `factory`, `metric`, `embedding_dimension`, `accelerator`, `index_uri`, `ids_uri`, `checksum`, `created_at`).
2. Implement runtime helpers (`write_manifest`, `validate_manifest`) that emit/verify payloads against the schema and compute deterministic checksum fields.
3. Produce `schema/examples/problem_details/faiss-index-build-timeout.json` and helper code that generates the RFC 9457 payload with retry hints, correlation ID, and documentation URL.
4. Update CLI/factory error handling to log exceptions with `exc_info=True`, print the Problem Details JSON, and signal failure via `typer.Exit(1)`.

### 5. Regression tests

1. Build shared fixtures (`tests/vectorstore/conftest.py`) for normalized vectors, temporary directories, mocked GPU contexts, Prometheus registries, and deterministic correlation IDs.
2. Create parametrized CPU/GPU parity tests validating top-k results and score parity, skipping GPU paths gracefully when hardware or bindings are unavailable.
3. Add CLI integration tests using `typer.testing.CliRunner` to assert structured logging, Problem Details emission, and exit codes across success and failure flows.
4. Simulate timeout, IO, and GPU cloning failures, validating metrics increments, fallback behaviour, and exception chains.
5. Validate manifest round-trips by reading saved payloads, running `jsonschema` validation, and reloading the index through the factory.
6. Enable doctest/xdoctest for updated modules to ensure documentation examples execute successfully.

### 6. Quality gates & validation

1. After implementation and test updates, run `uv run ruff format && uv run ruff check --fix` scoped to touched paths.
2. Execute `uv run pyrefly check` and `uv run pyright --warnings --pythonversion=3.13` for `src/vectorstore_faiss`, `src/search_api`, and `src/orchestration`, confirming no suppressions are necessary.
3. Run `pytest -q` (with GPU tests auto-skipping when unsupported) alongside `uv run python -m tests.observability.test_prometheus --faiss-index` to capture telemetry coverage.
4. Regenerate docs/artifacts only if schema examples require publication, verifying `make artifacts && git diff --exit-code` remains clean.
5. Document command outputs and observability screenshots/logs in the PR checklist to evidence compliance.

## Risks & Mitigations

- **GPU dependency drift:** CI environments may lack GPUs, so GPU tests must skip gracefully while still exercising CPU fallbacks; integration tests on GPU-enabled runners validate the full path.
- **Schema incompatibility:** Legacy manifests may not match the new schema. Provide conversion utilities with explicit Problem Details errors and document upgrade steps.
- **Observability overhead:** Additional metrics/logging could impact performance. Measure durations before/after instrumentation and keep logging at INFO with structured `extra` fields only.

## Migration

1. Implement type surface and stub fixes; verify pyrefly/pyright pass.
2. Introduce factory abstraction and refactor CLI; add observability instrumentation.
3. Define schema, Problem Details sample, and integrate with persistence helpers.
4. Add regression tests and ensure they pass locally (CPU) and on GPU-enabled staging.
5. Update docs, run `make artifacts`, and validate the change via `openspec validate`.
6. Publish migration notes covering new env vars, schema version, and metrics names.
