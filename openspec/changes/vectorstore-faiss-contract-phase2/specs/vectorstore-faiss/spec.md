## ADDED Requirements
### Requirement: FAISS Vectorstore Type Contract
The system SHALL expose typed FAISS vectorstore surfaces whose runtime exports and stubs stay in sync, including `FloatArray`, `IntArray`, `StrArray`, `VecArray`, and constructor signatures used by namespace bridges.

#### Scenario: Namespace bridge satisfies type checkers
- **GIVEN** the repository with updated stubs under `stubs/vectorstore_faiss` and `stubs/search_api`
- **WHEN** `uv run pyrefly check` and `uv run pyright --warnings --pythonversion=3.13` execute
- **THEN** no missing-module-attribute or not-a-type errors are reported for `kgfoundry.vectorstore_faiss.gpu` or `kgfoundry.search_api.faiss_adapter`

#### Scenario: Stub contract mirrors runtime exports
- **GIVEN** an isolated type-checking environment that imports `kgfoundry.vectorstore_faiss.gpu` and `kgfoundry.search_api.faiss_adapter`
- **WHEN** the modules are inspected via `__all__` and `typing.get_type_hints`
- **THEN** the exported aliases and constructors match the runtime definitions, and public docstrings describe parameters per NumPy style without requiring `# type: ignore`

#### Scenario: Docstrings provide runnable examples
- **GIVEN** doctest/xdoctest runs across the FAISS modules
- **WHEN** the documented examples for alias usage and adapter construction execute
- **THEN** they complete successfully without additional setup beyond provided fixtures and demonstrate canonical usage paths

### Requirement: Observable FAISS Index CLI
The system SHALL build FAISS indexes via dependency-injected factories that enforce timeouts, document idempotency, emit structured logs, and expose Prometheus metrics for success and failure.

#### Scenario: CPU and GPU adapters share the factory
- **GIVEN** the CLI is invoked with configuration selecting `accelerator="cpu"` and `accelerator="gpu"`
- **WHEN** `orchestration.cli:index_faiss` runs
- **THEN** it acquires a `FaissAdapter` from a factory, logs start/stop entries with `operation="index_faiss"`, correlation ID, accelerator label, and increments `kgfoundry_index_build_total{backend="faiss",accelerator}` plus records duration in `kgfoundry_index_build_duration_seconds`

#### Scenario: Failure surfaces Problem Details
- **GIVEN** a persistence error while saving the index
- **WHEN** the CLI handles the exception
- **THEN** it emits a structured error log, increments `kgfoundry_index_build_total{status="error"}`, and renders the Problem Details payload stored at `schema/examples/problem_details/faiss-index-build-timeout.json`

#### Scenario: Timeout documented and enforced
- **GIVEN** a build that exceeds the configured timeout budget
- **WHEN** the factory detects the timeout via `time.monotonic()`
- **THEN** it raises `IndexBuildError` with preserved cause, logs `status="timeout"`, and the CLI docstring example illustrates the corresponding Problem Details output

### Requirement: Regression Coverage for FAISS Vectorstore
The system SHALL provide parametrized regression tests covering CPU vs GPU parity, failure modes, and round-trip persistence validated against a JSON Schema manifest.

#### Scenario: CPU vs GPU parity test passes
- **GIVEN** pytest parametrizes accelerator mode over `{"cpu", "gpu"}` with synthetic normalized vectors
- **WHEN** tests build indexes via the factory and execute searches
- **THEN** both modes return identical result sets within tolerance, and the test suite passes under `pytest -q`

#### Scenario: Persistence schema validates
- **GIVEN** the CLI saves an index manifest and ID map
- **WHEN** tests validate the manifest against `schema/vectorstore/faiss-index-manifest.schema.json`
- **THEN** the manifest round-trips through the helper API without schema validation errors, and static analysis remains clean (no Ruff `TRY` or `BLE` suppressions)

#### Scenario: Manifest corruption detected
- **GIVEN** a manifest with mismatched checksum or missing required fields
- **WHEN** `validate_manifest` runs during `load_or_build`
- **THEN** it raises `VectorSearchError` with `raise ... from e`, logs structured details, and regression tests assert the Problem Details payload references remediation guidance

