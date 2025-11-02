# Design

## Context

Vector ingestion currently lives at the orchestration and search API boundaries. The CLI ingests JSON skeleton files and forwards numpy arrays to the FAISS factory, but the flow relies on bare `NDArray[Any, dtype[Any]]` annotations and unchecked `np.asarray` calls. This bleeds `Any` into downstream modules, forces `# type: ignore` suppressions, and masks malformed datasets (ragged rows, float64 payloads, empty vectors). Operators have no schema to validate inbound payloads, and ingestion failures surface as generic `ValueError` messages without remediation hints or Problem Details. Previous hardening phases focused on factories and namespace stubs; this change addresses the missing ingestion contract.

## Goals

- Establish a shared, immutable vector contract module with validation helpers that eliminate `Any` propagation across numpy arrays.
- Introduce schema-first validation and structured Problem Details for ingestion errors, aligning CLI and factory behaviour.
- Provide deterministic regression coverage (unit + integration) for coercion helpers, schema enforcement, and CLI failure outputs.
- Document the contract, examples, and migration guidance so downstream teams adopt the typed interfaces.

## Non-Goals

- Replacing FAISS adapters, persistence formats, or search result pipelines.
- Refactoring unrelated numpy usage elsewhere in the codebase (e.g., embedding generation).
- Delivering GPU-specific optimisations or streaming ingestion workflows.

## Decisions

1. **Shared contract module**: Create `kgfoundry_common.vector_types` as the single source of truth for vector ids, matrices, batches, and validation utilities. This package already mediates cross-cutting types and is allowed through import boundaries.
2. **Immutable dataclasses**: Represent vector batches with a frozen dataclass (`VectorBatch`) that stores `ids: tuple[str, ...]` and `matrix: VectorMatrix`, ensuring downstream consumers cannot mutate structures without deliberate copies.
3. **Explicit casting after validation**: Replace broad `# type: ignore` with runtime validation + `typing.cast` in helper functions, keeping type checkers satisfied while guaranteeing dtype/shape at runtime.
4. **Schema-first ingestion**: Publish `schema/vector-ingestion/vector-batch.v1.schema.json` (Draft 2020-12). Orchestration and factory loaders must validate payloads against the schema prior to constructing numpy arrays.
5. **Structured error reporting**: Standardise on `VectorValidationError` (new exception) and wrap ingestion failures in `IndexBuildError` with Problem Details payloads stored under `schema/examples/problem_details/vector-ingestion-invalid-vector.json`.
6. **Layered testing**: Add table-driven unit tests for validation helpers, schema round-trips, and Typer CLI integration tests, ensuring GPU-dependent paths skip gracefully but still validate vector ingestion semantics.

## Detailed Plan

### 1. Vector contract module

1. Implement `src/kgfoundry_common/vector_types.py` (and matching stub) exposing:
   - `VectorId = NewType("VectorId", str)`
   - `VectorDimension = NewType("VectorDimension", int)` for clarity
   - `VectorMatrix = npt.NDArray[np.float32]` (TypeAlias)
   - `@dataclass(frozen=True)` `VectorBatch(ids: tuple[VectorId, ...], matrix: VectorMatrix, dimension: VectorDimension)`
   - Validation helpers `coerce_vector_batch(obj: Iterable[Mapping[str, object]]) -> VectorBatch`, `validate_vector_batch(batch: VectorBatch) -> VectorBatch`, `assert_vector_matrix(obj: object) -> VectorMatrix`
   - Custom exception `class VectorValidationError(ValueError)` with failure metadata
   - NumPy-style docstrings + doctest examples that mirror schema examples.
2. Replace ad-hoc type hints in `orchestration.cli` and `search_api.vectorstore_factory` with imports from the shared module. Remove all `# type: ignore` entries related to vector typing.
3. Update stub packages (`stubs/kgfoundry_common/vector_types.pyi`, `stubs/orchestration/cli.pyi`, `stubs/search_api/vectorstore_factory.pyi` if present) to match runtime exports and keep pyrefly/mypy consistent.

### 2. Schema & validation integration

1. Author `schema/vector-ingestion/vector-batch.v1.schema.json` specifying:
   - Array of objects with `key` (non-empty string), `vector` (array of numbers), optional metadata future fields.
   - Constraints on vector length (`minItems: 1`, `maxItems` based on existing FAISS dimension budgets), dataset size, and number type.
   - Examples aligned with doctests.
2. Update schema indices/docs to register the new schema (e.g., `schema/index.json` if maintained; ensure `make artifacts` generates updated catalog).
3. Extend `_load_vectors_from_json()` to:
   - Load JSON payload.
   - Validate via `jsonschema.Draft202012Validator` using the canonical schema (cache validator).
   - Convert to `VectorBatch` using new helpers; catch `jsonschema.ValidationError` and map to `VectorValidationError` with context.
4. Update `FaissVectorstoreFactory` to accept `VectorBatch` directly, verifying length parity (`len(batch.ids) == batch.matrix.shape[0]`) and logging `dimension`, `count`, and dtype.

### 3. Error handling & Problem Details

1. Create `VectorIngestionProblemBuilder` in `kgfoundry_common.problem_details` (or dedicated helper) that produces RFC 9457 payloads referencing the new example JSON.
2. Extend orchestration CLI error handling:
   - Catch `VectorValidationError` / `jsonschema.ValidationError`.
   - Build Problem Details with remediation suggestions (e.g., "Ensure vectors share the same dimension and contain numeric values").
   - Include `invalidVectors` array summarising offending ids/dimensions, limit size to avoid log overload.
   - Emit structured log with `operation="vector_ingestion"`, `status="error"`, correlation ID.
   - Exit via `typer.Exit(1)` after printing Problem Details to stderr.
3. Update telemetry counters/histograms (if already defined) to tag vector ingestion failures separately (e.g., `kgfoundry_index_build_total{stage="ingestion",status="error"}`).

### 4. Testing strategy

1. Add `tests/vector_ingestion/conftest.py` providing fixtures for sample payloads, schema validators, and correlation IDs.
2. Implement unit tests (`tests/vector_ingestion/test_vector_types.py`) covering:
   - Successful coercion from lists/tuples/numpy arrays.
   - Rejection of ragged vectors, empty vectors, non-numeric values.
   - TypeGuard behaviour (`assert_vector_matrix`) to satisfy mypy.
3. Add schema round-trip tests verifying sample payloads validate and invalid ones fail with descriptive messages.
4. Build Typer CLI integration tests ensuring both success path (valid dataset) and failure path (invalid dataset) produce expected outputs, exit codes, and logs (can assert using `caplog`).
5. Ensure doctest/xdoctest covers new NumPy-style examples in `vector_types` module.

### 5. Documentation & artifacts

1. Author `docs/reference/vector-ingestion.md` describing the contract, schema, Problem Details, and examples.
2. Link documentation to existing vectorstore guidance and ensure nav metadata references the new module (`__navmap__` updates).
3. Run `make artifacts` to regenerate schema indexes and doc navigation; ensure clean git diff.

### 6. Quality gates & validation

1. Run targeted `uv run ruff format && uv run ruff check --fix` for touched modules.
2. Execute `uv run pyrefly check` and `uv run mypy --config-file mypy.ini` to confirm zero suppressions.
3. Run new regression suite with `uv run pytest -q tests/vector_ingestion -q` (plus CLI integration). Document results for PR template.
4. Validate schema/meta with `jsonschema -i <example> schema/vector-ingestion/vector-batch.v1.schema.json` or project equivalent.
5. Update `openspec` validation via `openspec validate vector-ingestion-typed-arrays-phase1 --strict` prior to PR.

## Risks & Mitigations

- **Performance regressions**: Schema validation may introduce overhead on large datasets. Mitigate via validator caching, optional fast-path when payload already typed, and documenting performance budgets.
- **Breaking downstream scripts**: Third-party tooling built on old, loosely-typed helpers may fail. Provide migration notes, fallback compatibility shim, and coordinate rollout with consumers.
- **Telemetry noise**: Additional failure logs and Problem Details must avoid leaking sensitive payload data. Redact vector contents and cap emitted sample size.

## Migration

1. Implement shared vector contract module and replace references in orchestration + search API.
2. Integrate schema validation and Problem Details into ingestion flows.
3. Add regression tests, documentation, and artifacts.
4. Coordinate rollout by staging ingestion jobs with dry-run validation; update ops runbooks.
5. Deprecate any legacy helpers (documented warnings) and remove them in a follow-up phase after adoption metrics confirm success.


