## ADDED Requirements
### Requirement: Typed Vector Batch Contracts
The system SHALL expose a shared `kgfoundry_common.vector_types` module that defines immutable vector batch contracts (`VectorId`, `VectorMatrix`, `VectorBatch`) with runtime validation helpers and public docstrings that conform to the NumPy style guide.

#### Scenario: Type checkers accept vector helpers
- **GIVEN** the repository with the new `kgfoundry_common.vector_types` module and corresponding stub exports
- **WHEN** `uv run pyrefly check` and `uv run mypy --config-file mypy.ini` execute over `kgfoundry_common`, `search_api`, and `orchestration`
- **THEN** neither tool reports `Any`-propagating annotations nor requires `# type: ignore` directives for vector ingestion boundaries

#### Scenario: Runtime validation enforces dtype and shape
- **GIVEN** callers construct `VectorBatch` instances through `coerce_vector_batch()` or `validate_vector_batch()`
- **WHEN** the helper receives vectors with inconsistent lengths, non-float inputs, or dtypes other than `float32`
- **THEN** it raises a typed `VectorValidationError` that preserves the failing index and reason, and no numpy `Any` escapes to downstream consumers

### Requirement: Vector Payload Schema Alignment
The system SHALL publish a canonical JSON Schema `schema/vector-ingestion/vector-batch.v1.schema.json` describing the vector ingestion payload, and runtime loaders SHALL validate payloads against this schema before producing typed batches.

#### Scenario: Schema validates canonical payloads
- **GIVEN** the schema stored at `schema/vector-ingestion/vector-batch.v1.schema.json`
- **WHEN** regression tests validate representative payloads via `jsonschema.Draft202012Validator`
- **THEN** well-formed payloads that satisfy the documented dimension bounds pass without warnings, and malformed payloads (missing `key`, ragged vectors, empty arrays) trigger `jsonschema.ValidationError`

#### Scenario: Loader enforces schema before casting
- **GIVEN** `_load_vectors_from_json()` in `orchestration.cli` and `FaissVectorstoreFactory.build_from_payload()`
- **WHEN** they parse JSON input
- **THEN** they validate against the canonical schema, emit structured `ProblemDetails` on failure, and return a `VectorBatch` whose dtype is guaranteed to be `float32` and whose matrix shape matches the ids count

### Requirement: Structured Boundary Errors
The system SHALL translate ingestion failures into RFC 9457 Problem Details envelopes using `kgfoundry_common.problem_details`, including remediation guidance and correlation identifiers, while logging structured context and preserving exception chains.

#### Scenario: Problem Details emitted on validation failure
- **GIVEN** a vector file containing non-numeric entries or inconsistent lengths
- **WHEN** `_load_vectors_from_json()` encounters the invalid record
- **THEN** it logs the offending `key` and index, raises `IndexBuildError` chained from `VectorValidationError`, and renders a Problem Details payload stored under `schema/examples/problem_details/vector-ingestion-invalid-vector.json`

#### Scenario: CLI reports actionable remediation hints
- **GIVEN** `kgfoundry orchestration cli index_faiss` processes a malformed payload
- **WHEN** the command exits with failure
- **THEN** STDERR includes the Problem Details JSON with `type`, `title`, `detail`, `instance`, `remediation`, and `invalidVectors` fields so operators can correct the dataset without inspecting stack traces

### Requirement: Regression Coverage for Typed Vector Ingestion
The system SHALL maintain regression tests that exercise vector ingestion across happy path, edge cases, and failure scenarios, ensuring alignment between runtime helpers, schema validation, and documentation.

#### Scenario: Table-driven batch tests pass
- **GIVEN** parametrized tests under `tests/vector_ingestion/test_vector_types.py`
- **WHEN** they feed coercion helpers lists, tuples, numpy arrays, and memoryviews of numeric data
- **THEN** tests assert the returned `VectorBatch` enforces dtype/shape invariants, and mismatched row lengths raise `VectorValidationError`

#### Scenario: CLI integration test covers Problem Details
- **GIVEN** a Typer `CliRunner` test invoking `index_faiss` with both valid and invalid payloads
- **WHEN** the invalid payload run fails
- **THEN** the captured output matches the canonical Problem Details example, metrics counters remain unchanged for successes, and structured logs include `operation="vector_ingestion"` with correlation IDs


