## MODIFIED Requirements
### Requirement: Authoritative artifact models
The system SHALL represent every documentation artifact (symbol index, symbol delta,
reverse lookup tables, changelog manifests) with immutable Pydantic V2 models defined
under `docs/_types/artifacts.py`. Models MUST:

1. Declare `model_config = ConfigDict(frozen=True, ser_json_timedelta='iso8601')` or
   equivalent to prevent mutation and guarantee deterministic serialization.
2. Use fully annotated PEP 695 generics to capture metadata variants without
   resorting to `typing.Any`.
3. Expose typed constructor helpers (`from_payload`, `build`, `compute_delta`) that
   validate input data and surface schema metadata to callers.

### Requirement: Schema conformity with metadata contracts
Artifact models MUST serialize to payloads that validate against
`schema/docs/*.schema.json`, treat those schema documents as the source of truth, and
record both `schema_id` and `schema_version` in their serialized output. Round-trip
helpers SHALL verify checksum fields and reject mismatched schema versions.

### Requirement: Typed loader facades
Integrations with Griffe and optional Sphinx components SHALL operate through typed
facades under `docs/_types/griffe.py` and associated stubs. Public APIs MUST NOT
accept or return `typing.Any`; variadic arguments SHALL be typed via explicit
overloads that mirror runtime behavior while preserving strict type checking.

### Requirement: Problem Details and exception taxonomy
Artifact validation errors MUST raise from a dedicated `ArtifactModelError`
hierarchy housed in `kgfoundry_common.errors`. Each error SHALL expose the artifact
name, schema metadata, and underlying cause, and it MUST be convertible to RFC 9457
Problem Details via `kgfoundry_common.problem_details`. A canonical error payload
SHALL live at `schema/examples/problem_details/docs-artifact-validation.json`.

### Requirement: Type-clean quality gates
Artifact models, loaders, stubs, and docs scripts SHALL compile under Ruff, Pyright,
Pyrefly, and MyPy without suppressions. Violations such as `ANN401`, `EM101/102`,
`TRY003`, and tuple indexing errors MUST be structurally eliminated rather than
ignored.

### Requirement: Execution tracking
The change record SHALL enumerate implementation, testing, docs, and rollout tasks
with checkbox tracking so reviewers can audit that every gate (schema updates, test
coverage, artifact regeneration) completed before archive.

### Requirement: Regression coverage
Table-driven pytest suites SHALL cover round-trips, checksum mismatches, delta
classification, and optional dependency fallbacks. Tests MUST assert both success
and failure Problem Details payloads, including schema metadata fields and
correlation IDs. Doctest/xdoctest examples SHALL run as part of pytest.

### Requirement: Documentation and migration guidance
Contributor documentation SHALL describe the hardened artifact models, schema
metadata requirements, exception taxonomy, and testing strategy. Docs MUST include a
copy-ready example demonstrating model construction, serialization, and error
handling.

## ADDED Scenarios
#### Scenario: Schema metadata enforced
- **GIVEN** a legacy payload missing `schema_id`
- **WHEN** `ArtifactSymbolIndex.from_payload` executes
- **THEN** it raises `ArtifactValidationError` containing the schema namespace,
  Problem Details payload defined at
  `schema/examples/problem_details/docs-artifact-validation.json`, and preserves the
  original exception via `raise ... from e`

#### Scenario: Round-trip fidelity with checksum
- **GIVEN** a valid artifact payload and checksum
- **WHEN** it round-trips through `model_to_payload` and `from_payload`
- **THEN** the payload matches byte-for-byte, the checksum remains unchanged, and
  schema version metadata is verified in logs and test assertions

#### Scenario: Typed loader integration
- **GIVEN** the docs build pipeline running without Sphinx optional dependencies
- **WHEN** the typed loader facade attempts to load them
- **THEN** it raises a descriptive `ArtifactDependencyError` (subclass of
  `ArtifactModelError`) that passes Pyright/MyPy checks without requiring
  `typing.Any`

#### Scenario: Regression tests enforce Problem Details
- **GIVEN** pytest parametrized cases for missing fields, checksum mismatch, and
  invalid enum values
- **WHEN** the tests run
- **THEN** each failure path asserts the emitted Problem Details payload matches the
  canonical JSON example and includes traceable correlation IDs

#### Scenario: Documentation example remains runnable
- **GIVEN** doctest execution over `docs/_types/artifacts.py`
- **WHEN** the embedded example constructs an artifact model, serializes it, and
  handles a validation failure
- **THEN** doctest passes without additional setup and demonstrates the recommended
  Problem Details handling workflow

