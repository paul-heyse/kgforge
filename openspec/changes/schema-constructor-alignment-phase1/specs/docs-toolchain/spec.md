## ADDED Requirements
### Requirement: Schema-aligned constructors
Documentation, navmap, and docstring-builder pipelines SHALL invoke Pydantic constructors
with keyword arguments that match the canonical JSON Schema field names. Legacy payloads
SHALL be normalized through migration helpers that emit structured warnings, Problem
Details payloads, and metrics capturing migration counts.

#### Scenario: Canonical constructor usage in docs pipeline
- **GIVEN** `docs/_scripts/build_symbol_index.py`
- **WHEN** it constructs artifact models during `make artifacts`
- **THEN** every invocation of `model_construct`/`model_validate` uses canonical casing
  (`schemaVersion`, `schemaId`, `deprecatedIn`, `policyVersion`, etc.), structured logs note
  `status="validated"`, and Pyright/MyPy detect no keyword mismatch errors

#### Scenario: Legacy payload migration with observability
- **GIVEN** a payload containing snake_case fields (`schema_version`, `schema_id`)
- **WHEN** the migration helper processes the payload
- **THEN** it normalizes keys to canonical casing, emits a single deprecation warning with
  correlation ID, increments a `schema_alignment_migrations_total` metric, and returns a
  model that passes schema round-trip validation

#### Scenario: Invalid key rejection produces Problem Details
- **GIVEN** a payload containing unknown fields (e.g., `schema_version_extra`)
- **WHEN** the migration helper executes
- **THEN** it raises `ArtifactValidationError` with an RFC 9457 Problem Details payload
  referencing the offending key, remediation guidance, and the relevant schema ID, and
  logs `status="error"`

#### Scenario: Doctest-backed example demonstrates workflow
- **GIVEN** doctest execution over the constructors documentation
- **WHEN** the example constructs a model using canonical casing, then passes a legacy
  payload through the migration helper
- **THEN** doctest passes without additional setup, verifying both success and migration
  behaviors and showing the emitted Problem Details structure

