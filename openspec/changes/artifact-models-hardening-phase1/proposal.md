## Why
The documentation artifact models accumulated ad-hoc error strings, `Any`-typed
constructors, and incomplete schema enforcement. These gaps surface across Ruff,
MyPy, Pyright, and Pyrefly, and they block the docs pipeline from guaranteeing
Problem Details parity or schema-backed regressions. We need a deliberate
hardening effort that re-establishes typed models, structured exception handling,
and table-driven validation before we tackle broader observability work.

## What Changes
- [ ] **MODIFIED**: `docs/_types/artifacts.py` reorganized around immutable
  Pydantic models with explicit generics, zero `Any` leakage, and helper
  factories that emit schema metadata (`schema_id`, `schema_version`).
- [ ] **ADDED**: Dedicated exception taxonomy (`ArtifactModelError` and
  subtypes) mapped to RFC 9457 Problem Details, including a canonical JSON
  example under `schema/examples/problem_details/docs-artifact-validation.json`.
- [ ] **MODIFIED**: Docs build scripts and tests updated to consume the new
  exception helpers, validate round-trips, and assert Problem Details payloads
  across success and failure paths.
- [ ] **ADDED**: Parametrized regression tests covering happy-path serialization,
  schema violations, loader fallbacks, and delta computations; doctest snippets
  embedded in public APIs demonstrating usage.
- [ ] **MODIFIED**: Griffe/Sphinx stubs rewritten to remove `Any` splats and
  align with the typed loader facades already consumed by the docs pipeline.

## Impact
- **Affected specs:** `docs-artifacts-type-safety`
- **Affected code:** `docs/_types/artifacts.py`, `docs/_scripts/*.py`,
  `tests/docs/test_artifact_models.py`, `tests/docs/test_symbol_delta.py`,
  `schema/examples/problem_details/**`, `stubs/griffe/**`
- **Data contracts:** Documentation artifact schemas in `schema/docs/**`
  (non-breaking but enforced); new Problem Details example for artifact
  validation.
- **Rollout:** Ship under feature branch `openspec/artifact-models-hardening-phase1`,
  run full quality gates locally before PR, and coordinate with docs owners to
  validate `make artifacts` output.

## Acceptance
- [ ] Artifact models expose fully annotated public APIs with frozen Pydantic
  models, zero `Any`, and docstrings meeting NumPy style.
- [ ] Validation helpers raise `ArtifactModelError` subclasses and emit the
  canonical Problem Details payload when schema constraints fail.
- [ ] Ruff, Pyright, Pyrefly, and MyPy run clean for artifact models, docs
  scripts, and stubs without suppressions.
- [ ] Table-driven pytest suites cover round-trips, delta computation, optional
  dependency fallbacks, and regression of schema violations.
- [ ] Docs build (`make artifacts`) completes with the hardened models and emits
  structured logs referencing schema/version metadata.

## Out of Scope
- Broader observability test expansion tracked in
  `testing-observability-phase6`.
- Rebuilding the docs site navigation or Agent Portal ingestion (verified as
  consumers only).
- Migrating artifact storage formats; JSON remains the source of truth.

## Risks / Mitigations
- **Risk:** Tightened validation could block builds on legacy payloads.
  - **Mitigation:** Provide migration helpers with explicit upgrade notes and
    guard rails in tests for known legacy shapes.
- **Risk:** Stub rewrites may drift from upstream `griffe` releases.
  - **Mitigation:** Capture version pin in design notes, document upstream diff,
    and upstream a PR once stable.
- **Risk:** New Problem Details schema may conflict with existing examples.
  - **Mitigation:** Coordinate with docs maintainers, add regression tests tied
    to the new example, and validate via `make artifacts`.

