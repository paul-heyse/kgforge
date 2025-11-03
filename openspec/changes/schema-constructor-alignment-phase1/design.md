## Context
- Docs, navmap, and docstring-builder pipelines already lean on Pydantic V2 models, yet
  several entry points still construct models with snake_case keywords. This bypasses the
  schema metadata guarantees, leads to inconsistent Problem Details emissions, and keeps
  legacy payload behaviour implicit.
- Schema field names are canonicalized in JSON Schema 2020-12 documents (`schemaVersion`,
  `schemaId`, `deprecatedIn`, `policyVersion`, etc.). Divergence causes Pyright/Mypy
  keyword errors and requires manual patching in tests.
- Migration logic is fragmented: some scripts silently rename keys, others accept mixed
  casing without logging, making observability and regression testing difficult.

## Goals / Non-Goals
- **Goals**
  1. Guarantee every constructor call path enforces canonical schema casing and surfaces
     Problem Details when payloads deviate.
  2. Provide reusable migration utilities that normalize legacy payloads, emit structured
     warnings, and expose metrics for observability.
  3. Tighten regression coverage through parametrized pytest cases, doctests, and
     round-trip assertions that confirm schema fidelity and checksum stability.
  4. Document the constructor contract and migration workflow so contributors have
     copy-ready, runnable examples.
- **Non-Goals**
  - Rewriting schema definitions or adding new artifact formats.
  - Altering plugin registry/griffe stubs (handled separately).
  - Introducing asynchronous pipelines or new telemetry exports.

## Decisions
- Create `align_schema_fields` utilities housed in `docs/_types/alignment.py` (exact path
  TBD) that accept mappings, normalize casing, and raise `ArtifactValidationError` with
  RFC 9457 Problem Details when unknown keys are encountered.
- Use PEP 695 generics to type the migration helpers so return types stay linked to the
  target Pydantic model (`TypeVar[ModelT, BaseModel, BaseModel]`).
- Centralize schema metadata constants (e.g., valid field sets per artifact) to avoid
  magic strings inside scripts; leverage `Literal` or `enum.StrEnum` where appropriate.
- Extend doctest examples to include both canonical constructor usage and legacy
  migration, ensuring examples execute via `pytest --doctest-modules`.
- Update pytest suites with table-driven cases covering valid canonical payloads, legacy
  casing, missing keys, and extra keys; enforce byte-level equality after round-trips.

## Alternatives Considered
- **Ad-hoc casing fixes per script** — rejected because it invites drift and duplicates
  logic.
- **Generating constructors from schemas** — deferred pending evaluation of tooling
  complexity and maintenance overhead.
- **Permitting mixed casing indefinitely** — rejected; violates design standards and keeps
  type checkers noisy.

## Risks & Mitigations
- **Regression risk across pipelines** — mitigated via comprehensive test coverage, CI
  quality gates, and incremental rollout per module.
- **Warning fatigue** — mitigated by using the observability helpers to emit structured
  warnings once per distinct payload signature and exposing counters for monitoring.
- **Helper indirection** — mitigated by thorough documentation, type hints, and doctest
  examples illustrating usage.

## Migration Plan
1. Inventory constructor usage with tooling (ripgrep + targeted audits) and record
   affected modules.
2. Implement alignment helpers, schema metadata constants, and Problem Details error types.
3. Update constructors across docs scripts, navmap models, and docstring-builder modules to
   call the helper before instantiation.
4. Wire migration helpers into ingestion points likely to receive legacy payloads;
   introduce structured logging/metrics for migrations performed.
5. Refresh fixtures, doctests, and pytest suites to exercise canonical + legacy flows,
   including schema round-trips and checksum validation.
6. Execute the quality-gate loop and regenerate artifacts/documentation before submitting
   the change.

