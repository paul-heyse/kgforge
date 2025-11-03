## Context
- The docs artifact pipeline currently mixes frozen Pydantic V2 models with
  placeholder helpers that still leak `Any`, untyped iterables, and ad-hoc error
  strings. Ruff flags `EM101/102` and `TRY003`, while Pyright and MyPy report
  tuple index errors and calls with outdated keyword names.
- Validation errors surface as string literals rather than structured Problem
  Details, leading to inconsistent CLI behavior and untestable failure paths.
- Loader integrations (Griffe, Sphinx optional facades) still rely on stubs with
  `*args: Any` and `**kwargs: Any`, defeating the strict type gates mandated in
  `AGENTS.md`.
- Regression coverage for artifact deltas is limited; the existing tests miss
  schema metadata assertions, Problem Details payload verification, and table-
  driven validation of change categories.

## Goals / Non-Goals
- **Goals**
  - Deliver immutable, fully annotated artifact models with schema metadata and
    helper factories that keep JSON schemas as the source of truth.
  - Introduce a dedicated exception taxonomy for artifact validation, mapped to
    RFC 9457 Problem Details with a canonical example payload.
  - Remove `Any` from documentation stubs and loaders, keeping Pyright, Pyrefly,
    MyPy, and Ruff clean without suppressions.
  - Expand regression tests (pytest + doctest) to cover round-trips, deltas,
    schema violations, and optional dependency fallbacks.
  - Document the design and migration guidance for docs maintainers, ensuring
    `make artifacts` remains deterministic.
- **Non-Goals**
  - Replacing JSON artifacts with alternative serialization formats.
  - Reworking the broader docs toolchain orchestration (tracked under
    `tools-architecture-hardening`).
  - Adding new external dependencies beyond typing aids or problem details
    helpers already in the repository.

## Decisions
- Use Pydantic V2 `BaseModel` with `ConfigDict(frozen=True)` for all artifact
  models, leveraging `typing.Self` and PEP 695 generics to express variant
  payloads (e.g., `SymbolField[T_Metadata]`).
- Expose a shared `ArtifactModelError` hierarchy under
  `kgfoundry_common.errors.artifacts` that encapsulates schema metadata and emits
  Problem Details via `kgfoundry_common.problem_details.to_problem_details`.
- Implement model construction helpers (`SymbolIndexModel.build`,
  `ArtifactDeltaModel.compute`) that accept validated inputs, log schema IDs, and
  return frozen models ready for serialization.
- Regenerate or author JSON schema examples ensuring the new
  `schema/examples/problem_details/docs-artifact-validation.json` demonstrates
  the enriched error payload.
- Replace `Any` arguments in `stubs/griffe/**/*.pyi` with concrete overloads and
  typed call signatures mirroring the runtime facades; add tests to prevent
  regressions.
- Expand pytest suites with `@pytest.mark.parametrize` tables covering round-
  trips, validation failures, delta classifications, and optional dependency
  fallbacks; integrate doctest snippets in public docstrings to keep examples
  runnable.

## Alternatives
- **Keep msgspec Structs** – Rejected; we already migrated to Pydantic V2 and
  reintroducing Structs would regress type checker support and docs.
- **Tolerate `Any` via targeted ignores** – Rejected; violates the zero-
  suppression policy and would perpetuate fragile interfaces.
- **Emit generic HTTP 400 errors** – Rejected; fails the Problem Details
  requirement and reduces observability.

## Risks / Trade-offs
- **Schema tightening may break local scripts** – Mitigated by providing upgrade
  notes, examples, and regression tests that codify the new expectations.
- **Stub maintenance overhead** – Mitigated by documenting upstream version
  ranges and preparing an upstream contribution where practical.
- **Increased test duration** – Mitigated by consolidating fixtures, using
  in-memory payloads, and skipping optional dependency scenarios when modules
  are absent.

## Migration
1. Introduce the new exception hierarchy in `kgfoundry_common.errors` with
   accompanying Problem Details helper and JSON example.
2. Refactor artifact models to use typed generics, schema metadata properties,
   and structured raise paths; update builders and serializers accordingly.
3. Rewrite loader/stub interfaces to eliminate `Any`, adding overload-based
   shims where runtime APIs remain dynamic.
4. Update docs scripts and tests to consume the hardened APIs, asserting schema
   IDs, version tags, and Problem Details payloads.
5. Expand pytest + doctest coverage, regenerate documentation artifacts, and run
   full quality gates before submitting the PR.

