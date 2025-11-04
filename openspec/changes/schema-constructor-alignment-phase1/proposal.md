## Why
Our documentation and tooling pipelines now rely on frozen Pydantic V2 models, yet many
callers still invoke `model_construct`, `model_validate`, or direct constructors with
legacy snake_case keywords (`schema_version`, `schema_id`, `deprecated_in`). The drift
creates schema mismatches, blocks deterministic round-trips, and erodes the Problem
Details guarantees required by `AGENTS.md`. A focused alignment initiative is necessary to
restore schema fidelity, guide legacy payload migrations, and keep doctest-backed
examples authoritative.

## What Changes
- [ ] **Schema-aligned constructors** — Update every constructor invocation across
  `docs/_types`, docs build scripts, navmap helpers, and docstring-builder pipelines to use
  canonical JSON Schema casing (`schemaVersion`, `schemaId`, `deprecatedIn`, etc.).
- [ ] **Migration helpers** — Provide centralized utilities that normalize legacy casing,
  emit structured deprecation warnings, and surface RFC 9457 Problem Details when invalid
  fields are supplied.
- [ ] **Regression coverage** — Expand table-driven pytest suites and doctests to cover
  canonical constructors, legacy migration paths, and schema round-trips.
- [ ] **Documentation refresh** — Update contributor guides and inline docstrings so all
  runnable examples demonstrate the aligned constructor workflow and observability hooks.

## Impact
- **Capability:** `docs-toolchain`
- **Code paths:** `docs/_types/artifacts.py`, `docs/_scripts/*.py`,
  `tools/navmap/document_models.py`, `tools/docstring_builder/**/*.py`, associated tests
  and doctests.
- **Contracts:** JSON Schemas remain the source of truth; alignment enforces canonical
  field names and Problem Details semantics for validation errors.
- **Delivery:** Implement on branch `openspec/schema-constructor-alignment-phase1` with the
  full quality-gate loop (`ruff`, `pyright`, `pyrefly`, `pyright`, `pytest`, `make artifacts`).

## Acceptance
- [ ] All constructor invocations use canonical schema field names; legacy casing is only
  accepted through the migration helper and emits structured warnings.
- [ ] Schema round-trip tests and doctests verify byte-for-byte payload fidelity, checksum
  stability, and observability metadata.
- [ ] Ruff, Pyright, Pyrefly, and MyPy report zero errors for touched modules without
  suppressions.
- [ ] Documentation and Problem Details examples demonstrate the aligned constructor
  workflow, including remediation guidance for invalid payloads.

## Out of Scope
- Changing schema documents themselves or introducing new artifact formats.
- Refactoring plugin registry or Griffe stubs (handled by other changes).
- Building new telemetry backends beyond existing logging/metrics integrations.

## Risks / Mitigations
- **Legacy payload breakage** — Mitigated by the migration helper, structured warnings, and
  table-driven regression tests covering known legacy shapes.
- **Cross-pipeline regressions** — Mitigated by comprehensive pytest suites, doctests, and
  the mandatory quality-gate loop.
- **Documentation drift** — Mitigated by embedding runnable examples and verifying via
  `make artifacts` plus doctest execution.

