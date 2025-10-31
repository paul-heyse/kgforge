# Lint & Type Hardening Change

This change formalizes the plan to eliminate the outstanding Ruff and mypy
violations across `src/**` (excluding the independent docstring/formatting sweep
and governance/tooling work tracked elsewhere).

```
openspec/changes/lint-type-hardening/
├── proposal.md
├── tasks.md
├── design.md
└── specs/
    └── quality-hardening/
        └── spec.md
```

Run validation before requesting review:

```bash
openspec validate lint-type-hardening --strict
```

## Phased Execution at a Glance
- **Phase 1 · Foundation:** build shared infrastructure before touching call sites — pathlib helpers/codemods (R1), exception taxonomy & Problem Details registry (R2), secure serialization helpers (R3), schema/model scaffolding (R4), vector search protocols (R5), runtime settings (R7), and structured logging/metrics envelope (R8). Each lands with unit tests and zero downstream churn.
- **Phase 2 · Adoption:** migrate product areas (search adapters → registry → embeddings → orchestration) onto the new helpers while executing requirements R6–R17. Use the per-requirement checklist in `tasks.md`, running acceptance gates after each requirement to prevent drift.

## Clean Baseline Definition
A change is “clean” when:
- `uv run ruff format && uv run ruff check --fix` is green with **no new suppressions** and rule families `D, ANN, S, PTH, TRY, ARG, DTZ` pass.
- `uv run pyrefly check`, `uv run mypy src`, and doctests (`pytest --doctest-modules`) succeed without ignores.
- All schemas validate against the JSON Schema 2020-12 meta-schema and the OpenAPI doc passes `spectral lint`.
- The canonical Problem Details example validates and is referenced in docs/tests.
- Import-linter contracts and the “no new suppressions” script pass.

## Quickstart for Junior Developers
- Read files in order: `proposal.md` → `design.md` (Implementation Blueprint, phase plan, Testing Matrix, Verification callouts) → `specs/quality-hardening/spec.md` → `tasks.md`.
- The plan tracks seventeen requirements (R1–R17). Phase 1 tasks focus on shared infrastructure; Phase 2 tasks migrate each surface. Each checkbox lists specific files, tests, codemods, and verification commands.
- Maintain the core principles: PEP 257 docstrings (first line imperative), fully typed public APIs, schema-first contracts, structured logging/Problem Details with stable codes, and zero lint/type ignores without ticket links.
- After each requirement, run the acceptance commands listed in `tasks.md` and paste outputs (including codemod logs, import-linter results, and suppression checks) into your PR.
- If you encounter missing third-party types, add `.pyi` stubs under `stubs/` rather than introducing `Any`.
- Coordinate with docstring/governance workstreams when editing overlapping files, and flag scope deviations early so the proposal/design can be updated.


