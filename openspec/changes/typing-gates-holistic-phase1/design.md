# Design

## Context

Ruff, Pyright, and MyPy jointly report >700 violations tied to type-only imports executing at runtime. Modules across `docs/_scripts/`, `tools/`, and `src/` eagerly import heavy dependencies—FAISS, FastAPI, numpy, and project-internal bridges—purely to satisfy annotations. Because postponed annotations are inconsistently applied, cyclic imports persist, stubs drift from runtime modules, and tooling breaks when optional extras are missing. Multiple packages skirt the layering rules by importing from private namespaces (e.g., `docs._types`, `_cache`) to avoid circular dependencies, further entrenching the issue. We need a systemic, shared solution that establishes a canonical typing façade, enforces postponed annotations, and guards the contract via automated tooling.

## Goals
- Adopt postponed annotations (PEP 563 / PEP 649 behaviour) across all targeted modules to eliminate runtime evaluation of type hints.
- Centralise type-only imports behind sanctioned façades with helpers that expose safe re-exports for tooling and runtime packages alike.
- Enforce typing gates through Ruff configuration, custom lint tooling, and CI to prevent regressions.
- Demonstrate runtime determinism by executing key CLIs and docs scripts without optional dependencies installed.

## Non-Goals
- Altering domain business logic or introducing new external dependencies beyond lint/test tooling.
- Refactoring unrelated Ruff violations (e.g., complexity or async warnings) unless directly touched by typing changes.
- Migrating the entire repo to lazy imports for non-typing purposes; scope remains type-only hygiene.

## Decisions
1. **Project-wide postponed annotations:** Every Python module in scope SHALL enable postponed annotations via `from __future__ import annotations` until Python 3.13 becomes default, at which point we switch to module-level `__future__` removal where appropriate.
2. **Typing façade modules:** Introduce `kgfoundry_common.typing` and `docs.typing` packages that gather shared type aliases, protocols, and heavy imports, re-exporting them exclusively under TYPE_CHECKING blocks.
3. **TYPE_CHECKING utility helpers:** Provide helpers (`from kgfoundry_common.typing import gate_import`) that wrap imports, ensuring consistent patterning and easing lint enforcement.
4. **Ruff enforcement:** Configure Ruff to treat `TC00x`, `INP001`, and `PLC2701` as errors, while adding repository-specific rules that flag missing postponed annotations and direct private-module imports.
5. **CI verification:** Add a deterministic check (`python -m tools.lint.check_typing_gates`) plus pytest coverage ensuring targeted modules execute without optional dependencies; failing checks gate merges.

## Detailed Plan

### 1. Inventory & Baseline
1. Capture current Ruff `TC00x`, `INP001`, and `PLC2701` violations; export metrics for before/after comparison.
2. Identify modules lacking postponed annotations and catalogue private-module or heavy third-party imports used strictly for typing.
3. Document runtime entry points (CLIs, docs scripts) that currently fail when optional dependencies are absent.

### 2. Typing Façade Infrastructure
1. Create `src/kgfoundry_common/typing/__init__.py` exposing utilities: `TYPE_CHECKING`, `TypeAlias`, centralized aliases (NavMap, ProblemDetails, numpy dtypes), and helper decorators for deferred imports.
2. Add mirror modules for tooling/doc pipelines (`docs.typing`, `tools.typing`) that re-export shared helpers and bridge to runtime types when available.
3. Deprecate direct imports from private modules by providing compatibility shims with warnings, scheduled for removal after migration.

### 3. Postponed Annotation Adoption
1. Write an automated fixer (`tools/typing/apply_postponed_annotations.py`) that inserts `from __future__ import annotations` or validates Python 3.13 semantics.
2. Apply the fixer across targeted directories (`src/`, `docs/_scripts/`, `tools/`, `tests/`), ensuring module docstrings and encoding declarations remain intact.
3. Update Ruff configuration to include a custom rule verifying the directive placement and to disallow accidental reversion.

### 4. Type-Gated Import Refactor
1. Replace direct imports of heavy dependencies within modules with TYPE_CHECKING blocks referencing the new façade.
2. Ensure runtime code now obtains required symbols lazily via helper functions (`resolve_numpy()`), with explicit exceptions if the dependency is legitimately required at runtime.
3. Update docstrings, stubs, and namespace bridges to import from the façade, removing `# type: ignore` pragmas.

### 5. Verification & Tooling
1. Implement `tools/lint/check_typing_gates.py` to scan ASTs, ensuring `TYPE_CHECKING` guards exist for third-party/application imports used solely in annotations.
2. Extend pytest suites to execute representative CLIs (docs builders, navmap tools) inside virtualenvs lacking optional extras, confirming runtime import safety.
3. Add documentation in `AGENTS.md` and change logs summarizing the new rules, plus update onboarding scripts to run the gating check.

## Risks & Mitigations
- **Hidden dependency regressions:** runtime paths may still require certain imports. We mitigate by adding targeted runtime tests and by documenting explicit dependency requirements in module docstrings.
- **Performance overhead:** importing the façade may add minimal indirection. Keep helpers lightweight and memoized; profile hot paths to confirm negligible impact.
- **Developer learning curve:** new helpers may be unfamiliar. Provide examples, code mods, and IDE snippets to standardise usage.

## Migration
1. Land façade modules and helpers behind feature flags; publish short-term compatibility shims.
2. Roll postponed annotations and TYPE_CHECKING conversions in batches (docs, tools, runtime) with targeted Ruff gating per batch.
3. After each batch, run full quality gates: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pytest -q`, and the new `python -m tools.lint.check_typing_gates`.
4. Update documentation and announce new patterns via `openspec` change notes.
5. Remove compatibility shims and enforce façade usage once CI proves stable for two consecutive releases.


