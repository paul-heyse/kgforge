# Design

## Context

Phase 1 introduced postponed annotations, typing façades, and lint tooling, but deferred full migration to avoid blocking ongoing feature work. Compatibility shims presently forward imports from deprecated private modules (`docs._types`, `_cache`, `_ParameterKind`, etc.) to the façade. Ruff still ignores several directories via targeted per-file ignores, and import-linter does not yet enforce façade-only imports. Runtime CLIs rely on environment-specific optional dependencies remaining installed; when missing, some entry points continue to crash before reaching guarded code. Phase 2 aims to eliminate these transitional states, making TYPE_CHECKING compliance and façade usage systemic and verifiable.

## Goals
- Remove all compatibility shims and private-module imports, ensuring every package consumes the sanctioned façade modules.
- Tighten lint/type gates so violations fail fast at pre-commit/CI, including import-linter contracts for façade boundaries.
- Establish deterministic smoke tests that execute tooling/CLI entry points without optional extras, verifying runtime import safety.
- Document and operationalise the new enforcement (AGENTS.md, onboarding, release notes) to align future work with the standards.

## Non-Goals
- Rewriting business logic or altering dependency graphs beyond typing-related imports.
- Addressing non-typing Ruff warnings unless touched by the migration.
- Adding new façade modules beyond the existing common/docs/tools trio.

## Decisions
1. **Façade exclusivity:** All modules SHALL import shared types via the façade; direct imports from private packages or optional dependencies (for annotations) are prohibited.
2. **Import-linter contract:** Introduce a new import-linter configuration that enforces façade usage for type-only imports, integrated into CI and local tooling.
3. **Shim retirement:** Compatibility shims created in Phase 1 will be removed after migrations; attempts to import them will raise informative `ImportError` messages.
4. **Smoke suite:** Build a dedicated smoke test harness that runs key CLIs with optional dependencies removed, ensuring deterministic behaviour.
5. **Enforcement automation:** Update Ruff configuration and pre-commit hooks to include custom checks for façade imports and missing TYPE_CHECKING guards, backed by AST tooling.

## Detailed Plan

### 1. Audit & Codemod Preparation
1. Generate a comprehensive inventory of modules still referencing private `_types`, `_cache`, or heavy third-party imports outside TYPE_CHECKING blocks.
2. Develop codemods using LibCST or Bowler to replace legacy imports with façade equivalents, including docstrings and `__all__` updates.
3. Stage codemods by subsystem (docs, tools, runtime, tests) to simplify review and rollback.

### 2. Façade Adoption & Shim Removal
1. Apply codemods to each subsystem, running Ruff/pyright/pyrefly after each batch.
2. Update stubs and namespace bridges to reference façade modules directly; align `__all__` lists and documentation.
3. Remove compatibility shims, replacing them with explicit `ImportError` guidance.

### 3. Gating Enhancements
1. Update Ruff configuration to drop per-file ignores introduced in Phase 1 and enable repository-specific rules against private-module imports.
2. Extend `tools.lint.check_typing_gates` with new heuristics detecting direct heavy imports in annotations.
3. Configure import-linter (new contract `typing-facade-only`) to ensure packages reference the façade for shared types.

### 4. Smoke Testing Infrastructure
1. Implement `tools/tests/typing_gate_smoke.py` (or pytest equivalent) that removes optional dependencies and executes representative CLIs/docs scripts.
2. Integrate the smoke suite into CI as a parallel job with caching for virtualenv setup.
3. Add targeted unit tests verifying `ImportError` messages from removed shims and verifying façade helper behaviour.

### 5. Documentation & Onboarding
1. Update AGENTS.md and onboarding documentation to describe the enforced façade-only policy and how to run the smoke suite locally.
2. Provide migration notes (CHANGELOG, internal comms) summarising removed modules and new enforcement steps.
3. Add IDE snippets/templates to encourage developers to use façade imports (e.g., `from kgfoundry_common.typing import TYPE_CHECKING`).

## Risks & Mitigations
- **Risk:** Codemods may introduce subtle import-order differences.  
  **Mitigation:** run Ruff format after each codemod and add targeted regression tests for modules with side effects.
- **Risk:** Import-linter false positives may block legitimate cross-imports.  
  **Mitigation:** maintain an allowlist with documented justification and schedule follow-up to eliminate allowances.
- **Risk:** Smoke suite could mask legitimate runtime dependency requirements.  
  **Mitigation:** encode expected failure modes as Problem Details, ensuring tests assert correct behaviour rather than success-only.

## Migration
1. Land codemods and subsystem migrations sequentially (docs → tools → runtime → tests) with CI verification after each.
2. Once migrations stabilise, remove shims and update import-linter/Ruff configs in the same change to avoid drift.
3. Introduce the smoke suite and CI job, gating merges on its success.
4. Announce enforcement changes and update documentation.
5. Monitor CI for two release cycles; if clean, remove any temporary allowances and archive the change.


