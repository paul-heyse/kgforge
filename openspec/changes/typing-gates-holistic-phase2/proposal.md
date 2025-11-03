## Why
Phase 1 established postponed annotations, façade helpers, and gating scripts, but large portions of the repository still depend on compatibility shims and private imports. Numerous modules in `src/`, `docs/`, and `tools/` continue to import heavy dependencies directly for annotations, and several CLI/tooling entry points skip the new façade in favor of legacy `_types` or `_cache` modules. Without a follow-up we cannot retire the shims, enforce architectural boundaries through import-linter, or guarantee that new code paths remain import-clean. Phase 2 closes these gaps by migrating every remaining module, wiring the façade enforcement into CI, and formalising the runtime smoke tests so TYPE_CHECKING compliance becomes a permanent gate.

## What Changes
- [ ] **MODIFIED**: `code-quality/typing-gates` capability spec to mandate full façade adoption, compatibility-shim removal, and CI import-linter enforcement.
- [ ] **ADDED**: Migration scripts and codemods to replace residual private-module imports with façade references across all packages.
- [ ] **MODIFIED**: Ruff and import-linter configuration to block direct imports of retired modules and to verify TYPE_CHECKING guards during pre-commit and CI.
- [ ] **ADDED**: Runtime & tooling smoke suites that execute representative CLIs with optional dependencies absent, wired into CI as a dedicated job.
- [ ] **REMOVED**: Temporary compatibility shims and deprecated `_types` modules once dependent packages are migrated, with deprecation warnings cleared.

## Impact
- **Packages**: sweeping updates across `src/`, `tools/`, `docs/`, `tests/`, and stub packages to consume façades exclusively.
- **Tooling**: import-linter contract updates, Ruff config tightening, new pre-commit hook, and CI job to run the smoke suite.
- **Docs**: AGENTS.md, developer onboarding, and migration notes updated to reflect shim removal and new enforcement gates.
- **Operations**: ensures all tooling runs deterministically even when optional dependencies are missing, reducing setup friction for new contributors and automation.

- [ ] Ruff (`uv run ruff format && uv run ruff check --fix`) reports zero `TC00x`, `PLC2701`, or private-module violations; new rules prevent regressions.
- [ ] Import-linter contracts pass and guarantee façades are the only cross-package typing surface.
- [ ] Pyright/Pyrefly/Mypy succeed without new suppressions after shim removal.
- [ ] CI smoke job (`python -m tools.tests.typing_gate_smoke`) succeeds with optional dependencies uninstalled.

## Out of Scope
- Introducing new façade packages beyond the existing common/docs/tools trio.
- Replacing third-party dependencies or changing business logic unrelated to typing hygiene.
- Performance optimisations unrelated to import determinism.

## Risks / Mitigations
- **Risk:** Removing shims may break lingering call sites.  
  **Mitigation:** codemod with static analysis, add temporary failing tests pointing to new façades, and stage removal behind feature flag until verification passes.
- **Risk:** Import-linter rules could produce false positives.  
  **Mitigation:** iterate rules in dry-run mode, add targeted exemptions where layering intentionally crosses, and document justified ignores.
- **Risk:** CI smoke tests lengthen pipeline time.  
  **Mitigation:** parallelise jobs and cache environments; scope smoke suite to high-impact CLIs only.

## Alternatives Considered
- Leaving compatibility shims indefinitely — rejected; encourages drift and contradicts the zero-suppression mandate.
- Enforcing via documentation alone — rejected; lacks enforcement and would regress quickly.
- Narrowing scope to runtime packages only — rejected; tooling and docs scripts are primary sources of TC00x violations and must adopt the façade.


