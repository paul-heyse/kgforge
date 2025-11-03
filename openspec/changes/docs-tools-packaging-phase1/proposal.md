## Why
Docs and tooling scripts remain loosely structured collections of `.py` files with mixed executable bits and ad-hoc module boundaries. Ruff currently flags `INP001`, `EXE00x`, and `SLF001` violations because directories lack `__init__.py`, executable flags are inconsistent, and downstream packages import private paths such as `docs._types` and `tools.docstring_builder._cache`. This ambiguity blocks import-linter enforcement, keeps dependency metadata stale (optional extras still list scripts individually), and complicates Pyright/Mypy module resolution. Without formal packages and sanctioned entrypoints, we cannot guarantee deterministic imports or reuse shared helpers in a type-safe way.

## What Changes
- [ ] **ADDED**: `docs.tools.packaging` capability spec codifying package structure, CLI entrypoints, and sanctioned public import surfaces for docs and tooling modules.
- [ ] **MODIFIED**: `docs/` and `tools/` trees to include `__init__.py`, explicit subpackage layouts, and Typer/CLI entry modules with consistent shebang and permission semantics.
- [ ] **MODIFIED**: `pyproject.toml` optional-dependency groups and console entrypoints so docs/tool CLIs are exposed via packages instead of raw scripts.
- [ ] **ADDED**: Import façade modules (`docs.toolchain`, `tools.cli`) that re-export public APIs while preventing imports from private `_types`/`_cache` modules.
- [ ] **MODIFIED**: Tests, stubs, and docstrings to reference the new package namespaces and to validate that `python -m docs.toolchain.build_symbol_index` and equivalent commands succeed.

## Impact
- **Packages**: `docs/_scripts`, `docs/scripts`, `tools/`, stub packages, and consuming runtimes (`tests/docs`, `tests/tools`, `src/kgfoundry_common`).
- **Tooling**: Additional lint/typing guards to enforce package boundaries and normalized executable bits; import-linter contract updates.
- **Docs & Artifacts**: Regenerated docs and CLI help text reflecting new module names; Agent Portal links updated to new package paths.
- **Operations**: Deterministic CLI invocations via `python -m` or installed console scripts; simplified dependency management for optional extras.

- [ ] Ruff (`uv run ruff format && uv run ruff check --fix`) reports zero `INP001`, `EXE00x`, or private-import violations across docs/tooling modules.
- [ ] Pyright, Pyrefly, and MyPy succeed with new package namespaces and without private-module suppressions.
- [ ] CLI smoke tests (`python -m docs.toolchain.build_symbol_index`, `python -m tools.navmap.build`) pass without executable flags or mismatched shebangs.
- [ ] Import-linter contract enforcing `docs.*`/`tools.*` public surfaces passes in CI.

## Out of Scope
- Refactoring business logic inside docs/tooling beyond packaging and entrypoint formalization.
- Introducing new CLI functionality unrelated to packaging alignment.
- Altering deployment pipelines or docker images (only documentation/metadata updates).

## Risks / Mitigations
- **Risk:** Packaging changes may break downstream imports relying on legacy paths.  
  **Mitigation:** Provide compatibility re-export modules with deprecation warnings during the transition, backed by regression tests.
- **Risk:** Adjusting optional-deps metadata might miss transitive dependencies.  
  **Mitigation:** Audit existing extras, add tests that install minimal extras in CI, and run `uv run pip-audit` post-change.
- **Risk:** CLI restructuring could regress automation scripts.  
  **Mitigation:** Add smoke tests capturing both legacy and new invocation patterns; retain legacy entrypoints temporarily with clear removal timeline.

## Alternatives Considered
- Retaining script-based layout with additional lint suppressions — rejected; perpetuates ambiguity and blocks type-safe imports.
- Creating a monolithic `docs_toolchain` package — rejected; keeping docs and tools packages distinct preserves ownership boundaries and avoids unnecessary coupling.
- Defer packaging until after typing-gate Phase 2 — rejected; packaging is prerequisite for robust import-linter rules and façade adoption.


