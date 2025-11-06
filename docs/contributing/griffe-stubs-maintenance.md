# Griffe Typing: Upstream Integration (Stubs Decommissioned)

## Summary

As of **2025-11-06** the project no longer vendors custom type stubs for the
`griffe` package. We rely on the upstream library (and its bundled
`py.typed` metadata) directly in both runtime code and tests. This document
records the new expectations for keeping the documentation toolchain aligned
with Griffe without the maintenance burden of a parallel stub tree.

## Why the stubs were removed

- **Duplicate maintenance**: the vendored `stubs/griffe/**` mirror had to be
  kept in sync with every upstream release and frequently drifted.
- **Upstream coverage**: modern Griffe releases expose robust typing support
  and ship `py.typed`, making the local copies redundant.
- **Testing parity**: exercising the real library in our tests gives higher
  confidence that CLI tooling, MkDocs helpers, and symbol builders behave
  correctly in production.

## New workflow

1. **Runtime and tests use the real library**
   - Import `griffe` directly; avoid `safe_import_griffe` only when the code path
     truly needs to degrade gracefully without the dependency.
   - Tests should monkeypatch `griffe.load` / `griffe.load_extensions` if they need
     to simulate failure scenarios rather than installing stub modules.

2. **Dependency management**
   - Griffe remains an explicit dependency in `pyproject.toml` and is resolved via
     `uv sync`.
   - When upgrading, run the documentation toolchain smoke tests (see below) to
     confirm compatibility.

3. **Quality gates**
   - `uv run pyright docs/_scripts` – ensures type checkers see the upstream
     annotations.
   - `uv run pytest tests/tools/mkdocs_suite/test_gen_module_pages.py` – covers
     extension discovery and fallback behaviour with the real loader.
   - `uv run pytest tests/docs/test_griffe_facade.py` – guards the docs facade
     that wraps Griffe objects.

4. **Downstream tooling**
   - The docstring builder doctor now only tracks the remaining vendored stubs
     (`stubs/libcst`, `stubs/mkdocs_gen_files`). No action is required for Griffe.

## Migration checklist (completed)

- [x] Delete `stubs/griffe/**`.
- [x] Update tests to import / monkeypatch the real `griffe` package.
- [x] Remove doc tooling references to the old stub path.
- [x] Refresh documentation (`openapi/_augment_cli.yaml`, `tools/mkdocs_suite/api_registry.yaml`,
  `docs/api/interfaces.md`) to ensure CLI metadata points at the canonical builder.

## Verifying future upgrades

When Griffe releases a new version:

1. Run `uv sync` to pull the update into `uv.lock`.
2. Execute the documentation toolchain tests mentioned above.
3. If regressions appear, file an issue with reproduction steps, including
   the failing command (`render_module_pages`, symbol index build, etc.).
4. If an upstream typing bug is discovered, prefer opening a patch in Griffe
   rather than reintroducing local stubs.

## Historical reference

For archival purposes, the previous stub-maintenance workflow (covering
`stubs/griffe/**`) is preserved in Git history prior to commit
`docs-typing-2025-11-06`. Consult that snapshot only if you need details about
how the deprecated stubs were structured.
