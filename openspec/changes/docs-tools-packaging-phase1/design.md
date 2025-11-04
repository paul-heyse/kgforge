# Design

## Context

Docs tooling evolved organically with standalone scripts housed in `docs/_scripts/` and `docs/scripts/`, many of which are executed directly via `python docs/_scripts/foo.py`. Similarly, the `tools/` directory contains Typer apps, lint helpers, and build utilities that expose private modules (`_types`, `_cache`) without a formal package structure. These choices cause:

- Ruff `INP001` violations because directories lack `__init__.py`, leaving implicit namespace packages.
- Mixed executable permissions (`EXE001`, `EXE002`, `EXE005`) where some scripts have shebangs but are not executable or vice versa.
- Downstream imports that reach into private modules, triggering `SLF001` warnings and making pyright/pyright resolution brittle.
- Optional dependency groups in `pyproject.toml` that reference scripts rather than packages, limiting reuse of shared helpers.

Formalizing docs and tooling as packages unlocks stronger import-linter contracts, simplifies typing-gate enforcement, and clarifies public APIs for consumers.

## Goals

1. Convert `docs/` and `tools/` script trees into explicit Python packages with `__init__.py`, module docstrings, and sanctioned public exports.
2. Normalize CLI entrypoints: each executable script SHALL either become a `python -m package.module` entry or register as a console script with consistent shebang/permissions.
3. Provide explicit public wrappers (façades) so downstream imports no longer use private `_types`, `_cache`, or hidden attributes.
4. Update dependency metadata (pyproject extras, stub packages, tests) to reflect new namespaces and guarantee type-checking stability.

## Non-Goals

- Rewriting business logic within doc generation or tooling flows.
- Changing Problem Details taxonomy or logging semantics beyond packaging moves.
- Merging docs and tools functionality into a single monolithic package; boundaries remain intact.

## Decisions

1. **Package layout:** Introduce top-level packages `docs.toolchain` (for former `_scripts` and `scripts`) and `tools.cli`/`tools.lib` modules. Each subdirectory receives `__init__.py` with module-level exports and NumPy-style docstrings.
2. **CLI normalization:** Replace ad-hoc executable flags with Typer or argparse entry modules named `cli.py`. Register console scripts via `pyproject` while ensuring modules support `python -m package.cli` usage.
3. **Public façades:** Create `docs.toolchain.api` and `tools.cli.api` modules that re-export previously private helpers. Deprecate private access with warnings and plan removal timeline.
4. **Metadata alignment:** Update `pyproject.toml` optional dependencies (`[project.optional-dependencies] docs`, `tools`) and stub packages to match the new module names; reflect changes in documentation.
5. **Enforcement:** Add import-linter rules and Ruff configuration to treat direct imports from retired private modules as errors, ensuring new structure persists.

## Detailed Plan

### 1. Package Structure Migration

1.1 **Inventory scripts & entrypoints** – generate a manifest listing all `.py` files under `docs/_scripts/`, `docs/scripts/`, and `tools/` with current permissions, shebangs, and imports. Classify each script as CLI entry, library helper, or candidate for consolidation.

1.2 **Create packages** – add `__init__.py` files to `docs/_scripts`, `docs/scripts`, `tools/`, and their subdirectories. Each `__init__.py` documents public API exports and attaches `__all__` metadata. Document modules using NumPy-style docstrings referencing related capability specs.

1.3 **Reorganize modules** – rename directories to reflect purpose (`docs/toolchain/build`, `tools/navmap`, etc.). Move common helpers into `docs/toolchain/api.py` or `tools/lib/` modules. Ensure tests import from the new packages.

1.4 **Shebang & permission audit** – enforce the rule: CLI entry modules contain a valid shebang and executable bit; library modules drop both. Incorporate validation into the `tools.lint.check_executable_bits` helper.

### 2. CLI Formalization

2.1 **Typer/argparse wrappers** – for each CLI script, create a `main(argv: Sequence[str] | None = None) -> int` function with docstring describing purpose, parameters, and exits. Wrap existing logic and ensure `__main__` guard returns `SystemExit(main())`.

2.2 **Console script registration** – modify `pyproject.toml` to add `console_scripts` entrypoints (e.g., `docs-build-symbol-index = docs.toolchain.build_symbol_index:main`). Validate entrypoints via `uv run python -m docs.toolchain.build_symbol_index --help` and the installed command in a virtualenv.

2.3 **Legacy shim handling** – provide transitional modules (e.g., `docs/_scripts/build_symbol_index.py`) that import from the new package and emit deprecation warnings. Add tests ensuring deprecation warnings trigger once.

### 3. Public API Façades

3.1 **Façade modules** – implement `docs.toolchain.api` and `tools.cli.api` re-exporting curated functions/classes. Document them in module docstrings and align with typing façade modules from Phase 1.

3.2 **Deprecate private imports** – replace occurrences of `from docs._scripts.shared import ...` or `tools.docstring_builder._cache` with façade imports. Add runtime guards raising `ImportError` if private modules are imported directly post-migration.

3.3 **Stub & test updates** – update stub files under `stubs/docs/**` and `stubs/tools/**` to match new module hierarchies. Adjust tests to reference façade modules and confirm `__all__` completeness.

### 4. Metadata & Enforcement

4.1 **Optional dependency audit** – review `pyproject.toml` extras (`docs`, `tools`) to ensure dependencies map to new package entrypoints. Add integration tests installing extras into isolated venvs and running core CLIs.

4.2 **Import-linter contract** – define `docs-toolchain-public` and `tools-cli-public` contracts preventing runtime packages from importing deprecated modules. Integrate into CI and pre-commit.

4.3 **Lint & typing** – run full gate (`uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pyright --warnings --pythonversion=3.13`) after restructuring. Add targeted pytest suites verifying CLI behaviour and docstring coverage.

## Risks & Mitigations

- **Runtime breakage** – automation scripts may rely on old import paths. Mitigate with compatibility shims, deprecation warnings, and release notes describing migration timeline.
- **Packaging regressions** – incorrect `__init__.py` exports may hide functionality. Mitigate by adding unit tests verifying `__all__` and using `inspect.getmembers` to confirm exports.
- **Optional extra drift** – missing dependency updates could break CLIs. Mitigate by adding smoke tests that install extras and run CLIs in CI.

## Migration

1. Land package structure changes for docs, followed by tooling, each with compatibility shims and updated tests.
2. Introduce CLI wrappers and console scripts, maintaining legacy entrypoints temporarily with warnings.
3. Update metadata, stubs, and enforcement tooling; run full gate and smoke tests.
4. Communicate changes via CHANGELOG and AGENTS.md; set removal date for shims and track progress in release checklist.


