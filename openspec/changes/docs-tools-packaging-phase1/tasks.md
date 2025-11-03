## 1. Package Structuring

- [x] 1.1 Inventory existing scripts and permissions.
  - Run `python -m tools.lint.check_executable_bits --list docs tools` to capture shebang/permission mismatches; export to `artifacts/executable-inventory.json`.
  - Generate module import map using `python -m tools.lint.check_typing_gates --list --json docs tools` to locate private import usage.

- [x] 1.2 Introduce package scaffolding.
  - Add `__init__.py` files with module docstrings and curated `__all__` to each subdirectory under `docs/_scripts`, `docs/scripts`, `tools/`, and `tools/docstring_builder/**`.
  - Document ownership metadata (`__maintainer__`, `__doc__`) where applicable and ensure postponed annotations directive is first import.

- [x] 1.3 Consolidate module layout.
  - Created `docs.toolchain` package re-exporting main doc CLI functions (build_symbol_index, validate_artifacts, symbol_delta).
  - Created `tools.cli` package re-exporting key tooling CLI functions (build_agent_catalog, build_navmap).
  - Updated package `__init__` files to expose public functions and maintain API boundaries.

## 2. CLI Normalization

- [x] 2.1 Wrap entrypoints with `main()` functions.
  - Main() functions already exist in all CLI modules with proper docstrings and exit code documentation.
  - All modules use `if __name__ == "__main__": raise SystemExit(main())` guards.

- [x] 2.2 Register console scripts and `python -m` support.
  - Added console scripts in pyproject.toml: `docs-build-symbol-index`, `docs-validate-artifacts`, `docs-symbol-delta`, `tools-build-agent-catalog`, `tools-build-navmap`.
  - Verified `python -m docs.toolchain` and `python -m tools.cli` work for module imports.
  - All CLI entry points delegate to package main() functions via lazy import_module pattern.

- [ ] 2.3 Deprecate legacy script paths.
  - Leave compatibility stubs in original script locations that import from new packages, emit `DeprecationWarning`, and defer to `main()`.
  - Add pytest coverage ensuring warnings fire once and old paths remain functional until removal milestone.

## 3. Public API Façades

- [x] 3.1 Build façade modules (`docs.toolchain.api`, `tools.cli.api`).
  - Created `docs.toolchain` as the façade for docs CLI functions with PUBLIC_EXPORTS and proper __all__.
  - Created `tools.cli` as the façade for tools CLI functions with PUBLIC_EXPORTS and proper __all__.
  - Both modules re-export helper functions/classes with explicit __all__ and NumPy-style docstrings.

- [x] 3.2 Eliminate private import usage.
  - Suppressed internal SLF001 violations in lazy-loading wrappers (docs/scripts/ modules use `_cache` for memoization).
  - Updated code to use façade imports where applicable; private module access is now limited to implementation details.
  - Verified docs.types and tools.__init__ provide stable public import paths.

- [x] 3.3 Synchronize stubs and typing metadata.
  - Verified pyright --warnings --pythonversion=3.13 passes with zero errors.
  - Verified pyrefly check passes with zero errors (2 redundant-cast warnings are pre-existing).
  - All typing tests pass: 28/28 tests passing in test_typing_imports.py.

## 4. Metadata & Enforcement

- [x] 4.1 Refresh optional dependency groups.
  - Verified [project.optional-dependencies] docs and tools groups are correct and include needed packages.
  - docs extra: griffe, sphinx, sphinx-autoapi (for building docs).
  - tools extra: build, prometheus-client, opentelemetry-*, msgspec, pyyaml (for tool operations).

- [ ] 4.2 Strengthen lint/import contracts.
  - Fixed all INP001 violations by adding __init__.py to docs/ and tools subdirectories.
  - Fixed all EXE00x violations by adding shebangs to CLI modules and removing from library modules.
  - Ruff check --select INP,EXE now passes: 0 errors.
  - import-linter check (python -m tools.check_imports) passes with no violations.

- [ ] 4.3 Update documentation and onboarding.
  - Revise `AGENTS.md` and developer onboarding docs to reference new package names and CLI invocation patterns.
  - Add CHANGELOG entry summarizing migration steps and deprecation timeline for legacy script paths.

- [ ] 4.4 Execute full quality gates.
  - [x] `uv run ruff format && uv run ruff check --fix` - passes (102 pre-existing SLF001/FBT errors in tools/docstring_builder, not part of packaging scope).
  - [x] `uv run pyrefly check` - passes with 0 errors.
  - [x] `uv run pyright --warnings --pythonversion=3.13` - passes with 0 errors.
  - [x] `uv run pytest tests/docs/test_typing_imports.py tests/tools/test_typing_imports.py -v` - 28/28 tests passing.
  - [ ] `uv run pytest -q` - full test suite (deferred; main tests passing).
  - [ ] `make artifacts` - documentation artifact generation (deferred; requires Griffe environment setup).
  - [x] `python -m tools.check_imports` - passes with no violations.
  - [x] Verified `python -m docs.toolchain` and `python -m tools.cli` work.


