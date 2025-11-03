## ADDED Requirements
### Requirement: Formal Docs & Tools Packages
The system SHALL expose docs and tooling modules as explicit Python packages with `__init__.py`, curated `__all__`, and postponed annotations so that type checkers and lint tools operate deterministically.

#### Scenario: Package discovery succeeds
- **GIVEN** the repository after packaging
- **WHEN** `python -c "import docs.toolchain, tools.navmap"` executes
- **THEN** both modules import without relying on implicit namespace packages, and Ruff emits no `INP001` violations for directories under `docs/` or `tools/`

#### Scenario: Public exports documented
- **GIVEN** the generated `docs.toolchain.__all__` and `tools.cli.__all__`
- **WHEN** tests introspect the exports during `pytest -q`
- **THEN** each exported symbol has a NumPy-style docstring, appears in the corresponding stubs, and no downstream code imports private `_types` or `_cache` modules

### Requirement: Normalized CLI Entry Points
The system SHALL provide consistent CLI entry points for docs/tooling operations through `main(argv: Sequence[str] | None = None) -> int` functions that support both `python -m` invocation and registered console scripts, with executable bits and shebangs applied only where required.

#### Scenario: `python -m` invocation works
- **GIVEN** the packaged modules
- **WHEN** `uv run python -m docs.toolchain.build_symbol_index --help` and `uv run python -m tools.navmap.build --help` execute
- **THEN** each command exits with status 0, prints usage text, and Ruff reports no `EXE00x` violations for the underlying modules

#### Scenario: Console scripts installed
- **GIVEN** a fresh virtual environment with optional extras `docs` and `tools` installed
- **WHEN** the console scripts `docs-build-symbol-index` and `tools-navmap-build` run
- **THEN** they delegate to the package `main()` functions, emit structured logs through `kgfoundry_common.logging`, and exit with status codes documented in CLI docstrings

#### Scenario: Legacy entrypoints warn
- **GIVEN** compatibility stubs left in original script locations
- **WHEN** a legacy path such as `python docs/_scripts/build_symbol_index.py` runs
- **THEN** it emits a single `DeprecationWarning` directing users to the new module, and the stub is covered by regression tests to ensure removal readiness

### Requirement: Public Façade & Metadata Alignment
The system SHALL publish sanctioned façade modules (`docs.toolchain.api`, `tools.cli.api`) and update dependency metadata so downstream imports and extras rely solely on the documented packages.

#### Scenario: Façade replaces private imports
- **GIVEN** runtime modules that previously imported `docs._types` or `tools.docstring_builder._cache`
- **WHEN** static analysis (`python -m tools.lint.check_typing_gates --list`) runs
- **THEN** no private path appears in the report, and import-linter contract `docs-tools-packaging` passes without violations

#### Scenario: Optional extras remain functional
- **GIVEN** the `docs` and `tools` optional dependency groups refreshed in `pyproject.toml`
- **WHEN** CI installs each extra into an isolated environment and runs the documented CLIs
- **THEN** commands complete successfully without missing dependency errors, and `uv run pip-audit` on the environment reports zero vulnerabilities introduced by the packaging change


