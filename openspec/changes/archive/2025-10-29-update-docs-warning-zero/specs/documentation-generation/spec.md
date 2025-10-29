## ADDED Requirements

### Requirement: Validated Sphinx Extension Configuration
The system SHALL load only Sphinx extensions that expose a `setup()` function and conform to documented configuration schemas.

#### Scenario: Unsupported extensions rejected at review time
- **WHEN** `docs/conf.py` is updated
- **THEN** the configuration excludes `numpydoc_validation` (and any other extension lacking `setup()`), instead invoking the validator via CLI prior to the Sphinx build

#### Scenario: Configuration types match Sphinx expectations
- **WHEN** Sphinx reads `conf.py`
- **THEN** `numpydoc_validation_exclude` (and similar options) are defined using the exact container type Sphinx documents (e.g., `set` not `list`), avoiding runtime warnings

#### Scenario: Unit test guards against regressions
- **WHEN** the documentation tooling test suite runs
- **THEN** a test importing `docs.conf` asserts the invalid extension is absent and configuration types are correct

### Requirement: Reliable Intersphinx Inventories
The system SHALL maintain intersphinx mappings that resolve successfully at build time and fail fast when upstream URLs change.

#### Scenario: Live inventories resolve
- **WHEN** `make html` (or `tools/update_docs.sh`) begins
- **THEN** intersphinx URLs for Typer (`https://typer.tiangolo.com/latest/objects.inv`) and DuckDB (`https://duckdb.org/docs/api/python_api/objects.inv`) respond with HTTP 200 and loadable inventories

#### Scenario: Early failure on unreachable inventories
- **WHEN** an intersphinx URL returns HTTP 404/500 or times out
- **THEN** `docs/conf.py` raises a configuration error, aborting the build before Sphinx emits warnings

#### Scenario: CI guard rails
- **WHEN** the documentation CI job runs
- **THEN** a pre-build smoke test verifies every configured intersphinx URL, preventing stale links from reaching main

### Requirement: AutoAPI Output Integrated into Navigation
The system SHALL expose all generated API documents via at least one toctree so Sphinx recognises them as part of the navigation graph.

#### Scenario: API landing page includes hidden toctree
- **WHEN** AutoAPI emits Markdown into `docs/api/**`
- **THEN** `docs/api/index.md` contains a hidden glob toctree referencing every package/module page

#### Scenario: Main index references API landing page
- **WHEN** `docs/index.md` renders
- **THEN** it includes (explicitly or via hidden toctree) the API landing page so Sphinx marks the entire tree as included

#### Scenario: Regression check for toctree coverage
- **WHEN** the documentation build concludes
- **THEN** a test parses `_build/json/index.fjson` (or another machine-readable artefact) and asserts every `docs/api/**` page appears in at least one toctree

### Requirement: Schema Catalogue Glob Resolution
The system SHALL publish generated schema artefacts under the path expected by `docs/reference/schemas/index.md` so the glob matches at least one document every build.

#### Scenario: Glob locates exported schemas
- **WHEN** `tools/docs/export_schemas.py` writes JSON files
- **THEN** the files reside in the directory referenced by the toctree glob (or the glob is updated to match the actual directory)

#### Scenario: Build fails if schemas disappear
- **WHEN** schemas are missing or the glob matches nothing
- **THEN** the documentation build treats it as an error (caught via warnings-as-errors policy)

### Requirement: Gallery References Resolve
The system SHALL ensure gallery examples reference real importable modules so generated documentation links resolve without `py:mod` warnings.

#### Scenario: Gallery scripts import real packages
- **WHEN** Sphinx-Gallery processes `examples/*.py`
- **THEN** each script imports packages that exist within the repository (`kgfoundry_common`, etc.), or relies on an intentional shim that Sphinx can resolve

#### Scenario: Doctest / xdoctest coverage prevents regressions
- **WHEN** doctests run on gallery examples
- **THEN** missing module imports surface as test failures rather than runtime warnings during doc builds

### Requirement: Zero-Warning Documentation Builds
The system SHALL run the Sphinx build in warnings-as-errors mode and assert that the generated `sphinx-warn.log` is empty after every pipeline execution.

#### Scenario: Sphinx invoked with -W
- **WHEN** `tools/update_docs.sh` calls Sphinx
- **THEN** it passes `-W`, causing any warning to fail the build immediately

#### Scenario: Warning log post-check
- **WHEN** the build finishes
- **THEN** the script verifies `sphinx-warn.log` exists and contains zero warnings, failing otherwise

#### Scenario: Documentation guidelines updated
- **WHEN** contributors consult `README-AUTOMATED-DOCUMENTATION.md`
- **THEN** they find instructions for running the zero-warning pipeline locally and guidance for troubleshooting new warnings

