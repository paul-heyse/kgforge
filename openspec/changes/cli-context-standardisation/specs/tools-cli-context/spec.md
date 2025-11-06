## ADDED Requirements

### Requirement: Shared CLI Context Registry
The system SHALL provide a shared registry under `tools/_shared/cli_context_registry.py` that centralises CLI metadata definitions, enforces validation, and exposes cached accessors for downstream consumers.

#### Scenario: Definition resolves canonical settings
- **GIVEN** the registry registers a CLI definition for `download`
- **WHEN** `settings_for("download")` executes
- **THEN** the returned `CLIToolSettings` uses the repository canonical augment and registry paths (`openapi/_augment_cli.yaml`, `tools/mkdocs_suite/api_registry.yaml`) and the interface ID from the registered definition

#### Scenario: Tooling context caches results
- **GIVEN** a CLI definition exists for `download`
- **WHEN** `context_for("download")` is called twice in the same process
- **THEN** both calls return the identical `CLIToolingContext` instance (no duplicate filesystem work), and the module docstring documents the caching behaviour

#### Scenario: Version resolver applies fallbacks
- **GIVEN** a registry definition that specifies version fallbacks (`kgfoundry-tools`, `kgfoundry`)
- **WHEN** the first package is absent but the second exists
- **THEN** the version resolver returns the version of the second package and the resulting `CLIToolSettings.version` matches that value

#### Scenario: Duplicate registration prevented
- **GIVEN** an attempt to register a second definition with key `"download"`
- **WHEN** `register_cli("download", definition)` executes
- **THEN** the registry raises `ValueError` describing the conflicting key and lists the existing definition summary

#### Scenario: Unknown CLI raises descriptive error
- **GIVEN** no definition registered under key `"unknown-cli"`
- **WHEN** `settings_for("unknown-cli")` executes
- **THEN** the function raises `KeyError` whose message includes the missing key and the set of valid keys, providing remediation guidance per `AGENTS.md`

### Requirement: CLI Modules Delegate to the Registry
Each `cli_context.py` module SHALL derive its public constants and helper functions from the shared registry while preserving API compatibility, docstring quality, and doctest coverage.

#### Scenario: Public helpers remain stable
- **GIVEN** downstream tooling imports `src/download/cli_context.get_cli_settings`
- **WHEN** the function executes after the refactor
- **THEN** it returns a `CLIToolSettings` instance equivalent to the pre-refactor version (`bin_name="kgf"`, matching interface ID/title) and the function signature/docstring remain unchanged

#### Scenario: Module-level constants match registry
- **GIVEN** the registry stores an entry for `orchestration` with `command="orchestration"` and `title="KGFoundry Orchestration"`
- **WHEN** `src/orchestration/cli_context.CLI_COMMAND` and `.CLI_TITLE` are inspected
- **THEN** both constants reflect the registry definition and appear in the module’s `__all__` export list

#### Scenario: Operation overrides surface consistently
- **GIVEN** `codeintel/indexer/cli_context` defines an override mapping for `"symbols"`
- **WHEN** `get_operation_override("symbols")` executes
- **THEN** it delegates to the registry and returns the registered `OperationOverrideModel`, or `None` for non-existent overrides, without duplicating resolution logic

#### Scenario: Multi-CLI modules validate input
- **GIVEN** `docs/_scripts/cli_context` exposes multiple CLI definitions
- **WHEN** `get_cli_settings("docs-build-symbol-index")` executes
- **THEN** the helper returns metadata for that command, and passing an unknown command raises `KeyError` with guidance referencing the registry

#### Scenario: Docstrings reference registry and pass doctest
- **GIVEN** any `cli_context.py` docstring
- **WHEN** doctest/xdoctest runs
- **THEN** the embedded example invoking registry-backed helpers executes successfully and the narrative references the shared registry contract

### Requirement: Validation & Testing
The system SHALL provide automated validation that the registry and module delegations function correctly and satisfy the CLI façade quality gates.

#### Scenario: Registry unit tests remain green
- **GIVEN** `tests/tools/test_cli_context_registry.py`
- **WHEN** `uv run pytest -q` executes
- **THEN** the suite passes, covering registration success, duplicate-key errors, caching behaviour, version fallback logic, and operation override retrieval

#### Scenario: Module integration smoke tests pass
- **GIVEN** the refactored CLI context modules
- **WHEN** the integration smoke test imports each module and calls `get_cli_settings`/`get_cli_context`
- **THEN** all helpers return instances of the correct types with expected field values and no runtime warnings are raised

#### Scenario: Static analysis remains clean
- **GIVEN** the new registry module and updated CLI modules
- **WHEN** `uv run ruff format && uv run ruff check --fix`, `uv run pyright --warnings --pythonversion=3.13`, and `uv run pyrefly check` run
- **THEN** the commands complete without warnings or errors for all touched files

#### Scenario: Documentation and spec validation succeed
- **GIVEN** contributor documentation references the registry
- **WHEN** `make artifacts && git diff --exit-code` and `openspec validate cli-context-standardisation --strict` execute
- **THEN** both commands succeed, confirming docs are regenerated and the change proposal satisfies OpenSpec formatting rules

