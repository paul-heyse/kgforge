## ADDED Requirements
### Requirement: Shared CLI Tooling Context
The documentation tooling SHALL provide a shared CLI context module that loads augmentation metadata, registry interfaces, and OpenAPI operation contexts through typed APIs so scripts generate consistent CLI artefacts.

#### Scenario: CLI generator uses shared context
- **GIVEN** the OpenAPI generator entry point `tools/typer_to_openapi_cli.py`
- **WHEN** the CLI runs with default augment and registry paths
- **THEN** it constructs a `CLIConfig` via `tools._shared.cli_tooling.load_cli_tooling_context` and no longer executes bespoke augment parsing logic

#### Scenario: MkDocs CLI diagram consumes shared module
- **GIVEN** the MkDocs CLI diagram script `tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py`
- **WHEN** it collects operations
- **THEN** it imports the shared context helper, retrieves operations from the returned `CLIConfig.operation_context`, and produces diagrams whose tags and anchors match the OpenAPI generator output

#### Scenario: Shared helpers validate inputs
- **GIVEN** a missing or malformed augment file
- **WHEN** `load_cli_tooling_context` executes
- **THEN** it raises `CLIConfigError` with an RFC 9457 Problem Details payload and does not allow downstream scripts to proceed with inconsistent metadata

#### Scenario: Typed consumers remain lint clean
- **GIVEN** Ruff, Pyright, and Pyrefly checks run on the tooling modules
- **WHEN** the shared context is in place
- **THEN** the analyzers report zero errors related to missing attributes, ambiguous dict access, or duplicated augment parsing logic
