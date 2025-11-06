# orchestration.cli_context

Typer-powered orchestration command suite covering indexing flows, API bootstrapping,
and end-to-end demonstrations. Each command maps to a generated OpenAPI operation
consumed by the MkDocs suite.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/orchestration/cli_context.py)

## Sections

- **Public API**

## Contents

### orchestration.cli_context.get_augment_metadata

::: orchestration.cli_context.get_augment_metadata

### orchestration.cli_context.get_cli_config

::: orchestration.cli_context.get_cli_config

### orchestration.cli_context.get_cli_context

::: orchestration.cli_context.get_cli_context

### orchestration.cli_context.get_cli_settings

::: orchestration.cli_context.get_cli_settings

### orchestration.cli_context.get_interface_metadata

::: orchestration.cli_context.get_interface_metadata

### orchestration.cli_context.get_operation_context

::: orchestration.cli_context.get_operation_context

### orchestration.cli_context.get_operation_override

::: orchestration.cli_context.get_operation_override

### orchestration.cli_context.get_registry_metadata

::: orchestration.cli_context.get_registry_metadata

### orchestration.cli_context.get_tooling_metadata

::: orchestration.cli_context.get_tooling_metadata

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Sequence`, `pathlib.Path`, `tools.AugmentMetadataModel`, `tools.CLIToolSettings`, `tools.CLIToolingContext`, `tools.OperationOverrideModel`, `tools.RegistryInterfaceModel`, `tools.RegistryMetadataModel`, `tools.ToolingMetadataModel`, `tools.cli_context_registry.CLIContextDefinition`, `tools.cli_context_registry.augment_for`, `tools.cli_context_registry.context_for`, `tools.cli_context_registry.default_version_resolver`, `tools.cli_context_registry.interface_for`, `tools.cli_context_registry.operation_override_for`, `tools.cli_context_registry.register_cli`, `tools.cli_context_registry.registry_for`, `tools.cli_context_registry.settings_for`, `tools.cli_context_registry.tooling_metadata_for`, `tools.typer_to_openapi_cli.CLIConfig`, `tools.typer_to_openapi_cli.OperationContext`, `typing.TYPE_CHECKING`, `typing.cast`

**Imported by:** [orchestration.cli](./orchestration/cli.md)

## Autorefs Examples

- [orchestration.cli_context.get_augment_metadata][]
- [orchestration.cli_context.get_cli_config][]
- [orchestration.cli_context.get_cli_context][]

## Neighborhood

```d2
direction: right
"orchestration.cli_context": "orchestration.cli_context" { link: "./orchestration/cli_context.md" }
"__future__.annotations": "__future__.annotations"
"orchestration.cli_context" -> "__future__.annotations"
"collections.abc.Sequence": "collections.abc.Sequence"
"orchestration.cli_context" -> "collections.abc.Sequence"
"pathlib.Path": "pathlib.Path"
"orchestration.cli_context" -> "pathlib.Path"
"tools.AugmentMetadataModel": "tools.AugmentMetadataModel"
"orchestration.cli_context" -> "tools.AugmentMetadataModel"
"tools.CLIToolSettings": "tools.CLIToolSettings"
"orchestration.cli_context" -> "tools.CLIToolSettings"
"tools.CLIToolingContext": "tools.CLIToolingContext"
"orchestration.cli_context" -> "tools.CLIToolingContext"
"tools.OperationOverrideModel": "tools.OperationOverrideModel"
"orchestration.cli_context" -> "tools.OperationOverrideModel"
"tools.RegistryInterfaceModel": "tools.RegistryInterfaceModel"
"orchestration.cli_context" -> "tools.RegistryInterfaceModel"
"tools.RegistryMetadataModel": "tools.RegistryMetadataModel"
"orchestration.cli_context" -> "tools.RegistryMetadataModel"
"tools.ToolingMetadataModel": "tools.ToolingMetadataModel"
"orchestration.cli_context" -> "tools.ToolingMetadataModel"
"tools.cli_context_registry.CLIContextDefinition": "tools.cli_context_registry.CLIContextDefinition"
"orchestration.cli_context" -> "tools.cli_context_registry.CLIContextDefinition"
"tools.cli_context_registry.augment_for": "tools.cli_context_registry.augment_for"
"orchestration.cli_context" -> "tools.cli_context_registry.augment_for"
"tools.cli_context_registry.context_for": "tools.cli_context_registry.context_for"
"orchestration.cli_context" -> "tools.cli_context_registry.context_for"
"tools.cli_context_registry.default_version_resolver": "tools.cli_context_registry.default_version_resolver"
"orchestration.cli_context" -> "tools.cli_context_registry.default_version_resolver"
"tools.cli_context_registry.interface_for": "tools.cli_context_registry.interface_for"
"orchestration.cli_context" -> "tools.cli_context_registry.interface_for"
"tools.cli_context_registry.operation_override_for": "tools.cli_context_registry.operation_override_for"
"orchestration.cli_context" -> "tools.cli_context_registry.operation_override_for"
"tools.cli_context_registry.register_cli": "tools.cli_context_registry.register_cli"
"orchestration.cli_context" -> "tools.cli_context_registry.register_cli"
"tools.cli_context_registry.registry_for": "tools.cli_context_registry.registry_for"
"orchestration.cli_context" -> "tools.cli_context_registry.registry_for"
"tools.cli_context_registry.settings_for": "tools.cli_context_registry.settings_for"
"orchestration.cli_context" -> "tools.cli_context_registry.settings_for"
"tools.cli_context_registry.tooling_metadata_for": "tools.cli_context_registry.tooling_metadata_for"
"orchestration.cli_context" -> "tools.cli_context_registry.tooling_metadata_for"
"tools.typer_to_openapi_cli.CLIConfig": "tools.typer_to_openapi_cli.CLIConfig"
"orchestration.cli_context" -> "tools.typer_to_openapi_cli.CLIConfig"
"tools.typer_to_openapi_cli.OperationContext": "tools.typer_to_openapi_cli.OperationContext"
"orchestration.cli_context" -> "tools.typer_to_openapi_cli.OperationContext"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"orchestration.cli_context" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"orchestration.cli_context" -> "typing.cast"
"orchestration.cli": "orchestration.cli" { link: "./orchestration/cli.md" }
"orchestration.cli" -> "orchestration.cli_context"
"orchestration.cli_context_code": "orchestration.cli_context code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/orchestration/cli_context.py" }
"orchestration.cli_context" -> "orchestration.cli_context_code" { style: dashed }
```

