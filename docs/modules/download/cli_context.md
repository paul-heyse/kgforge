# download.cli_context

Downloader command suite that sources external corpora (currently OpenAlex) using the shared
CLI tooling contracts. Emits structured envelopes and metadata so downstream tooling (OpenAPI,
diagrams, documentation) remains in sync without bespoke glue.

## Sections

- **Public API**

## Contents

### download.cli_context._resolve_cli_version

::: download.cli_context._resolve_cli_version

### download.cli_context.get_augment_metadata

::: download.cli_context.get_augment_metadata

### download.cli_context.get_cli_config

::: download.cli_context.get_cli_config

### download.cli_context.get_cli_context

::: download.cli_context.get_cli_context

### download.cli_context.get_cli_settings

::: download.cli_context.get_cli_settings

### download.cli_context.get_interface_metadata

::: download.cli_context.get_interface_metadata

### download.cli_context.get_operation_context

::: download.cli_context.get_operation_context

### download.cli_context.get_operation_override

::: download.cli_context.get_operation_override

### download.cli_context.get_registry_metadata

::: download.cli_context.get_registry_metadata

### download.cli_context.get_tooling_metadata

::: download.cli_context.get_tooling_metadata

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Sequence`, `functools.lru_cache`, `importlib.metadata.PackageNotFoundError`, `importlib.metadata.version`, `pathlib.Path`, `tools.AugmentMetadataModel`, `tools.CLIToolSettings`, `tools.CLIToolingContext`, `tools.OperationOverrideModel`, `tools.RegistryInterfaceModel`, `tools.RegistryMetadataModel`, `tools.ToolingMetadataModel`, `tools.load_cli_tooling_context`, `tools.typer_to_openapi_cli.CLIConfig`, `tools.typer_to_openapi_cli.OperationContext`, `typing.TYPE_CHECKING`, `typing.cast`

**Imported by:** [download.cli](cli.md)

## Autorefs Examples

- [download.cli_context._resolve_cli_version][]
- [download.cli_context.get_augment_metadata][]
- [download.cli_context.get_cli_config][]

## Neighborhood

```d2
direction: right
"download.cli_context": "download.cli_context" { link: "cli_context.md" }
"__future__.annotations": "__future__.annotations"
"download.cli_context" -> "__future__.annotations"
"collections.abc.Sequence": "collections.abc.Sequence"
"download.cli_context" -> "collections.abc.Sequence"
"functools.lru_cache": "functools.lru_cache"
"download.cli_context" -> "functools.lru_cache"
"importlib.metadata.PackageNotFoundError": "importlib.metadata.PackageNotFoundError"
"download.cli_context" -> "importlib.metadata.PackageNotFoundError"
"importlib.metadata.version": "importlib.metadata.version"
"download.cli_context" -> "importlib.metadata.version"
"pathlib.Path": "pathlib.Path"
"download.cli_context" -> "pathlib.Path"
"tools.AugmentMetadataModel": "tools.AugmentMetadataModel"
"download.cli_context" -> "tools.AugmentMetadataModel"
"tools.CLIToolSettings": "tools.CLIToolSettings"
"download.cli_context" -> "tools.CLIToolSettings"
"tools.CLIToolingContext": "tools.CLIToolingContext"
"download.cli_context" -> "tools.CLIToolingContext"
"tools.OperationOverrideModel": "tools.OperationOverrideModel"
"download.cli_context" -> "tools.OperationOverrideModel"
"tools.RegistryInterfaceModel": "tools.RegistryInterfaceModel"
"download.cli_context" -> "tools.RegistryInterfaceModel"
"tools.RegistryMetadataModel": "tools.RegistryMetadataModel"
"download.cli_context" -> "tools.RegistryMetadataModel"
"tools.ToolingMetadataModel": "tools.ToolingMetadataModel"
"download.cli_context" -> "tools.ToolingMetadataModel"
"tools.load_cli_tooling_context": "tools.load_cli_tooling_context"
"download.cli_context" -> "tools.load_cli_tooling_context"
"tools.typer_to_openapi_cli.CLIConfig": "tools.typer_to_openapi_cli.CLIConfig"
"download.cli_context" -> "tools.typer_to_openapi_cli.CLIConfig"
"tools.typer_to_openapi_cli.OperationContext": "tools.typer_to_openapi_cli.OperationContext"
"download.cli_context" -> "tools.typer_to_openapi_cli.OperationContext"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"download.cli_context" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"download.cli_context" -> "typing.cast"
"download.cli": "download.cli" { link: "cli.md" }
"download.cli" -> "download.cli_context"
```

