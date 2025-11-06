"""Shared CLI context helpers for the code-intel indexer command suite.

The helpers mirror the shared tooling contracts adopted by other CLIs
(`download`, `orchestration`, `tools.docstring_builder`). They expose cached
accessors for the :class:`~tools.CLIToolSettings`,
:class:`~tools.CLIToolingContext`, and typed augment/registry metadata models so
the Typer application, OpenAPI generator, MkDocs scripts, and downstream
observability tooling all consume the same canonical configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tools import (
    AugmentMetadataModel,
    CLIToolingContext,
    CLIToolSettings,
    OperationOverrideModel,
    RegistryInterfaceModel,
    RegistryMetadataModel,
    ToolingMetadataModel,
)
from tools.cli_context_registry import (
    CLIContextDefinition,
    augment_for,
    context_for,
    default_version_resolver,
    interface_for,
    operation_override_for,
    register_cli,
    registry_for,
    settings_for,
    tooling_metadata_for,
)

if TYPE_CHECKING:
    from tools.typer_to_openapi_cli import CLIConfig, OperationContext


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to resolve augment and registry metadata paths."""


CLI_OPERATION_IDS: dict[str, str] = {
    "query": "cli.codeintel.query",
    "symbols": "cli.codeintel.symbols",
}
"""Mapping of subcommand names to canonical augment/registry operation IDs."""


_CLI_KEY = "codeintel"
_CLI_DEFINITION = CLIContextDefinition(
    command="codeintel",
    title="KGFoundry CodeIntel Indexer",
    interface_id="codeintel-indexer",
    operation_ids=CLI_OPERATION_IDS,
    bin_name="kgf-codeintel",
    version_resolver=default_version_resolver("kgfoundry-codeintel", "kgfoundry"),
)

register_cli(_CLI_KEY, _CLI_DEFINITION)

CLI_COMMAND = _CLI_DEFINITION.command
"""Human-friendly command label used for envelopes and logging."""


CLI_TITLE = _CLI_DEFINITION.title
"""Title surfaced in Typer help text and OpenAPI metadata."""


CLI_INTERFACE_ID = _CLI_DEFINITION.interface_id
"""Registry interface identifier for the code-intel indexer CLI."""


def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry metadata inputs.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing augment and registry metadata paths.
    """
    return settings_for(_CLI_KEY)


def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for the code-intel CLI.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment, registry, and configuration data.
    """
    return context_for(_CLI_KEY)


def get_cli_config() -> CLIConfig:
    """Return the typed ``CLIConfig`` extracted from the tooling context.

    Returns
    -------
    CLIConfig
        Typed configuration consumed by OpenAPI and Typer integrations.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the operation context helper used for OpenAPI generation.

    Returns
    -------
    OperationContext
        Helper instance for resolving operation metadata during generation.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle for code-intel.

    Returns
    -------
    ToolingMetadataModel
        Immutable bundle combining augment and registry metadata.
    """
    return tooling_metadata_for(_CLI_KEY)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata describing code-intel CLI operations.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata model scoped to the code-intel CLI operations.
    """
    return augment_for(_CLI_KEY)


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata backing the code-intel CLI interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata describing the code-intel CLI interface definition.
    """
    return registry_for(_CLI_KEY)


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return registry interface metadata for the code-intel CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata linked to the code-intel CLI.
    """
    return interface_for(_CLI_KEY)


def get_operation_override(
    subcommand: str, *, tokens: Sequence[str] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for ``subcommand`` when defined.

    Returns
    -------
    OperationOverrideModel | None
        Operation override metadata when present; otherwise ``None``.
    """
    return operation_override_for(_CLI_KEY, subcommand=subcommand, tokens=tokens)


__all__ = [
    "CLI_COMMAND",
    "CLI_INTERFACE_ID",
    "CLI_OPERATION_IDS",
    "CLI_TITLE",
    "REPO_ROOT",
    "get_augment_metadata",
    "get_cli_config",
    "get_cli_context",
    "get_cli_settings",
    "get_interface_metadata",
    "get_operation_context",
    "get_operation_override",
    "get_registry_metadata",
    "get_tooling_metadata",
]
