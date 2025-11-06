"""Shared CLI context for the navmap tooling command suite.

This module mirrors the shared CLI configuration helpers adopted across other
CLIs (download, orchestration, codeintel). It exposes cached accessors for the
canonical :class:`~tools.CLIToolSettings`, :class:`~tools.CLIToolingContext`, and
typed augment/registry metadata models so the navmap tooling consumes the same
facade as the rest of the system.
"""

from __future__ import annotations

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
    "build": "navmap.build",
    "check": "navmap.check",
}
"""Mapping of subcommand names to canonical operation identifiers."""


_CLI_KEY = "navmap"
_CLI_DEFINITION = CLIContextDefinition(
    command="navmap",
    title="KGFoundry Navmap Builder",
    interface_id="navmap-cli",
    operation_ids=CLI_OPERATION_IDS,
    bin_name="tools-navmap",
    version_resolver=default_version_resolver("kgfoundry-tools", "kgfoundry"),
)

register_cli(_CLI_KEY, _CLI_DEFINITION)

CLI_COMMAND = _CLI_DEFINITION.command
"""Command label used for CLI envelopes and structured logging."""


CLI_TITLE = _CLI_DEFINITION.title
"""Human-readable CLI title used by help text and metadata."""


CLI_INTERFACE_ID = _CLI_DEFINITION.interface_id
"""Registry interface identifier for the navmap tooling CLI."""


def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry metadata inputs.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing augment and registry metadata paths.
    """
    return settings_for(_CLI_KEY)


def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for navmap operations.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment, registry, and CLI configuration.
    """
    return context_for(_CLI_KEY)


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration extracted from the tooling context.

    Returns
    -------
    CLIConfig
        Typed configuration consumed by OpenAPI and CLI integrations.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the shared operation context helper used for metadata lookups.

    Returns
    -------
    OperationContext
        Helper used to resolve operation metadata and overrides.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle.

    Returns
    -------
    ToolingMetadataModel
        Immutable bundle combining augment and registry metadata.
    """
    return tooling_metadata_for(_CLI_KEY)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata describing navmap CLI operation overrides.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata scoped to the navmap CLI operations.
    """
    return augment_for(_CLI_KEY)


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata backing the navmap CLI interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata describing the navmap CLI interface.
    """
    return registry_for(_CLI_KEY)


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return the registry interface metadata for the navmap CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata object associated with the navmap CLI.
    """
    return interface_for(_CLI_KEY)


def get_operation_override(
    subcommand: str, *, tokens: tuple[str, ...] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for the given subcommand when present.

    Parameters
    ----------
    subcommand : str
        CLI subcommand name to look up.
    tokens : tuple[str, ...] | None, optional
        Optional command tokens for nested operation resolution.

    Returns
    -------
    OperationOverrideModel | None
        Override metadata when defined; otherwise ``None``.
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
