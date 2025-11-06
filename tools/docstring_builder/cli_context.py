"""Shared CLI context helpers for the docstring builder tooling suite.

The canonical CLI metadata (augment + registry) powers downstream tooling
including OpenAPI generation, CLI diagrams, and automation scripts. This module
exposes lightweight accessors so external integrations—such as
``tools/generate_docstrings.py``—can load the same typed metadata without
importing the heavy Typer CLI entry point.
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
    "generate": "docstrings.generate",
    "fix": "docstrings.fix",
    "fmt": "docstrings.fmt",
    "update": "docstrings.update",
    "check": "docstrings.check",
    "diff": "docstrings.diff",
    "lint": "docstrings.lint",
    "measure": "docstrings.measure",
    "list": "docstrings.list",
    "harvest": "docstrings.harvest",
    "schema": "docstrings.schema",
    "clear-cache": "docstrings.clear_cache",
    "clear_cache": "docstrings.clear_cache",
    "doctor": "docstrings.doctor",
}
"""Mapping of subcommand names to canonical operation identifiers."""


_CLI_KEY = "docstrings"
_CLI_DEFINITION = CLIContextDefinition(
    command="docstrings",
    title="Docstring Builder CLI",
    interface_id="docstring-builder-cli",
    operation_ids=CLI_OPERATION_IDS,
    bin_name="docstring-builder",
    version_resolver=default_version_resolver("kgfoundry-tools", "kgfoundry"),
)

register_cli(_CLI_KEY, _CLI_DEFINITION)

CLI_COMMAND = _CLI_DEFINITION.command
"""Logical command name used for envelopes and documentation artifacts."""


CLI_TITLE = _CLI_DEFINITION.title
"""Human-readable CLI title sourced from registry metadata."""


CLI_INTERFACE_ID = _CLI_DEFINITION.interface_id
"""Registry interface identifier for the docstring builder CLI."""


def get_cli_settings() -> CLIToolSettings:
    """Return canonical CLI settings for the docstring builder suite.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing augment and registry metadata.
    """
    return settings_for(_CLI_KEY)


def get_cli_context() -> CLIToolingContext:
    """Return the lazily loaded CLI tooling context.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment and registry metadata.
    """
    return context_for(_CLI_KEY)


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration consumed by downstream tooling.

    Returns
    -------
    CLIConfig
        Typed configuration generated from augment and registry metadata.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the helper used to construct OpenAPI operation payloads.

    Returns
    -------
    OperationContext
        Helper object for resolving operation metadata and OpenAPI payloads.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle.

    Returns
    -------
    ToolingMetadataModel
        Immutable bundle combining augment and registry metadata models.
    """
    return tooling_metadata_for(_CLI_KEY)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata for docstring builder operations.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata scoped to docstring builder commands.
    """
    return augment_for(_CLI_KEY)


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata describing the docstring builder interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata containing interface definitions for the CLI.
    """
    return registry_for(_CLI_KEY)


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return the registry interface metadata for the docstring builder CLI.

    Returns
    -------
    RegistryInterfaceModel
        Registry interface entry associated with the CLI.
    """
    return interface_for(_CLI_KEY)


def get_operation_override(
    subcommand: str, *, tokens: tuple[str, ...] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for ``subcommand`` when available.

    Parameters
    ----------
    subcommand : str
        Subcommand name to resolve.
    tokens : tuple[str, ...] | None, optional
        Optional token sequence for nested command structures.

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
