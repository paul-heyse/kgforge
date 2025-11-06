"""Shared CLI context loaders for the download command suite.

The download CLI participates in repository-wide metadata contracts handled by the public
``tools`` facade. This module encapsulates the configuration needed to load the typed augment and
registry metadata, exposing cached helpers that other modules (for example ``download.cli``) can
import without duplicating path logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
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
    load_cli_tooling_context,
)

if TYPE_CHECKING:
    from tools.typer_to_openapi_cli import CLIConfig, OperationContext


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to resolve augment and registry paths."""

CLI_COMMAND = "download"
"""Logical command name used for CLI envelopes and logging."""

CLI_TITLE = "KGFoundry Downloader"
"""Human-readable title for CLI documentation and Typer help text."""

CLI_INTERFACE_ID = "download-cli"
"""Registry interface identifier backing the download CLI."""

CLI_OPERATION_IDS: dict[str, str] = {
    "harvest": "cli.download.harvest",
}
"""Mapping of subcommand names to canonical operation identifiers."""


def _resolve_cli_version() -> str:
    """Return the installed kgfoundry package version used for CLI metadata.

    Returns
    -------
    str
        Detected ``kgfoundry`` package version, or ``"0.0.0"`` when unavailable.
    """
    try:
        return pkg_version("kgfoundry")
    except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
        return "0.0.0"


@lru_cache(maxsize=1)
def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry paths.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing the shared augment and registry metadata.
    """
    return CLIToolSettings(
        bin_name="kgf",
        title=CLI_TITLE,
        version=_resolve_cli_version(),
        augment_path=REPO_ROOT / "openapi" / "_augment_cli.yaml",
        registry_path=REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml",
        interface_id=CLI_INTERFACE_ID,
    )


@lru_cache(maxsize=1)
def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for the download interface.

    Returns
    -------
    CLIToolingContext
        Tooling context bundling augment, registry, and CLI configuration models.
    """
    return load_cli_tooling_context(get_cli_settings())


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration extracted from the tooling context.

    Returns
    -------
    CLIConfig
        CLI configuration providing access to the operation context helper.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the operation context used for OpenAPI and diagram generation.

    Returns
    -------
    OperationContext
        Operation context derived from the CLI configuration.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment and registry metadata bundle.

    Returns
    -------
    ToolingMetadataModel
        Immutable metadata bundle combining augment and registry models.
    """
    return ToolingMetadataModel(augment=get_augment_metadata(), registry=get_registry_metadata())


def get_augment_metadata() -> AugmentMetadataModel:
    """Return the augment metadata model for the download CLI.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata describing operations, tag groups, and extras.
    """
    return get_cli_context().augment


def get_registry_metadata() -> RegistryMetadataModel:
    """Return the registry metadata model backing the download CLI.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata containing typed interface entries.
    """
    return get_cli_context().registry


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return interface metadata for the configured download CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata defined in the registry for ``CLI_INTERFACE_ID``.

    Raises
    ------
    KeyError
        Raised when the registry metadata does not contain the configured interface.
    """
    interface = get_registry_metadata().interface(CLI_INTERFACE_ID)
    if interface is None:  # pragma: no cover - misconfiguration guarded by type checks
        msg = f"Registry metadata missing interface '{CLI_INTERFACE_ID}'."
        raise KeyError(msg)
    return interface


def get_operation_override(
    subcommand: str, *, tokens: Sequence[str] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for ``subcommand`` when available.

    Parameters
    ----------
    subcommand : str
        CLI subcommand name to resolve.
    tokens : Sequence[str] | None, optional
        Optional token sequence used for fallback matching when a direct operation ID
        mapping does not exist.

    Returns
    -------
    OperationOverrideModel | None
        Augment override model when defined; otherwise ``None``.
    """
    operation_id = CLI_OPERATION_IDS.get(subcommand)
    if operation_id is None:
        return None
    return get_augment_metadata().operation_override(operation_id, tokens=tokens)


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
