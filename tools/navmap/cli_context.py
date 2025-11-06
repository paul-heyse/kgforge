"""Shared CLI context for the navmap tooling command suite.

This module mirrors the shared CLI configuration helpers adopted across other
CLIs (download, orchestration, codeintel). It exposes cached accessors for the
canonical :class:`~tools.CLIToolSettings`, :class:`~tools.CLIToolingContext`, and
typed augment/registry metadata models so the navmap tooling consumes the same
facade as the rest of the system.
"""

from __future__ import annotations

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
"""Repository root used to resolve augment and registry metadata paths."""


CLI_COMMAND = "navmap"
"""Command label used for CLI envelopes and structured logging."""


CLI_TITLE = "KGFoundry Navmap Builder"
"""Human-readable CLI title used by help text and metadata."""


CLI_INTERFACE_ID = "navmap-cli"
"""Registry interface identifier for the navmap tooling CLI."""


CLI_OPERATION_IDS: dict[str, str] = {
    "build": "navmap.build",
}
"""Mapping of subcommand names to canonical operation identifiers."""


def _resolve_cli_version() -> str:
    """Return the installed package version associated with the CLI metadata."""
    for distribution in ("kgfoundry-tools", "kgfoundry"):
        try:
            return pkg_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - editable installs fallback
            continue
    return "0.0.0"


@lru_cache(maxsize=1)
def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry metadata inputs."""
    return CLIToolSettings(
        bin_name="tools-navmap",
        title=CLI_TITLE,
        version=_resolve_cli_version(),
        augment_path=REPO_ROOT / "openapi" / "_augment_cli.yaml",
        registry_path=REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml",
        interface_id=CLI_INTERFACE_ID,
    )


@lru_cache(maxsize=1)
def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for navmap operations."""
    return load_cli_tooling_context(get_cli_settings())


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration extracted from the tooling context."""
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the shared operation context helper used for metadata lookups."""
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle."""
    context = get_cli_context()
    return ToolingMetadataModel(augment=context.augment, registry=context.registry)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata describing navmap CLI operation overrides."""
    return get_cli_context().augment


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata backing the navmap CLI interface."""
    return get_cli_context().registry


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return the registry interface metadata for the navmap CLI.

    Raises
    ------
    KeyError
        Raised when the registry metadata does not contain the interface.
    """
    interface = get_registry_metadata().interface(CLI_INTERFACE_ID)
    if interface is None:  # pragma: no cover - misconfiguration guard
        msg = f"Registry metadata missing interface '{CLI_INTERFACE_ID}'."
        raise KeyError(msg)
    return interface


def get_operation_override(
    subcommand: str, *, tokens: tuple[str, ...] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for the given subcommand when present."""
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
