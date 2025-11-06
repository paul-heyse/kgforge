"""Shared CLI context helpers for the docstring builder tooling suite.

The canonical CLI metadata (augment + registry) powers downstream tooling
including OpenAPI generation, CLI diagrams, and automation scripts. This module
exposes lightweight accessors so external integrations—such as
``tools/generate_docstrings.py``—can load the same typed metadata without
importing the heavy Typer CLI entry point.
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


CLI_COMMAND = "docstrings"
"""Logical command name used for envelopes and documentation artifacts."""


CLI_TITLE = "Docstring Builder CLI"
"""Human-readable CLI title sourced from registry metadata."""


CLI_INTERFACE_ID = "docstring-builder-cli"
"""Registry interface identifier for the docstring builder CLI."""


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


def _resolve_cli_version() -> str:
    """Return the version string advertised for the CLI metadata.

    Returns
    -------
    str
        Installed package version or ``"0.0.0"`` when unknown.
    """
    for distribution in ("kgfoundry-tools", "kgfoundry"):
        try:
            return pkg_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - editable installs fallback
            continue
    return "0.0.0"


@lru_cache(maxsize=1)
def get_cli_settings() -> CLIToolSettings:
    """Return canonical CLI settings for the docstring builder suite.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing augment and registry metadata.
    """
    return CLIToolSettings(
        bin_name="docstring-builder",
        title=CLI_TITLE,
        version=_resolve_cli_version(),
        augment_path=REPO_ROOT / "openapi" / "_augment_cli.yaml",
        registry_path=REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml",
        interface_id=CLI_INTERFACE_ID,
    )


@lru_cache(maxsize=1)
def get_cli_context() -> CLIToolingContext:
    """Return the lazily loaded CLI tooling context.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment and registry metadata.
    """
    return load_cli_tooling_context(get_cli_settings())


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
    context = get_cli_context()
    return ToolingMetadataModel(augment=context.augment, registry=context.registry)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata for docstring builder operations.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata scoped to docstring builder commands.
    """
    return get_cli_context().augment


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata describing the docstring builder interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata containing interface definitions for the CLI.
    """
    return get_cli_context().registry


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return the registry interface metadata for the docstring builder CLI.

    Returns
    -------
    RegistryInterfaceModel
        Registry interface entry associated with the CLI.

    Raises
    ------
    KeyError
        Raised when the registry metadata omits the expected interface.
    """
    interface = get_registry_metadata().interface(CLI_INTERFACE_ID)
    if interface is None:  # pragma: no cover - defensive guard for misconfiguration
        msg = f"Registry metadata missing interface '{CLI_INTERFACE_ID}'."
        raise KeyError(msg)
    return interface


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
