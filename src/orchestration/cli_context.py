"""Shared CLI context for the orchestration command suite.

This module mirrors the shared tooling helpers introduced for other CLIs
(`download`, `tools.docstring_builder`) so orchestration commands can load typed
augment and registry metadata without duplicating configuration logic. The
helpers expose cached accessors for the `CLIToolSettings`, `CLIToolingContext`,
and associated metadata models consumed by downstream tooling (OpenAPI
generation, MkDocs diagrams, navmap loader, etc.).
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
"""Repository root, used to locate augment and registry metadata."""


CLI_COMMAND = "orchestration"
"""Root command label used for CLI envelopes and logging."""


CLI_TITLE = "KGFoundry Orchestration"
"""Human-readable CLI title consumed by Typer help text and OpenAPI metadata."""


CLI_INTERFACE_ID = "orchestration-cli"
"""Registry interface identifier for the orchestration CLI suite."""


CLI_OPERATION_IDS: dict[str, str] = {
    "index-bm25": "cli.index_bm25",
    "index_bm25": "cli.index_bm25",
    "index-faiss": "cli.index_faiss",
    "index_faiss": "cli.index_faiss",
    "api": "cli.api",
    "e2e": "cli.e2e",
}
"""Mapping of subcommand names to canonical operation identifiers."""


def _resolve_cli_version() -> str:
    """Return the installed kgfoundry package version for CLI metadata.

    Returns
    -------
    str
        Installed ``kgfoundry`` version or ``"0.0.0"`` when unavailable.
    """
    try:
        return pkg_version("kgfoundry")
    except PackageNotFoundError:  # pragma: no cover - editable installs fallback
        return "0.0.0"


@lru_cache(maxsize=1)
def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry metadata inputs.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing augment and registry metadata paths.
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
    """Return the cached CLI tooling context for orchestration commands.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment, registry, and CLI config.
    """
    return load_cli_tooling_context(get_cli_settings())


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration extracted from the tooling context.

    Returns
    -------
    CLIConfig
        Typed configuration consumed by CLI generators.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the operation context helper used for OpenAPI generation.

    Returns
    -------
    OperationContext
        Helper used by tooling to resolve operation metadata.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle for orchestration.

    Returns
    -------
    ToolingMetadataModel
        Immutable bundle combining augment and registry metadata.
    """
    context = get_cli_context()
    return ToolingMetadataModel(augment=context.augment, registry=context.registry)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata for orchestration operations.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata describing orchestration CLI operations.
    """
    return get_cli_context().augment


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata backing the orchestration CLI interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata describing orchestration interfaces.
    """
    return get_cli_context().registry


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return registry interface metadata for the orchestration CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata linked to the orchestration CLI.

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
    subcommand: str, *, tokens: Sequence[str] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for the given subcommand when available.

    Returns
    -------
    OperationOverrideModel | None
        Augment override metadata when defined; otherwise ``None``.
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
