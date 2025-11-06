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


CLI_COMMAND = "codeintel"
"""Human-friendly command label used for envelopes and logging."""


CLI_TITLE = "KGFoundry CodeIntel Indexer"
"""Title surfaced in Typer help text and OpenAPI metadata."""


CLI_INTERFACE_ID = "codeintel-indexer"
"""Registry interface identifier for the code-intel indexer CLI."""


CLI_OPERATION_IDS: dict[str, str] = {
    "query": "cli.codeintel.query",
    "symbols": "cli.codeintel.symbols",
}
"""Mapping of subcommand names to canonical augment/registry operation IDs."""


def _resolve_cli_version() -> str:
    """Return the installed package version backing the CLI metadata.

    Returns
    -------
    str
        Detected version string or ``"0.0.0"`` when the package is unavailable.
    """
    for distribution in ("kgfoundry-codeintel", "kgfoundry"):
        try:
            return pkg_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - editable installs fallback
            continue
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
        bin_name="kgf-codeintel",
        title=CLI_TITLE,
        version=_resolve_cli_version(),
        augment_path=REPO_ROOT / "openapi" / "_augment_cli.yaml",
        registry_path=REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml",
        interface_id=CLI_INTERFACE_ID,
    )


@lru_cache(maxsize=1)
def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for the code-intel CLI.

    Returns
    -------
    CLIToolingContext
        Composite context bundling augment, registry, and configuration data.
    """
    return load_cli_tooling_context(get_cli_settings())


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
    context = get_cli_context()
    return ToolingMetadataModel(augment=context.augment, registry=context.registry)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return augment metadata describing code-intel CLI operations.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata model scoped to the code-intel CLI operations.
    """
    return get_cli_context().augment


def get_registry_metadata() -> RegistryMetadataModel:
    """Return registry metadata backing the code-intel CLI interface.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata describing the code-intel CLI interface definition.
    """
    return get_cli_context().registry


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return registry interface metadata for the code-intel CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata linked to the code-intel CLI.

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
    """Return augment override metadata for ``subcommand`` when defined.

    Returns
    -------
    OperationOverrideModel | None
        Operation override metadata when present; otherwise ``None``.
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
