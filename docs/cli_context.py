"""Public shim exposing documentation CLI context helpers."""

from __future__ import annotations

from docs._scripts.cli_context import (
    CLI_COMMAND,
    CLI_INTERFACE_ID,
    CLI_OPERATION_IDS,
    CLI_TITLE,
    get_augment_metadata,
    get_cli_config,
    get_cli_context,
    get_cli_definition,
    get_cli_settings,
    get_interface_metadata,
    get_operation_context,
    get_operation_override,
    get_registry_metadata,
    get_tooling_metadata,
)

__all__ = [
    "CLI_COMMAND",
    "CLI_INTERFACE_ID",
    "CLI_OPERATION_IDS",
    "CLI_TITLE",
    "get_augment_metadata",
    "get_cli_config",
    "get_cli_context",
    "get_cli_definition",
    "get_cli_settings",
    "get_interface_metadata",
    "get_operation_context",
    "get_operation_override",
    "get_registry_metadata",
    "get_tooling_metadata",
]
