"""Public re-export of the CLI context registry helpers."""

from __future__ import annotations

from tools._shared.cli_context_registry import (
    REGISTRY,
    CLIContextDefinition,
    augment_for,
    context_for,
    default_version_resolver,
    definition_for,
    interface_for,
    operation_override_for,
    register_cli,
    registry_for,
    settings_for,
    tooling_metadata_for,
)

__all__ = [
    "REGISTRY",
    "CLIContextDefinition",
    "augment_for",
    "context_for",
    "default_version_resolver",
    "definition_for",
    "interface_for",
    "operation_override_for",
    "register_cli",
    "registry_for",
    "settings_for",
    "tooling_metadata_for",
]
