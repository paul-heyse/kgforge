"""Public wrapper for :mod:`tools._shared.cli`."""

from __future__ import annotations

from tools._shared.cli import (
    CLI_ENVELOPE_SCHEMA,
    CLI_ENVELOPE_SCHEMA_ID,
    CLI_ENVELOPE_SCHEMA_VERSION,
    CliEnvelope,
    CliEnvelopeBuilder,
    CliErrorEntry,
    CliErrorStatus,
    CliFileResult,
    CliFileStatus,
    CliStatus,
    new_cli_envelope,
    render_cli_envelope,
    validate_cli_envelope,
)

__all__: tuple[str, ...] = (
    "CLI_ENVELOPE_SCHEMA",
    "CLI_ENVELOPE_SCHEMA_ID",
    "CLI_ENVELOPE_SCHEMA_VERSION",
    "CliEnvelope",
    "CliEnvelopeBuilder",
    "CliErrorEntry",
    "CliErrorStatus",
    "CliFileResult",
    "CliFileStatus",
    "CliStatus",
    "new_cli_envelope",
    "render_cli_envelope",
    "validate_cli_envelope",
)
