#!/usr/bin/env python3
"""Run tooling architecture checks built on pytestarch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import architecture
from tools._shared.cli import CliEnvelope, CliEnvelopeBuilder, CliStatus, render_cli_envelope
from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)


def _build_envelope(result: architecture.ArchitectureResult) -> CliEnvelope:
    status: CliStatus = "success" if result.is_success else "violation"
    builder = CliEnvelopeBuilder.create(command="check_imports", status=status)
    if not result.is_success:
        for violation in result.violations:
            builder.add_error(status="violation", message=violation)
    return builder.finish()


def main() -> int:
    """Execute the tooling architecture checks and emit a CLI envelope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write a JSON envelope to stdout with detailed violations.",
    )
    args: argparse.Namespace = parser.parse_args()

    result = architecture.enforce_tooling_layers()
    envelope = _build_envelope(result)

    output_json = cast(bool, getattr(args, "json", False))

    if output_json:
        sys.stdout.write(render_cli_envelope(envelope) + "\n")
    elif result.is_success:
        LOGGER.info("Tooling import layering verified (domain → adapters → io/cli).")
    else:
        for violation in result.violations:
            LOGGER.error(violation)

    return 0 if result.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
