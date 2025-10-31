#!/usr/bin/env python3
"""Wrapper script to run import-linter checks.

This script provides a consistent way to run import-linter checks
across different environments (local, CI, pre-commit).
"""

from __future__ import annotations

import sys
from pathlib import Path

from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)

try:
    from importlinter.application.use_cases import create_report, read_user_options
except ImportError:
    LOGGER.exception("import-linter not installed")
    LOGGER.info("Install with: uv sync")
    sys.exit(1)


def main() -> int:
    """Run import-linter checks.

    Returns
    -------
    int
        Exit code: 0 if all contracts pass, 1 if violations found.
    """
    config_path = Path("importlinter.cfg")
    if not config_path.exists():
        LOGGER.error("Config file not found: %s", config_path)
        return 1

    try:
        # Read configuration and create report
        user_options = read_user_options(config_filename=str(config_path))
        report = create_report(user_options=user_options)
    except Exception:
        LOGGER.exception("Error running import-linter")
        return 1
    else:
        if report.is_contracts_satisfied():
            LOGGER.info("✅ All import contracts satisfied")
            return 0
        # Print violations
        LOGGER.error("❌ Import contract violations found:")
        for contract in report.contracts:
            if not contract.is_kept():
                LOGGER.error("  - %s: %s", contract.name, contract.kept)
        return 1


if __name__ == "__main__":
    sys.exit(main())
