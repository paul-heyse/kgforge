#!/usr/bin/env python3
"""Check for new type ignores and noqa directives without TICKET: tags.

This script scans source files for new `# type: ignore` or `# noqa` directives
that don't include a `TICKET:` tag. It fails the build if untracked suppressions
are found, enforcing zero-tolerance for unmanaged suppressions.

Examples
--------
>>> python tools/check_new_suppressions.py src
# Exits 0 if all suppressions have TICKET: tags
# Exits 1 with detailed output if untracked suppressions found
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)

# Pattern to match type ignore or noqa with optional ticket tag
SUPPRESSION_PATTERN = re.compile(
    r"(?:#\s*(?:type:\s*ignore|noqa(?::\s*[\w,]+)?))(?:\s*#\s*(?:.*?TICKET:.*?))?$",
    re.MULTILINE,
)

TICKET_PATTERN = re.compile(r"TICKET:\s*\S+", re.IGNORECASE)


def check_file(file_path: Path) -> list[tuple[int, str]]:
    """Check a single file for untracked suppressions.

    Parameters
    ----------
    file_path : Path
        Path to the Python file to check.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, line_content) tuples for lines with untracked suppressions.
    """
    violations: list[tuple[int, str]] = []
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, start=1):
            # Check for type: ignore or noqa
            if ("# type: ignore" in line or "# noqa" in line) and not TICKET_PATTERN.search(line):
                violations.append((line_num, line.strip()))

    except (OSError, UnicodeDecodeError) as exc:
        # If we can't read the file, report but don't fail (may be binary)
        LOGGER.warning("Could not read %s: %s", file_path, exc)
        return []

    return violations


def check_directory(directory: Path) -> dict[Path, list[tuple[int, str]]]:
    """Check all Python files in a directory for untracked suppressions.

    Parameters
    ----------
    directory : Path
        Directory to scan for Python files.

    Returns
    -------
    dict[Path, list[tuple[int, str]]]
        Dictionary mapping file paths to lists of violations.
    """
    violations: dict[Path, list[tuple[int, str]]] = {}

    for py_file in directory.rglob("*.py"):
        file_violations = check_file(py_file)
        if file_violations:
            violations[py_file] = file_violations

    return violations


def main() -> int:
    """Check source files for untracked suppressions.

    Returns
    -------
    int
        Exit code: 0 if all suppressions tracked, 1 if violations found.
    """
    if len(sys.argv) < 2:
        LOGGER.error("Usage: python tools/check_new_suppressions.py <directory>")
        return 1

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        LOGGER.error("Directory %s does not exist", target_dir)
        return 1

    violations = check_directory(target_dir)

    if not violations:
        LOGGER.info("✅ All suppressions include TICKET: tags")
        return 0

    LOGGER.error("❌ Found suppressions without TICKET: tags:")
    LOGGER.error("")

    cwd = Path.cwd()
    for file_path, file_violations in sorted(violations.items()):
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            rel_path = file_path
        LOGGER.error("%s:", rel_path)
        for line_num, line_content in file_violations:
            LOGGER.error("  Line %s: %s", line_num, line_content)
        LOGGER.error("")

    LOGGER.error("Fix: Add TICKET: <ticket-id> to each suppression line.")
    LOGGER.error("Example: # type: ignore[misc]  # TICKET: ABC-123  # numpy dtype contains Any")

    return 1


if __name__ == "__main__":
    sys.exit(main())
