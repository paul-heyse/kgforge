#!/usr/bin/env python3
"""Check for new type ignores and noqa directives without TICKET: tags.

This script scans source files for new `# type-ignore` or `# noqa` directives
that don't include a `TICKET:` tag. It fails the build if untracked suppressions
are found, enforcing zero-tolerance for unmanaged suppressions.

Examples
--------
>>> python tools/check_new_suppressions.py src
# Exits 0 if all suppressions have TICKET: tags
# Exits 1 with detailed output if untracked suppressions found
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kgfoundry_common.errors import ConfigurationError
from tools._shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable


LOGGER = get_logger(__name__)

# Regex patterns for suppression markers and ticket tags within a comment token.
SUPPRESSION_MARKER = re.compile(
    r"#\s*(?:type\s*:\s*ignore|noqa(?:\s*[:\s][\w,]+)?)",
    re.IGNORECASE,
)
TICKET_PATTERN = re.compile(r"#\s*TICKET:\s*\S+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SuppressionViolation:
    """Record representing a suppression comment missing a ``TICKET:`` tag."""

    path: Path
    line_number: int
    line_preview: str


def _iter_suppression_comments(content: str) -> Iterable[tuple[int, str]]:
    """Yield ``(line_number, comment)`` pairs for suppression comments."""
    reader = io.StringIO(content).readline
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.COMMENT:
            continue
        comment = token.string
        if SUPPRESSION_MARKER.search(comment):
            yield token.start[0], comment


def check_file(file_path: Path) -> list[SuppressionViolation]:
    """Return suppressions lacking ``TICKET:`` metadata within ``file_path``."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        LOGGER.warning("Could not read %s", file_path, exc_info=exc)
        return []

    lines = content.splitlines()
    violations: list[SuppressionViolation] = []

    for line_number, comment in _iter_suppression_comments(content):
        if TICKET_PATTERN.search(comment):
            continue

        preview = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else ""
        violations.append(
            SuppressionViolation(path=file_path, line_number=line_number, line_preview=preview)
        )

    return violations


def check_directory(directory: Path) -> dict[Path, list[SuppressionViolation]]:
    """Return per-file suppressions lacking ``TICKET:`` metadata."""
    violations: dict[Path, list[SuppressionViolation]] = {}

    for py_file in sorted(directory.rglob("*.py")):
        file_violations = check_file(py_file)
        if file_violations:
            violations[py_file] = file_violations

    return violations


def _resolve_target_directories(paths: list[str]) -> list[Path]:
    """Resolve and validate target directories.

    Parameters
    ----------
    paths : list[str]
        List of directory paths to resolve.

    Returns
    -------
    list[Path]
        List of resolved Path objects.

    Raises
    ------
    ConfigurationError
        If any path doesn't exist or is not a directory.
    """
    resolved: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            msg = f"Directory does not exist: {path_str}"
            raise ConfigurationError(msg)
        if not path.is_dir():
            msg = f"Path is not a directory: {path_str}"
            raise ConfigurationError(msg)
        resolved.append(path.resolve())
    return resolved


def run_suppression_guard(directories: list[Path]) -> None:
    """Run suppression guard on the given directories.

    Parameters
    ----------
    directories : list[Path]
        List of directories to check.

    Raises
    ------
    ConfigurationError
        If any violations are found. The error context contains violation details.
    """
    violations: dict[Path, list[SuppressionViolation]] = {}
    for directory in directories:
        dir_violations = check_directory(directory)
        violations.update(dir_violations)

    if violations:
        violation_count = sum(len(v) for v in violations.values())
        files = [
            {
                "file": str(file_path),
                "violations": [
                    {
                        "line": v.line_number,
                        "preview": v.line_preview,
                    }
                    for v in file_violations
                ],
            }
            for file_path, file_violations in sorted(violations.items())
        ]
        message = f"Found {violation_count} suppression(s) without TICKET: tags"
        raise ConfigurationError(
            message,
            context={
                "violation_count": violation_count,
                "files": files,
            },
        )


def main() -> int:
    """Check source files for untracked suppressions."""
    if len(sys.argv) < 2:
        LOGGER.error("Usage: python tools/check_new_suppressions.py <directory>")
        return 1

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        LOGGER.error("Directory %s does not exist", target_dir)
        return 1

    violations = check_directory(target_dir)

    if not violations:
        LOGGER.info("✅ All suppressions include TICKET: tags", extra={"path": str(target_dir)})
        return 0

    LOGGER.error(
        "❌ Found suppressions without TICKET: tags",
        extra={
            "path": str(target_dir),
            "violation_count": sum(len(v) for v in violations.values()),
        },
    )

    cwd = Path.cwd()
    for file_path, file_violations in sorted(violations.items()):
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            rel_path = file_path
        LOGGER.error("%s:", rel_path, extra={"path": str(rel_path)})
        for violation in file_violations:
            LOGGER.error(
                "  Line %s: %s",
                violation.line_number,
                violation.line_preview,
                extra={
                    "path": str(rel_path),
                    "line": violation.line_number,
                    "line_preview": violation.line_preview,
                },
            )

    LOGGER.error("Fix: Add TICKET: <ticket-id> to each suppression line.")
    LOGGER.error("Example: # type-ignore[misc]  # TICKET: ABC-123  # numpy dtype contains Any")

    return 1


if __name__ == "__main__":
    sys.exit(main())
