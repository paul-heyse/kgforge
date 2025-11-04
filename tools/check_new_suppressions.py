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
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict, cast

from kgfoundry_common.errors import ConfigurationError
from tools._shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LOGGER = get_logger(__name__)

# Public exports for static type checkers and consumers.
__all__ = (
    "SuppressionGuardContext",
    "SuppressionGuardFileEntry",
    "SuppressionGuardReport",
    "SuppressionGuardViolationEntry",
    "SuppressionViolation",
    "build_guard_context",
    "check_directory",
    "check_file",
    "main",
    "resolve_target_directories",
    "run_suppression_guard",
)

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


class SuppressionGuardViolationEntry(TypedDict):
    """Payload describing a single suppression violation."""

    line: int
    preview: str


class SuppressionGuardFileEntry(TypedDict):
    """Collection of violations for a particular file."""

    file: str
    violations: list[SuppressionGuardViolationEntry]


class SuppressionGuardContext(TypedDict):
    """Problem Details context for suppression guard failures."""

    violation_count: int
    files: list[SuppressionGuardFileEntry]


@dataclass(frozen=True, slots=True)
class SuppressionGuardReport:
    """Aggregate results for a suppression guard run."""

    violations: Mapping[Path, tuple[SuppressionViolation, ...]]

    def __post_init__(self) -> None:
        """Freeze the violation mapping to an immutable proxy."""
        normalized: dict[Path, tuple[SuppressionViolation, ...]] = {
            path: tuple(entries) for path, entries in self.violations.items()
        }
        proxy = MappingProxyType(normalized)
        object.__setattr__(
            self,
            "violations",
            cast("Mapping[Path, tuple[SuppressionViolation, ...]]", proxy),
        )

    @property
    def violation_count(self) -> int:
        """Return the total number of violations discovered."""
        return sum(len(entries) for entries in self.violations.values())

    @property
    def is_clean(self) -> bool:
        """Return ``True`` when the report has no violations."""
        return self.violation_count == 0

    def to_context(self) -> SuppressionGuardContext:
        """Render the report as structured problem details context."""
        return build_guard_context(self)


def _iter_suppression_comments(content: str) -> Sequence[tuple[int, str]]:
    """Yield ``(line_number, comment)`` pairs for suppression comments."""
    reader = io.StringIO(content).readline
    matches: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.COMMENT:
            continue
        comment = token.string
        if SUPPRESSION_MARKER.search(comment):
            matches.append((token.start[0], comment))
    return tuple(matches)


def check_file(file_path: Path) -> tuple[SuppressionViolation, ...]:
    """Return suppressions lacking ``TICKET:`` metadata within ``file_path``."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        LOGGER.warning("Could not read %s", file_path, exc_info=exc)
        return ()

    lines = content.splitlines()
    violations: list[SuppressionViolation] = []

    for line_number, comment in _iter_suppression_comments(content):
        if TICKET_PATTERN.search(comment):
            continue

        preview = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else ""
        violations.append(
            SuppressionViolation(path=file_path, line_number=line_number, line_preview=preview)
        )

    return tuple(violations)


def check_directory(directory: Path) -> SuppressionGuardReport:
    """Return per-file suppressions lacking ``TICKET:`` metadata."""
    violations: dict[Path, tuple[SuppressionViolation, ...]] = {}

    for py_file in sorted(directory.rglob("*.py")):
        file_violations = check_file(py_file)
        if file_violations:
            violations[py_file] = file_violations

    return SuppressionGuardReport(violations=violations)


def resolve_target_directories(paths: Sequence[str]) -> list[Path]:
    """Resolve and validate target directories.

    Parameters
    ----------
    paths : Sequence[str]
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


def run_suppression_guard(directories: Sequence[Path]) -> SuppressionGuardReport:
    """Run suppression guard on the given directories."""
    aggregated: dict[Path, tuple[SuppressionViolation, ...]] = {}
    for directory in directories:
        report = check_directory(directory)
        aggregated.update(report.violations)

    final_report = SuppressionGuardReport(violations=aggregated)

    if not final_report.is_clean:
        message = f"Found {final_report.violation_count} suppression(s) without TICKET: tags"
        raise ConfigurationError(message, context=final_report.to_context())

    return final_report


def build_guard_context(report: SuppressionGuardReport) -> SuppressionGuardContext:
    """Construct structured context payload for Problem Details reporting."""
    files: list[SuppressionGuardFileEntry] = []
    for file_path, file_violations in sorted(report.violations.items()):
        files.append(
            {
                "file": str(file_path),
                "violations": [
                    {
                        "line": violation.line_number,
                        "preview": violation.line_preview,
                    }
                    for violation in file_violations
                ],
            }
        )

    return {
        "violation_count": report.violation_count,
        "files": files,
    }


def _extract_guard_context(error: ConfigurationError) -> SuppressionGuardContext | None:
    """Normalize the context payload attached to a suppression guard error."""
    context = dict(error.context or {})
    expected_keys = {"violation_count", "files"}
    if not expected_keys.issubset(context):
        return None
    return cast("SuppressionGuardContext", context)


def main(argv: Sequence[str] | None = None) -> int:
    """Check source files for untracked suppressions."""
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        LOGGER.error("Usage: python tools/check_new_suppressions.py <directories...>")
        return 1

    try:
        directories = resolve_target_directories(arguments)
    except ConfigurationError:
        LOGGER.exception("Failed to resolve suppression guard target directories")
        return 1

    report_context: SuppressionGuardContext
    try:
        run_suppression_guard(directories)
    except ConfigurationError as error:
        extracted_context = _extract_guard_context(error)
        if extracted_context is None:
            LOGGER.exception("❌ Found suppressions without TICKET: tags")
            return 1
        report_context = extracted_context
    else:
        LOGGER.info("✅ All suppressions include TICKET: tags")
        return 0

    LOGGER.error(
        "❌ Found suppressions without TICKET: tags",
        extra={"violation_count": report_context.get("violation_count", 0)},
    )

    cwd = Path.cwd()
    for file_entry in report_context.get("files", []):
        file_path = Path(file_entry["file"])
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            rel_path = file_path
        LOGGER.error("%s:", rel_path, extra={"path": str(rel_path)})
        for violation in file_entry["violations"]:
            LOGGER.error(
                "  Line %s: %s",
                violation["line"],
                violation["preview"],
                extra={
                    "path": str(rel_path),
                    "line": violation["line"],
                    "line_preview": violation["preview"],
                },
            )

    LOGGER.error("Fix: Add TICKET: <ticket-id> to each suppression line.")
    LOGGER.error("Example: # type-ignore[misc]  # TICKET: ABC-123  # numpy dtype contains Any")
    return 1


if __name__ == "__main__":
    sys.exit(main())
