#!/usr/bin/env python3
"""Validate Sphinx-Gallery examples used throughout the kgfoundry docs."""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Final, cast

from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)

TITLE_MAX_LENGTH = 79
UNDERLINE_TOLERANCE = 1
MIN_LINES_WITH_UNDERLINE: Final[int] = 2
BLANK_LINE_INDEX: Final[int] = 2

TITLE_UNDERLINE_PATTERN: Final[Pattern[str]] = re.compile(r"^(?P<char>=)\1*$")
CUSTOM_LABEL_PATTERN: Final[Pattern[str]] = re.compile(r"(?m)^\.\.\s+_gallery_[\w-]+:\s*$")
TAGS_PATTERN: Final[Pattern[str]] = re.compile(r"(?m)^\.\.\s+tags::\s*")
CONSTRAINTS_HEADER_PATTERN: Final[Pattern[str]] = re.compile(
    r"(?m)^Constraints\s*\n(?P<rule>[-=~`'^\"]{3,})\s*$",
)

__navmap__ = {
    "category": "docs",
    "stability": "stable",
    "exports": ["main", "validate_example_file"],
    "synopsis": "Validate Sphinx-Gallery examples for title and directive compliance.",
}

__all__ = [
    "GalleryValidationError",
    "check_custom_labels",
    "check_orphan_directive",
    "main",
    "validate_example_file",
    "validate_title_format",
]


@dataclass
class ValidationResult:
    """Accumulate validation errors for a single gallery example file."""

    path: Path
    errors: list[str]

    def extend(self, messages: Iterable[str]) -> None:
        """Append ``messages`` to the collected validation ``errors`` list."""
        self.errors.extend(messages)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no validation errors have been recorded."""
        return not self.errors


class GalleryValidationError(RuntimeError):
    """Raised when parsing or validation cannot proceed for a gallery example."""


@dataclass(slots=True)
class GalleryOptions:
    """CLI options extracted from argument parsing."""

    examples_dir: Path
    strict: bool
    verbose: bool
    fix: bool


def validate_title_format(docstring: str) -> tuple[bool, str]:
    """Validate Sphinx-Gallery title and underline formatting requirements.

    Parameters
    ----------
    docstring
        Raw module docstring extracted from a gallery example.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the title is well-formed, otherwise ``(False, reason)``.
    """
    lines = [line.rstrip() for line in inspect.cleandoc(docstring).splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)

    failure: str | None = None
    if not lines:
        failure = "docstring is empty"
    else:
        title = lines[0]
        if len(title) > TITLE_MAX_LENGTH:
            failure = f"title exceeds {TITLE_MAX_LENGTH} characters"
        elif len(lines) < MIN_LINES_WITH_UNDERLINE:
            failure = "missing underline under the title"
        else:
            underline = lines[1]
            if not TITLE_UNDERLINE_PATTERN.fullmatch(underline):
                failure = "title underline must be composed of '=' characters"
            elif abs(len(underline) - len(title)) > UNDERLINE_TOLERANCE:
                failure = "title underline length must match the title (±1 character)"
            elif len(lines) <= BLANK_LINE_INDEX or lines[BLANK_LINE_INDEX].strip():
                failure = "expected a blank line after the title underline"

    if failure is not None:
        return False, failure
    return True, ""


def check_orphan_directive(docstring: str) -> bool:
    """Return ``True`` when the docstring contains a redundant ``:orphan:`` directive."""
    return ":orphan:" in docstring


def check_custom_labels(docstring: str) -> list[str]:
    """Return custom anchor labels declared in ``docstring``.

    Sphinx-Gallery generates its own anchors; any ``.. _gallery_*:`` labels
    should be removed to avoid duplicates.
    """
    return cast(list[str], CUSTOM_LABEL_PATTERN.findall(docstring))


def _has_tags_directive(docstring: str) -> bool:
    """Return ``True`` if the docstring declares a ``..

    tags::`` directive.
    """
    return TAGS_PATTERN.search(docstring) is not None


def _has_constraints_section(docstring: str) -> bool:
    """Return ``True`` if a ``Constraints`` section header is present."""
    match = CONSTRAINTS_HEADER_PATTERN.search(docstring)
    if not match:
        return False
    rule = match.group("rule")
    if not isinstance(rule, str):
        return False
    return set(rule) == {"-"}


def _load_docstring(path: Path) -> str | None:
    """Extract the module docstring from ``path`` using ``ast`` parsing."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - should never happen for examples
        message = f"{path}: failed to parse module ({exc})"
        raise GalleryValidationError(message) from exc
    return ast.get_docstring(module, clean=False)


def validate_example_file(file_path: Path, *, strict: bool = False) -> list[str]:
    """Validate gallery example formatting and return a list of issues.

    Parameters
    ----------
    file_path
        Path to the Python example file.
    strict
        When ``True`` enable extra formatting checks (title punctuation,
        constraints bullets).

    Returns
    -------
    list[str]
        Human-readable validation errors. The list is empty when the example
        satisfies all rules.
    """
    errors: list[str] = []
    docstring = _load_docstring(file_path)
    if docstring is None:
        return ["missing module docstring"]

    ok, message = validate_title_format(docstring)
    if not ok:
        errors.append(message)

    if check_orphan_directive(docstring):
        errors.append("remove ':orphan:' directive (Sphinx-Gallery manages references)")

    labels = check_custom_labels(docstring)
    if labels:
        errors.append("remove custom '.. _gallery_*:' labels (Sphinx-Gallery generates them)")

    if not _has_tags_directive(docstring):
        errors.append("add a '.. tags::' directive describing the example")

    if not _has_constraints_section(docstring):
        errors.append("add a 'Constraints' section with a dashed underline")

    if strict:
        lines = inspect.cleandoc(docstring).splitlines()
        if lines and lines[0].endswith("."):
            errors.append("remove trailing period from the title")
        if sum(1 for line in lines if line.strip().startswith("- ")) < 1:
            errors.append("constraints section should enumerate at least one bullet")

    return errors


def _iter_example_files(examples_dir: Path) -> Iterable[Path]:
    """Yield Python files inside ``examples_dir`` (non-recursive)."""
    for path in sorted(examples_dir.glob("*.py")):
        if path.name.startswith("."):
            continue
        yield path


def main(examples_dir: Path, *, strict: bool = False, verbose: bool = False) -> int:
    """Validate every example in ``examples_dir`` and emit a summary.

    Parameters
    ----------
    examples_dir
        Directory containing Sphinx-Gallery example modules.
    strict
        When ``True`` enable the stricter validation rules.
    verbose
        When ``True`` print progress for passing files.

    Returns
    -------
    int
        ``0`` when all files pass, ``1`` when any fail, ``2`` for CLI errors.
    """
    results: list[ValidationResult] = []
    exit_code = 0
    for path in _iter_example_files(examples_dir):
        messages = validate_example_file(path, strict=strict)
        result = ValidationResult(path=path, errors=messages)
        results.append(result)
        if result.ok:
            if verbose:
                LOGGER.info("✔ %s", path)
            continue
        exit_code = 1
        for message in result.errors:
            LOGGER.error("✖ %s: %s", path, message)

    if exit_code == 0 and verbose:
        LOGGER.info("All gallery examples passed validation.")
    elif exit_code != 0:
        total = sum(len(result.errors) for result in results if not result.ok)
        failing = sum(1 for result in results if not result.ok)
        LOGGER.error("Found %d issue(s) across %d file(s).", total, failing)
    return exit_code


def _parse_args(argv: list[str]) -> GalleryOptions:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Sphinx-Gallery example docstrings for kgfoundry.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory containing gallery examples (default: examples/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable stricter validation rules (title punctuation, constraints bullets).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print success messages for passing files.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Reserved for future automatic fixes.",
    )
    parsed = parser.parse_args(argv)
    examples_dir = cast(Path, parsed.examples_dir)
    strict_flag = bool(cast(bool, parsed.strict))
    verbose_flag = bool(cast(bool, parsed.verbose))
    fix_flag = bool(cast(bool, parsed.fix))
    return GalleryOptions(
        examples_dir=examples_dir,
        strict=strict_flag,
        verbose=verbose_flag,
        fix=fix_flag,
    )


def _run_from_cli(argv: list[str]) -> int:
    """Entry point used by ``if __name__ == '__main__'`` guard."""
    options = _parse_args(argv)
    if options.fix:
        LOGGER.warning("Automatic fixing is not implemented yet.")
        return 2
    examples_dir = options.examples_dir.resolve()
    if not examples_dir.exists():
        LOGGER.error("Examples directory not found: %s", examples_dir)
        return 2
    return main(examples_dir, strict=options.strict, verbose=options.verbose)


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    sys.exit(_run_from_cli(sys.argv[1:]))
