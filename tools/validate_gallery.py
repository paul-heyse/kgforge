#!/usr/bin/env python3
"""Overview of validate gallery.

This module bundles validate gallery logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TITLE_MAX_LENGTH = 79
UNDERLINE_TOLERANCE = 1

TITLE_UNDERLINE_PATTERN = re.compile(r"^(?P<char>=)\1*$")
CUSTOM_LABEL_PATTERN = re.compile(r"(?m)^\.\.\s+_gallery_[\w-]+:\s*$")
TAGS_PATTERN = re.compile(r"(?m)^\.\.\s+tags::\s*")
CONSTRAINTS_HEADER_PATTERN = re.compile(
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
    """Model the ValidationResult.

    Represent the validationresult data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    path: Path
    errors: list[str]

    def extend(self, messages: Iterable[str]) -> None:
        """Compute extend.

        Carry out the extend operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        messages : collections.abc.Iterable
        messages : collections.abc.Iterable
            Description for ``messages``.

        Examples
        --------
        >>> from tools.validate_gallery import extend
        >>> extend(...)  # doctest: +ELLIPSIS
        """
        self.errors.extend(messages)

    @property
    def ok(self) -> bool:
        """Compute ok.

        Carry out the ok operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Returns
        -------
        bool
            Description of return value.

        Examples
        --------
        >>> from tools.validate_gallery import ok
        >>> result = ok()
        >>> result  # doctest: +ELLIPSIS
        """
        return not self.errors


class GalleryValidationError(RuntimeError):
    """Model the GalleryValidationError.

    Represent the galleryvalidationerror data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """


def validate_title_format(docstring: str) -> tuple[bool, str]:
    """Compute validate title format.

    Carry out the validate title format operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    docstring : str
    docstring : str
        Description for ``docstring``.

    Returns
    -------
    Tuple[bool, str]
        Description of return value.

    Examples
    --------
    >>> from tools.validate_gallery import validate_title_format
    >>> result = validate_title_format(...)
    >>> result  # doctest: +ELLIPSIS
    """
    lines = [line.rstrip() for line in inspect.cleandoc(docstring).splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return False, "docstring is empty"

    title = lines[0]
    if len(title) > TITLE_MAX_LENGTH:
        return False, f"title exceeds {TITLE_MAX_LENGTH} characters"

    if len(lines) < 2:
        return False, "missing underline under the title"

    underline = lines[1]
    if not TITLE_UNDERLINE_PATTERN.fullmatch(underline):
        return False, "title underline must be composed of '=' characters"

    if abs(len(underline) - len(title)) > UNDERLINE_TOLERANCE:
        return False, "title underline length must match the title (±1 character)"

    if len(lines) < 3 or lines[2].strip():
        return False, "expected a blank line after the title underline"

    return True, ""


def check_orphan_directive(docstring: str) -> bool:
    """Compute check orphan directive.

    Carry out the check orphan directive operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    docstring : str
    docstring : str
        Description for ``docstring``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.validate_gallery import check_orphan_directive
    >>> result = check_orphan_directive(...)
    >>> result  # doctest: +ELLIPSIS
    """
    return ":orphan:" in docstring


def check_custom_labels(docstring: str) -> list[str]:
    """Compute check custom labels.

    Carry out the check custom labels operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    docstring : str
    docstring : str
        Description for ``docstring``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.validate_gallery import check_custom_labels
    >>> result = check_custom_labels(...)
    >>> result  # doctest: +ELLIPSIS
    """
    return CUSTOM_LABEL_PATTERN.findall(docstring)


def _has_tags_directive(docstring: str) -> bool:
    """Return ``True`` if the docstring declares a ``..

    tags::`` directive.
    """
    return TAGS_PATTERN.search(docstring) is not None


def _has_constraints_section(docstring: str) -> bool:
    """Return ``True`` if a ``Constraints`` section header is present."""
    match = CONSTRAINTS_HEADER_PATTERN.search(docstring)
    return bool(match and set(match.group("rule")) == {"-"})


def _load_docstring(path: Path) -> str | None:
    """Extract the module docstring from ``path`` using ``ast`` parsing."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - should never happen for examples
        message = f"{path}: failed to parse module ({exc})"
        raise GalleryValidationError(message) from exc
    return ast.get_docstring(module, clean=False)


def validate_example_file(file_path: Path, *, strict: bool = False) -> list[str]:
    """Compute validate example file.

    Carry out the validate example file operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    file_path : Path
    file_path : Path
        Description for ``file_path``.
    strict : bool | None
    strict : bool | None, optional, default=False
        Description for ``strict``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.validate_gallery import validate_example_file
    >>> result = validate_example_file(...)
    >>> result  # doctest: +ELLIPSIS
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
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    examples_dir : Path
    examples_dir : Path
        Description for ``examples_dir``.
    strict : bool | None
    strict : bool | None, optional, default=False
        Description for ``strict``.
    verbose : bool | None
    verbose : bool | None, optional, default=False
        Description for ``verbose``.

    Returns
    -------
    int
        Description of return value.

    Examples
    --------
    >>> from tools.validate_gallery import main
    >>> result = main(...)
    >>> result  # doctest: +ELLIPSIS
    """
    results: list[ValidationResult] = []
    exit_code = 0
    for path in _iter_example_files(examples_dir):
        messages = validate_example_file(path, strict=strict)
        result = ValidationResult(path=path, errors=messages)
        results.append(result)
        if result.ok:
            if verbose:
                print(f"✔ {path}")
            continue
        exit_code = 1
        for message in result.errors:
            print(f"✖ {path}: {message}", file=sys.stderr)

    if exit_code == 0 and verbose:
        print("All gallery examples passed validation.")
    elif exit_code != 0:
        total = sum(len(result.errors) for result in results if not result.ok)
        failing = sum(1 for result in results if not result.ok)
        print(
            f"Found {total} issue(s) across {failing} file(s).",
            file=sys.stderr,
        )
    return exit_code


def _parse_args(argv: list[str]) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def _run_from_cli(argv: list[str]) -> int:
    """Entry point used by ``if __name__ == '__main__'`` guard."""
    args = _parse_args(argv)
    if args.fix:
        print("Automatic fixing is not implemented yet.", file=sys.stderr)
        return 2
    examples_dir = args.examples_dir.resolve()
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}", file=sys.stderr)
        return 2
    return main(examples_dir, strict=args.strict, verbose=args.verbose)


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    sys.exit(_run_from_cli(sys.argv[1:]))
