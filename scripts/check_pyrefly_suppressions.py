#!/usr/bin/env python3
"""Scan for unmanaged pyrefly and type: ignore suppressions.

This script enforces governance over type suppression usage by requiring that all
`# type: ignore` and `# pyrefly: ignore` pragmas include a ticket reference or
documented rationale. Suppressions without justification are flagged as errors.

Exit codes:
    0: No unmanaged suppressions found.
    1: Unmanaged suppressions detected; remediation guidance printed to stderr.
    2: Script error (invalid args, I/O failure).

Examples
--------
>>> # Scan the src/ directory
>>> python scripts/check_pyrefly_suppressions.py src/

>>> # Scan specific file
>>> python scripts/check_pyrefly_suppressions.py src/mymodule.py

>>> # Show help
>>> python scripts/check_pyrefly_suppressions.py --help
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

# Pattern to match unmanaged suppressions: type: ignore or pyrefly: ignore
# without a ticket reference (e.g., "# type: ignore[error-code] - ticket #123")
SUPPRESSION_PATTERN = re.compile(r"#\s*(?:type:\s*ignore|pyrefly:\s*ignore)(?:\[[\w,-]+\])?")
# Pattern to detect a ticket reference in a comment
TICKET_PATTERN = re.compile(r"(?:ticket|issue|#\d+|GH-\d+|TODO)", re.IGNORECASE)


def check_file(path: Path) -> list[tuple[int, str]]:
    """Scan a file for unmanaged suppressions.

    Parameters
    ----------
    path : Path
        Path to the file to scan.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, line_content) tuples for unmanaged suppressions.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        message = f"Failed to read {path}: {type(exc).__name__}"
        raise ValueError(message) from exc

    unmanaged: list[tuple[int, str]] = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        if SUPPRESSION_PATTERN.search(line) and not TICKET_PATTERN.search(line):
            unmanaged.append((line_num, line.rstrip()))

    return unmanaged


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912
    """Scan Python source files for unmanaged type suppressions.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments. If None, sys.argv[1:] is used.

    Returns
    -------
    int
        Exit code: 0 if no issues, 1 if suppressions found, 2 on error.

    Examples
    --------
    >>> # Scan current directory recursively
    >>> exit_code = main(["src/"])
    >>> exit_code
    0
    """
    parser = argparse.ArgumentParser(
        description="Check for unmanaged pyrefly and type: ignore suppressions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Suppressions must include a ticket reference (e.g., # type: ignore[error] - ticket #123)\n"
            "to be recognized as managed.\n\n"
            "Example:\n"
            "  python scripts/check_pyrefly_suppressions.py src/\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="File or directory paths to scan",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Exit with code 0 even if suppressions found (dry-run mode)",
    )

    args = parser.parse_args(argv)

    all_issues: dict[Path, list[tuple[int, str]]] = {}
    files_scanned = 0

    for target in cast(list[Path], args.paths):
        resolved = target.resolve()

        if not resolved.exists():
            print(f"error: path not found: {target}", file=sys.stderr)  # noqa: T201
            return 2

        if resolved.is_file():
            if resolved.suffix != ".py":
                continue
            files_scanned += 1
            try:
                issues = check_file(resolved)
                if issues:
                    all_issues[resolved] = issues
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)  # noqa: T201
                return 2
        elif resolved.is_dir():
            for py_file in sorted(resolved.rglob("*.py")):
                files_scanned += 1
                try:
                    issues = check_file(py_file)
                    if issues:
                        all_issues[py_file] = issues
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)  # noqa: T201
                    return 2

    if all_issues:
        print(  # noqa: T201
            f"error: found {len(all_issues)} files with unmanaged suppressions:\n",
            file=sys.stderr,
        )
        for file_path, issues in all_issues.items():
            print(f"\n{file_path}:", file=sys.stderr)  # noqa: T201
            for line_num, line_content in issues:
                print(  # noqa: T201
                    f"  {line_num:4d}: {line_content}",
                    file=sys.stderr,
                )
        print(  # noqa: T201
            "\nRemedy: Add a ticket reference to each suppression.\n"
            "Example: # type: ignore[misc] - ticket #456",
            file=sys.stderr,
        )
        exit_zero: bool = cast(bool, args.exit_zero)
        return 0 if exit_zero else 1

    print(f"✓ All {files_scanned} Python files clean (no unmanaged suppressions)")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
