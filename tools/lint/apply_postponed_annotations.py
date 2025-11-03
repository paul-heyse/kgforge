"""Utility to apply postponed annotations (PEP 563) to Python modules.

This script automatically inserts `from __future__ import annotations` into
Python modules, ensuring that type hints are no longer evaluated at import time.
It respects module docstrings, encoding declarations, and shebang lines.

## Design

1. Scans targeted directories for .py files
2. Checks if `from __future__ import annotations` is already present
3. If missing, inserts it after:
   - Shebang (#!/usr/bin/env python, etc.)
   - Encoding declaration (# -*- coding: utf-8 -*-)
   - Module docstring (triple-quoted strings at top)
4. Leaves other imports and code untouched
5. Reports summary: files processed, inserted count, errors

## Usage

    # Apply to entire src/ directory
    python -m tools.lint.apply_postponed_annotations src/

    # Apply to specific modules
    python -m tools.lint.apply_postponed_annotations docs/_scripts/ tools/

    # Check without modifying (dry-run)
    python -m tools.lint.apply_postponed_annotations --check-only src/

"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools._shared.logging import LoggerAdapter

from tools._shared.logging import get_logger


def should_skip_file(path: Path) -> bool:
    """Determine if a file should be skipped."""
    # Skip __pycache__, .git, .venv, etc.
    if any(part.startswith(".") for part in path.parts):
        return True
    return "pycache" in path.parts


def has_postponed_annotations(content: str) -> bool:
    """Check if file already has postponed annotations import."""
    return "from __future__ import annotations" in content


def extract_header_and_body(content: str) -> tuple[str, str]:
    """Extract header (shebang, encoding, docstring) from body.

    Returns
    -------
    tuple[str, str]
        (header_section, remaining_body)

    """
    lines = content.split("\n")
    header_lines: list[str] = []
    idx = 0

    # Shebang
    if idx < len(lines) and lines[idx].startswith("#!"):
        header_lines.append(lines[idx])
        idx += 1

    # Encoding declaration
    if idx < len(lines) and ("coding:" in lines[idx] or "coding=" in lines[idx]):
        header_lines.append(lines[idx])
        idx += 1

    # Module docstring: try to parse it
    if idx < len(lines):
        # Build a candidate for docstring extraction
        remaining = "\n".join(lines[idx:])
        try:
            tree = ast.parse(remaining)
            # Check if first statement is a docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                # It's a docstring; extract it by finding closing quote
                docstring_node = tree.body[0]
                # Count lines in docstring
                start_line = docstring_node.lineno or 1
                end_line = docstring_node.end_lineno or start_line
                docstring_lineno = end_line - start_line + 1
                header_lines.extend(lines[idx : idx + docstring_lineno])
                idx += docstring_lineno
        except (SyntaxError, IndexError):
            # If parsing fails, don't assume docstring
            pass

    header = "\n".join(header_lines)
    body = "\n".join(lines[idx:])
    return header, body


def apply_postponed_annotations(content: str) -> str:
    """Insert postponed annotations import if not present.

    Respects shebang, encoding, and module docstring.

    Parameters
    ----------
    content : str
        File content.

    Returns
    -------
    str
        Modified content with postponed annotations inserted.

    """
    if has_postponed_annotations(content):
        return content

    header, body = extract_header_and_body(content)

    # Build the new import statement
    import_line = "from __future__ import annotations\n"

    # Combine: header + import + body
    if header:
        # If header ends with newline, don't add extra
        if header.endswith("\n"):
            return header + import_line + body
        return header + "\n" + import_line + body

    return import_line + body


def process_directory(
    root: Path,
    *,
    check_only: bool = False,
    logger: LoggerAdapter | None = None,
) -> tuple[int, int, int]:
    """Process all Python files in a directory.

    Parameters
    ----------
    root : Path
        Root directory to scan.
    check_only : bool, optional
        If True, don't modify files, only report (default: False).
    logger : LoggerAdapter | None, optional
        Logger instance (default: None).

    Returns
    -------
    tuple[int, int, int]
        (files_processed, files_modified, errors)

    """
    active_logger = logger or get_logger(__name__)

    processed = 0
    modified = 0
    errors = 0

    py_files = sorted(root.rglob("*.py"))
    msg = f"Scanning {len(py_files)} Python files in {root}"
    active_logger.info(msg)

    for fpath in py_files:
        if should_skip_file(fpath):
            continue

        processed += 1
        try:
            content = fpath.read_text(encoding="utf-8")
            new_content = apply_postponed_annotations(content)

            if new_content != content:
                if not check_only:
                    fpath.write_text(new_content, encoding="utf-8")
                modified += 1
                relpath = fpath.relative_to(root)
                active_logger.info("  %s: annotations added", relpath)
        except Exception:
            errors += 1
            relpath = fpath.relative_to(root)
            active_logger.exception("  %s: failed", relpath)

    return processed, modified, errors


def main(argv: Sequence[str] | None = None) -> int:
    """Apply postponed annotations to specified directories.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments (directories or flags).
        Flags: --check-only, --help

    Returns
    -------
    int
        Exit code (0 = success, non-zero = failure).

    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Apply postponed annotations (PEP 563) to Python modules."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        help="Directories to process (default: src/)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check without modifying files",
    )

    args = parser.parse_args(argv)
    logger = get_logger(__name__)

    total_processed = 0
    total_modified = 0
    total_errors = 0

    # Cast for type safety (argparse.Namespace.directories is typed as Any)
    raw_directories = cast("Sequence[Path]", getattr(args, "directories", ()))
    directories = list(raw_directories) if raw_directories else [Path("src")]
    for directory in directories:
        if not directory.exists():
            msg = f"Directory not found: {directory}"
            logger.warning(msg)
            continue

        processed, modified, errors = process_directory(
            directory,
            check_only=bool(getattr(args, "check_only", False)),
            logger=logger,
        )
        total_processed += processed
        total_modified += modified
        total_errors += errors

    # Summary
    mode = "(CHECK-ONLY)" if bool(getattr(args, "check_only", False)) else "(MODIFIED)"
    summary_msg = (
        f"Summary {mode}: processed={total_processed}, "
        f"modified={total_modified}, errors={total_errors}"
    )
    logger.info(summary_msg)

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
