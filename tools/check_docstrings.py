#!/usr/bin/env python
"""Docstring quality checks shared by kgfoundry development workflows."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO / "src",
    REPO / "tools",
    REPO / "docs" / "_scripts",
]


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the docstring audit helper.

    The parser currently exposes a single flag, ``--no-todo``, which toggles the
    stricter placeholder validation step executed after Ruff runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-todo",
        action="store_true",
        help="Fail if docstrings contain placeholder text such as 'TODO'.",
    )
    return parser.parse_args()


def iter_docstrings(path: Path) -> Iterable[tuple[Path, int, str]]:
    """Yield ``(path, lineno, text)`` tuples for every docstring in ``path``.

    The generator emits the file path, starting line number, and raw docstring
    text for module, class, and function definitions, mirroring the locations that
    Ruff and other documentation tools inspect.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if (doc := ast.get_docstring(tree, clean=False)) is not None:
        lineno = tree.body[0].lineno if tree.body else 1
        yield path, lineno, doc
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                yield path, node.body[0].lineno, doc


def check_placeholders() -> int:
    """Return ``0`` when no placeholder keywords are found in docstrings.

    The function scans every Python file in :data:`TARGETS` and records occurrences
    of ``TODO``, ``TBD``, or ``FIXME`` inside docstrings. It prints a summary of the
    offending locations to ``stderr`` and returns ``1`` if any placeholders remain.
    """
    errors: list[str] = []
    keywords = {"TODO", "TBD", "FIXME"}

    for target in TARGETS:
        for file_path in target.rglob("*.py"):
            try:
                for _, lineno, doc in iter_docstrings(file_path):
                    if any(key in doc for key in keywords):
                        rel = file_path.relative_to(REPO)
                        errors.append(f"{rel}:{lineno} placeholder text in docstring")
            except SyntaxError:
                continue

    if errors:
        print("Docstring placeholder check failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    return 0


def main() -> None:
    """Run Ruff's docstring checks and optional placeholder validation.

    Raises
    ------
    SystemExit
        Raised with the exit status of :func:`check_placeholders` when
        ``--no-todo`` is provided and placeholder text is detected.
    """
    options = parse_args()

    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "D",
        *(str(path) for path in TARGETS if path.exists()),
    ]
    subprocess.run(cmd, check=True)

    if options.no_todo:
        raise SystemExit(check_placeholders())


if __name__ == "__main__":
    main()
