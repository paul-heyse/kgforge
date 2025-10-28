#!/usr/bin/env python
"""Check Docstrings utilities."""

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
    """Compute parse args.

    Carry out the parse args operation.

    Returns
    -------
    argparse.Namespace
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-todo",
        action="store_true",
        help="Fail if docstrings contain placeholder text such as 'TODO'.",
    )
    return parser.parse_args()


def iter_docstrings(path: Path) -> Iterable[tuple[Path, int, str]]:
    """Compute iter docstrings.

    Carry out the iter docstrings operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    Iterable[Tuple[Path, int, str]]
        Description of return value.
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
    """Compute check placeholders.

    Carry out the check placeholders operation.

    Returns
    -------
    int
        Description of return value.
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
    """Compute main.

    Carry out the main operation.

    Raises
    ------
    SystemExit
        Raised when validation fails.
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
