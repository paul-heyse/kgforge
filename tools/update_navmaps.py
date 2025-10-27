#!/usr/bin/env python
"""Validate navigation metadata without mutating docstrings."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def iter_python_files() -> list[Path]:
    """Return all candidate Python modules under the source tree."""
    search_root = SRC if SRC.exists() else ROOT
    return sorted(path for path in search_root.rglob("*.py") if path.is_file())


def module_docstring(path: Path) -> str | None:
    """Return the module docstring if present."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return None
    return ast.get_docstring(tree)


def main() -> None:
    """Ensure NavMap sections are not injected into module docstrings."""
    offenders: list[Path] = []
    for file_path in iter_python_files():
        doc = module_docstring(file_path)
        if doc and "NavMap:" in doc:
            offenders.append(file_path)
    if offenders:
        joined = "\n".join(str(path) for path in offenders)
        message = (
            "Module docstrings must not contain 'NavMap:' sections. "
            f"Found violations in:\n{joined}"
        )
        raise SystemExit(message)


if __name__ == "__main__":
    main()
