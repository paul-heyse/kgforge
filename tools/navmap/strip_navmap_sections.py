#!/usr/bin/env python
"""Overview of strip navmap sections.

This module bundles strip navmap sections logic for the kgfoundry stack. It groups related helpers
so downstream packages can import a single cohesive namespace. Refer to the functions and classes
below for implementation specifics.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def iter_module_nodes(path: Path) -> tuple[ast.Module, ast.Expr | None]:
    """Compute iter module nodes.

    Carry out the iter module nodes operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    Tuple[ast.Module, ast.Expr | None]
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.strip_navmap_sections import iter_module_nodes
    >>> result = iter_module_nodes(...)
    >>> result  # doctest: +ELLIPSIS
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if not tree.body:
        return tree, None
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return tree, first
    return tree, None


def clean_docstring(text: str) -> str:
    """Compute clean docstring.

    Carry out the clean docstring operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    text : str
        Description for ``text``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.strip_navmap_sections import clean_docstring
    >>> result = clean_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.strip().startswith("NavMap:"):
            break
        lines.append(raw.rstrip())
    cleaned = "\n".join(line for line in lines if line.strip())
    return cleaned or "Module documentation."


def rewrite_module(path: Path) -> bool:
    """Compute rewrite module.

    Carry out the rewrite module operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.strip_navmap_sections import rewrite_module
    >>> result = rewrite_module(...)
    >>> result  # doctest: +ELLIPSIS
    """
    tree, doc_expr = iter_module_nodes(path)
    if doc_expr is None:
        return False
    doc_text = ast.get_docstring(tree)
    if not doc_text or "NavMap:" not in doc_text:
        return False

    cleaned = clean_docstring(doc_text)
    new_lines = ['"""' + cleaned, '"""'] if "\n" in cleaned else ['"""' + cleaned + '"""']

    original_lines = path.read_text(encoding="utf-8").splitlines()
    start = doc_expr.lineno - 1
    end = doc_expr.end_lineno or doc_expr.lineno
    original_lines[start:end] = new_lines
    path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")
    return True


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Examples
    --------
    >>> from tools.navmap.strip_navmap_sections import main
    >>> main()  # doctest: +ELLIPSIS
    """
    changed = 0
    for file_path in sorted(SRC.rglob("*.py")):
        if rewrite_module(file_path):
            changed += 1
            relative = file_path.relative_to(ROOT)
            print(f"[navmap] Stripped legacy NavMap section from {relative}")
    if changed == 0:
        print("[navmap] No NavMap sections found.")


if __name__ == "__main__":
    main()
