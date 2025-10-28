"""Overview of add module docstrings.

This module bundles add module docstrings logic for the kgfoundry stack. It groups related helpers
so downstream packages can import a single cohesive namespace. Refer to the functions and classes
below for implementation specifics.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Compute module name.

    Carry out the module name operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
    path : Path
        Description for ``path``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import module_name
    >>> result = module_name(...)
    >>> result  # doctest: +ELLIPSIS
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def needs_docstring(text: str) -> bool:
    """Compute needs docstring.

    Carry out the needs docstring operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    text : str
    text : str
        Description for ``text``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import needs_docstring
    >>> result = needs_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Compute insert docstring.

    Carry out the insert docstring operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import insert_docstring
    >>> result = insert_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    """
    text = path.read_text()
    if not needs_docstring(text):
        return False
    name = module_name(path)
    doc = f'"""Module for {name}."""\n\n'
    lines = text.splitlines(keepends=True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if len(lines) > idx and lines[idx].startswith("#") and "coding" in lines[idx]:
        idx += 1
    lines.insert(idx, doc)
    path.write_text("".join(lines))
    return True


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Examples
    --------
    >>> from tools.add_module_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    """
    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
