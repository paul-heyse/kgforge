"""Provide utilities for module.

Auto-generated API documentation for the ``docs._scripts.build_symbol_index`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
docs._scripts.build_symbol_index
"""


from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from griffe import Object

try:
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from detect_pkg import detect_packages, detect_primary  # noqa: E402

SRC = ROOT / "src"
ENV_PKGS = os.environ.get("DOCS_PKG")

loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def iter_packages() -> list[str]:
    """Return iter packages.

    Auto-generated reference for the ``iter_packages`` callable defined in ``docs._scripts.build_symbol_index``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from docs._scripts.build_symbol_index import iter_packages
    >>> result = iter_packages()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    docs._scripts.build_symbol_index
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    

    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    packages = detect_packages()
    return packages or [detect_primary()]


rows: list[dict[str, object | None]] = []


def safe_attr(node: Object, attr: str, default: object | None = None) -> object | None:
    """Return safe attr.

    Auto-generated reference for the ``safe_attr`` callable defined in ``docs._scripts.build_symbol_index``.
    
    Parameters
    ----------
    node : Object
        Description for ``node``.
    attr : str
        Description for ``attr``.
    default : object, optional
        Description for ``default``.
    
    Returns
    -------
    object | None
        Description of return value.
    
    Examples
    --------
    >>> from docs._scripts.build_symbol_index import safe_attr
    >>> result = safe_attr(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    docs._scripts.build_symbol_index
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    

    try:
        return getattr(node, attr)
    except Exception:
        return default


def walk(node: Object) -> None:
    """Return walk.

    Auto-generated reference for the ``walk`` callable defined in ``docs._scripts.build_symbol_index``.
    
    Parameters
    ----------
    node : Object
        Description for ``node``.
    
    Examples
    --------
    >>> from docs._scripts.build_symbol_index import walk
    >>> walk(...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    docs._scripts.build_symbol_index
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    

    doc = safe_attr(node, "docstring")
    file_rel = safe_attr(node, "relative_package_filepath")
    rows.append(
        {
            "path": node.path,
            "kind": node.kind.value,
            "file": str(file_rel) if file_rel else None,
            "lineno": safe_attr(node, "lineno"),
            "endlineno": safe_attr(node, "endlineno"),
            "doc": (doc.value.split("\n\n")[0] if doc and getattr(doc, "value", None) else ""),
        }
    )
    try:
        members = list(node.members.values())
    except Exception:
        members = []
    for member in members:
        walk(member)


for pkg in iter_packages():
    root = loader.load(pkg)
    walk(root)

out = ROOT / "docs" / "_build"
out.mkdir(parents=True, exist_ok=True)

test_map_path = out / "test_map.json"
if test_map_path.exists():
    try:
        _test_map = json.loads(test_map_path.read_text())
    except json.JSONDecodeError:  # pragma: no cover - defensive
        _test_map = {}
else:
    _test_map = {}

for row in rows:
    path = row.get("path")
    if isinstance(path, str):
        row["tested_by"] = _test_map.get(path, [])
    else:
        row["tested_by"] = []

(out / "symbols.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} entries to {out / 'symbols.json'}")
