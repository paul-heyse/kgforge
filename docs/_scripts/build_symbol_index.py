"""Build Symbol Index utilities."""

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
    """Compute iter packages.

    Carry out the iter packages operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    packages = detect_packages()
    return packages or [detect_primary()]


rows: list[dict[str, object | None]] = []


def safe_attr(node: Object, attr: str, default: object | None = None) -> object | None:
    """Compute safe attr.

    Carry out the safe attr operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    attr : str
        Description for ``attr``.
    default : object | None
        Description for ``default``.

    Returns
    -------
    object | None
        Description of return value.
    """
    
    
    
    try:
        return getattr(node, attr)
    except Exception:
        return default


def walk(node: Object) -> None:
    """Compute walk.

    Carry out the walk operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
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
