"""
Build a machine-friendly symbol index for AI agents:
- fully qualified name
- kind
- file path (relative to repo)
- start/end lines
- first-line summary doc
Writes: docs/_build/symbols.json
"""
import json
import os
import sys
from pathlib import Path

from griffe.loader import GriffeLoader

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from detect_pkg import detect_packages, detect_primary
SRC = ROOT / "src"
ENV_PKGS = os.environ.get("DOCS_PKG")

loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def iter_packages():
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    packages = detect_packages()
    return packages or [detect_primary()]


rows = []


def walk(node):
    rows.append(
        {
            "path": node.path,
            "kind": node.kind.value,
            "file": getattr(node, "relative_package_filepath", None),
            "lineno": getattr(node, "lineno", None),
            "endlineno": getattr(node, "endlineno", None),
            "doc": (
                node.docstring.value.split("\n\n")[0]
                if getattr(node, "docstring", None)
                else ""
            ),
        }
    )
    for member in node.members.values():
        walk(member)


for pkg in iter_packages():
    root = loader.load(pkg)
    walk(root)

out = ROOT / "docs" / "_build"
out.mkdir(parents=True, exist_ok=True)
(out / "symbols.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} entries to {out/'symbols.json'}")
