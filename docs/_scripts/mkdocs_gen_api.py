"""Mkdocs Gen Api utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files
from griffe import Object

try:
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from detect_pkg import detect_packages, detect_primary  # noqa: E402

out = Path("api")
with mkdocs_gen_files.open(out / "index.md", "w") as f:
    f.write("# API Reference\n")


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    packages = detect_packages()
    return packages or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def write_node(node: Object) -> None:
    """Compute write node.

    Carry out the write node operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    """
    
    rel = node.path.replace(".", "/")
    page = out / rel / "index.md"
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(f"# `{node.path}`\n\n::: {node.path}\n")


for pkg in iter_packages():
    module = loader.load(pkg)
    write_node(module)
    for member in module.members.values():
        if member.is_package or member.is_module:
            write_node(member)
