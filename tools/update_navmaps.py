"""Regenerate module docstrings with NavMap sections.

This script inspects each package under ``src/`` (or the root package if
``src`` is absent) using Griffe, collects public classes and functions from
each module, and rewrites the module docstring so it contains a short nav map.

It is idempotent and updates only the first statement in the module when it is
already a docstring. If a module lacks a docstring, it is left untouched to
avoid altering unexpected files.
"""

from __future__ import annotations

import ast
import os
import textwrap
from collections.abc import Iterable
from pathlib import Path

from griffe import Object

try:  # Prefer modern import path; fall back for older Griffe.
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

from detect_pkg import detect_packages, detect_primary

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def iter_packages() -> list[str]:
    """Iterate packages specified via environment or auto-detected."""
    env_pkgs = os.environ.get("DOCS_PKG")
    if env_pkgs:
        return [pkg.strip() for pkg in env_pkgs.split(",") if pkg.strip()]
    detected = detect_packages()
    if detected:
        return detected
    return [detect_primary()]


def summarize(node: Object) -> str:
    """Return a truncated first sentence from the node docstring."""
    doc = getattr(node, "docstring", None)
    if doc and getattr(doc, "value", None):
        first = doc.value.strip().splitlines()[0].strip()
        first = first.rstrip(".")
        return textwrap.shorten(first, width=60, placeholder="…")
    return ""


def iter_public_members(node: Object) -> Iterable[Object]:
    """Yield public members for the provided module node."""
    members = getattr(node, "members", {})
    return [m for m in members.values() if not getattr(m, "name", "").startswith("_")]


def build_nav_lines(module: Object) -> list[str]:
    """Construct nav-map docstring lines for a module."""
    lines = [f"Module for {module.path}."]
    entries: list[str] = []
    for member in iter_public_members(module):
        kind = getattr(member.kind, "value", "")
        if kind not in {"class", "function"}:
            continue
        summary = summarize(member)
        name = getattr(member, "name", "")
        if summary:
            entries.append(f"- {name}: {summary}.")
        else:
            entries.append(f"- {name}.")
    if entries:
        lines.append("")
        lines.append("NavMap:")
        lines.extend(entries)
    return lines


def rewrite_docstring(file_path: Path, nav_lines: list[str]) -> None:
    """Rewrite Docstring.

    Parameters
    ----------
    file_path : Path
        TODO.
    nav_lines : list[str]
        TODO.

    Returns
    -------
    None
        TODO.
    """
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if not tree.body:
        return
    first = tree.body[0]
    if not isinstance(first, ast.Expr):
        return
    value = getattr(first, "value", None)
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return

    start = first.lineno - 1
    end = first.end_lineno or first.lineno
    lines = text.splitlines()
    new_docstring = '"""' + "\n".join(nav_lines) + "\n" + '"""'
    lines[start:end] = [new_docstring]
    new_text = "\n".join(lines) + "\n"
    if new_text != text:
        file_path.write_text(new_text, encoding="utf-8")


def update_module_doc(module: Object) -> None:
    """Rewrite the module docstring with a nav map if we can locate the file."""
    rel = getattr(module, "relative_package_filepath", None)
    if not rel:
        return
    base = SRC if SRC.exists() else ROOT
    file_path = (base / rel).resolve()
    if not file_path.exists() or file_path.suffix != ".py":
        return
    nav_lines = build_nav_lines(module)
    rewrite_docstring(file_path, nav_lines)


def main() -> None:
    """Run the nav-map regeneration entry point."""
    search_root = SRC if SRC.exists() else ROOT
    loader = GriffeLoader(search_paths=[str(search_root)])
    for pkg in iter_packages():
        module = loader.load(pkg)
        update_module_doc(module)
        for member in module.members.values():
            if getattr(member.kind, "value", "") == "module":
                update_module_doc(member)


if __name__ == "__main__":
    main()
