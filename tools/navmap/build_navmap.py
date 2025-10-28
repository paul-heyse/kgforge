#!/usr/bin/env python
"""Build Navmap utilities."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
OUT = REPO / "site" / "_build" / "navmap"
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]")
SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]")

G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "editor")
POLICY_VERSION = "1"


@dataclass(slots=True)
class ModuleInfo:
    """Describe ModuleInfo."""

    path: Path
    module: str
    exports: list[str]
    navmap: dict[str, Any]
    anchors: dict[str, int]
    sections: dict[str, int]


def _extract_string(node: ast.AST) -> str | None:
    """Compute extract string."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_strings(node: ast.AST) -> list[str]:
    """Compute extract strings."""

    match node:
        case ast.List(elts=elts) | ast.Tuple(elts=elts):
            return [s for elt in elts if (s := _extract_string(elt)) is not None]
        case _:
            return []


def _literal_eval(node: ast.AST, names: Mapping[str, object]) -> object:
    """Compute literal eval."""

    match node:
        case ast.Constant(value=value):
            return value
        case ast.List(elts=elts):
            return [_literal_eval(elt, names) for elt in elts]
        case ast.Tuple(elts=elts):
            return tuple(_literal_eval(elt, names) for elt in elts)
        case ast.Set(elts=elts):
            return {_literal_eval(elt, names) for elt in elts}
        case ast.Dict(keys=keys, values=values):
            return {
                _literal_eval(key, names): _literal_eval(value, names)
                for key, value in zip(keys, values, strict=True)
            }
        case ast.Name(id=name) if name in names:
            return names[name]
        case _:
            message = f"Unsupported expression in navmap literal: {ast.dump(node)}"
            raise ValueError(message)


def _module_name(py: Path) -> str | None:
    """Compute module name."""

    try:
        rel = py.relative_to(SRC)
    except ValueError:
        return None
    parts = rel.parent.parts if py.name == "__init__.py" else rel.with_suffix("").parts
    if not parts:
        return None
    return ".".join(parts)


def _rel(path: Path) -> str:
    """Compute rel."""

    return path.relative_to(REPO).as_posix()


def _git_sha() -> str:
    """Compute git sha."""

    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
        .decode("utf-8")
        .strip()
    )


def _gh_link(path: Path, start: int | None = None, end: int | None = None) -> str | None:
    """Build a GitHub permalink for the provided file and optional span."""

    if not (G_ORG and G_REPO):
        return None
    sha = G_SHA or _git_sha()
    fragment = ""
    if start and end and end != start:
        fragment = f"#L{start}-L{end}"
    elif start:
        fragment = f"#L{start}"
    rel_path = _rel(path)
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{sha}/{rel_path}{fragment}"


def _parse_module(py: Path) -> ModuleInfo | None:  # noqa: C901, PLR0912
    """Compute parse module."""

    module = _module_name(py)
    if not module:
        return None

    text = py.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py))

    exports: list[str] = []
    navmap: dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    exports = _extract_strings(node.value)
                elif target.id == "__navmap__":
                    names = {"__all__": exports}
                    try:
                        navmap = _literal_eval(node.value, names)
                    except Exception:
                        navmap = {}
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__navmap__"
            and node.value is not None
        ):
            names = {"__all__": exports}
            try:
                navmap = _literal_eval(node.value, names)
            except Exception:
                navmap = {}

    anchors: dict[str, int] = {}
    sections: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if match := ANCHOR_RE.match(line):
            anchors[match.group(1)] = lineno
        if match := SECTION_RE.match(line):
            sections[match.group(1)] = lineno

    return ModuleInfo(
        path=py,
        module=module,
        exports=exports,
        navmap=navmap,
        anchors=anchors,
        sections=sections,
    )


def _collect_modules(root: Path) -> list[ModuleInfo]:
    """Compute collect modules."""

    modules: list[ModuleInfo] = []
    for py in root.rglob("*.py"):
        info = _parse_module(py)
        if info:
            modules.append(info)
    return modules


def build_index(root: Path = SRC, json_path: Path | None = None) -> dict[str, Any]:
    """Compute build index.

    Carry out the build index operation.

    Parameters
    ----------
    root : Path | None
        Description for ``root``.
    json_path : Path | None
        Description for ``json_path``.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    
    
    
    
    
    
    











    sha = _git_sha()
    data: dict[str, Any] = {
        "commit": sha,
        "policy_version": POLICY_VERSION,
        "link_mode": LINK_MODE,
        "modules": {},
    }

    for info in _collect_modules(root):
        navmap = info.navmap or {}
        exports = list(dict.fromkeys(navmap.get("exports", info.exports)))
        sections = navmap.get("sections", [])
        module_meta = {
            key: navmap[key]
            for key in ("owner", "stability", "since", "deprecated_in")
            if navmap.get(key) is not None
        }
        symbols_meta = {
            name: dict(meta)
            for name, meta in navmap.get("symbols", {}).items()
        }

        if module_meta:
            for name, meta in symbols_meta.items():
                for key, value in module_meta.items():
                    meta.setdefault(key, value)

        links: dict[str, Any] = {
            "source": f"vscode://file/{_rel(info.path)}",
        }
        if LINK_MODE in {"github", "both"}:
            gh_url = _gh_link(info.path)
            if gh_url:
                links["github"] = gh_url

        data["modules"][info.module] = {
            "path": _rel(info.path),
            "exports": exports,
            "sections": sections,
            "section_lines": info.sections,
            "anchors": info.anchors,
            "links": links,
            "meta": symbols_meta,
            "module_meta": module_meta,
            "tags": navmap.get("tags", []),
            "synopsis": navmap.get("synopsis", ""),
            "see_also": navmap.get("see_also", []),
            "deps": navmap.get("deps", []),
        }

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=SRC,
        help="Directory tree to scan for navmap metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=OUT / "navmap.json",
        help="Path to write the navmap JSON output (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compute main.

    Carry out the main operation.

    Parameters
    ----------
    argv : List[str] | None
        Description for ``argv``.

    Returns
    -------
    int
        Description of return value.
    """
    
    
    
    
    
    
    











    args = _parse_args(argv)
    root = args.root.resolve()
    json_path = args.json.resolve()
    index = build_index(root=root, json_path=json_path)
    print(f"Wrote {_rel(json_path)} @ {index['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
