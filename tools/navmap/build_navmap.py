#!/usr/bin/env python3
"""Build the repository NavMap index.

Outputs:
  site/_build/navmap/navmap.json

Additions in this version:
- policy_version + link_mode at top-level
- commit-stable GitHub permalinks in entry["links"]["github"] when DOCS_LINK_MODE includes 'github'
- module_meta defaults (owner, stability, since, deprecated_in) inherited by per-symbol meta
- normalized (kebab-case) section IDs and stable, deduped exports

Environment:
  DOCS_LINK_MODE=editor|github|both   (default: editor)
  DOCS_GITHUB_ORG=<org>
  DOCS_GITHUB_REPO=<repo>
  DOCS_GITHUB_SHA=<sha>               (fallback to `git rev-parse HEAD`)
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
OUT = REPO / "site" / "_build" / "navmap"
OUT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT / "navmap.json"

# Link settings
G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "editor").lower()  # editor|github|both

SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]\s*$")
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]\s*$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

PLACEHOLDER_ALL = object()


class AllDictTemplate:
    """Sentinel representing a __all__-driven dict comprehension."""

    __slots__ = ("template",)

    def __init__(self, template: Any) -> None:
        self.template = template


def _literal_eval_navmap(node: ast.AST) -> Any:
    """Literal eval navmap.

    Parameters
    ----------
    node : ast.AST
        Description.

    Returns
    -------
    Any
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _literal_eval_navmap(...)
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__all__":
            return PLACEHOLDER_ALL
        raise ValueError(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal_eval_navmap(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            key = _literal_eval_navmap(key_node)
            if not isinstance(key, str):
                raise ValueError(key)
            result[key] = _literal_eval_navmap(value_node)
        return result
    if isinstance(node, ast.DictComp):
        # Only support comprehension of the form {name: TEMPLATE for name in __all__}
        if len(node.generators) != 1:
            raise ValueError("unsupported dict comprehension")
        target = node.generators[0].target
        iterator = node.generators[0].iter
        if not isinstance(target, ast.Name) or not isinstance(iterator, ast.Name):
            raise ValueError("unsupported dict comprehension target")
        if iterator.id != "__all__":
            raise ValueError("unsupported dict comprehension iterator")
        template = _literal_eval_navmap(node.value)
        return AllDictTemplate(template)
    if isinstance(node, ast.Set):
        return {_literal_eval_navmap(elt) for elt in node.elts}
    raise ValueError(ast.dump(node))


def _replace_placeholders(value: Any, exports: list[str]) -> Any:
    """Replace placeholders.

    Parameters
    ----------
    value : Any
        Description.
    exports : list[str]
        Description.

    Returns
    -------
    Any
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _replace_placeholders(...)
    """
    if value is PLACEHOLDER_ALL:
        return list(dict.fromkeys(exports))
    if isinstance(value, AllDictTemplate):
        template = value.template
        result: dict[str, Any] = {}
        for name in exports:
            resolved = _replace_placeholders(template, exports)
            if isinstance(resolved, dict):
                # Deep copy template and inject symbol name if applicable
                cloned = {k: _replace_placeholders(v, exports) for k, v in resolved.items()}
                result[name] = cloned
            else:
                result[name] = resolved
        return result
    if isinstance(value, list):
        items: list[Any] = []
        for entry in value:
            replaced = _replace_placeholders(entry, exports)
            if isinstance(replaced, list):
                items.extend(replaced)
            else:
                items.append(replaced)
        return items
    if isinstance(value, dict):
        return {k: _replace_placeholders(v, exports) for k, v in value.items()}
    if isinstance(value, set):
        items: set[Any] = set()
        for entry in value:
            replaced = _replace_placeholders(entry, exports)
            if isinstance(replaced, list):
                items.update(replaced)
            else:
                items.add(replaced)
        return items
    return value


@dataclass
class ModuleInfo:
    """Represent ModuleInfo.

    Attributes
    ----------
    attribute : type
        Description.

    Methods
    -------
    method()
        Description.

    Examples
    --------
    >>> ModuleInfo(...)
    """

    module: str
    path: Path
    exports: list[str]
    sections: dict[str, int]  # id -> lineno (1-based)
    anchors: dict[str, int]  # symbol -> lineno (1-based)
    navmap_dict: dict[str, Any]  # parsed __navmap__ (may be {})


def _rel(p: Path) -> str:
    """Return ``p`` relative to the repository root when possible."""
    try:
        return str(p.relative_to(REPO))
    except Exception:
        return str(p)


def _git_sha() -> str:
    """Return the current Git commit hash, falling back to environment overrides."""
    if G_SHA:
        return G_SHA
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "HEAD"


def _gh_link(path: Path, start: int | None, end: int | None) -> str | None:
    """Commit-stable GitHub permalink using #L anchors."""
    if not (G_ORG and G_REPO):
        return None
    sha = _git_sha()
    frag = ""
    if start and end and end >= start:
        frag = f"#L{start}-L{end}"
    elif start:
        frag = f"#L{start}"
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{sha}/{_rel(path)}{frag}"


def _module_name(py: Path) -> str | None:
    """Module name.

    Parameters
    ----------
    py : Path
        Description.

    Returns
    -------
    str | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _module_name(...)
    """
    if py.suffix != ".py":
        return None
    try:
        rel = py.relative_to(SRC)
    except Exception:
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None
    return ".".join(parts)


def _literal_list_of_strs(node: ast.AST) -> list[str] | None:
    """Best-effort get list/tuple of strings from AST node."""
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                vals.append(elt.value)
            elif isinstance(elt, ast.Name) and IDENT_RE.match(elt.id):
                vals.append(elt.id)
            else:
                return None
        return vals
    return None


def _parse_py(py: Path) -> tuple[dict[str, Any], list[str]]:
    """Return (__navmap__ dict, __all__ list or [])."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return {}, []

    nav_raw: dict[str, Any] | None = None
    exports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in targets:
                vals = _literal_list_of_strs(node.value)
                if vals is not None:
                    exports = vals
            if "__navmap__" in targets:
                try:
                    nav_raw = _literal_eval_navmap(node.value)
                    if not isinstance(nav_raw, dict):
                        nav_raw = {}
                except Exception:
                    nav_raw = {}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                vals = _literal_list_of_strs(node.value)
                if vals is not None:
                    exports = vals
            if node.target.id == "__navmap__":
                try:
                    nav_raw = _literal_eval_navmap(node.value)
                    if not isinstance(nav_raw, dict):
                        nav_raw = {}
                except Exception:
                    nav_raw = {}

    nav: dict[str, Any] = {}
    exports = list(dict.fromkeys(exports))
    if nav_raw:
        exports_hint = exports
        if not exports_hint:
            raw_exports = nav_raw.get("exports") if isinstance(nav_raw, dict) else None
            if isinstance(raw_exports, list):
                exports_hint = [x for x in raw_exports if isinstance(x, str)]
        try:
            nav = _replace_placeholders(nav_raw, exports_hint)
        except Exception:
            nav = {}
        if isinstance(nav.get("exports"), list):
            exports = list(dict.fromkeys(x for x in nav["exports"] if isinstance(x, str)))
    return nav, exports


def _scan_inline_markers(py: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (sections, anchors) with 1-based line numbers."""
    sections: dict[str, int] = {}
    anchors: dict[str, int] = {}
    try:
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            m = SECTION_RE.match(line)
            if m:
                sid = m.group(1)
                sections[sid] = i
            m = ANCHOR_RE.match(line)
            if m:
                anchors[m.group(1)] = i
    except Exception:
        pass
    return sections, anchors


def _kebab(s: str) -> str:
    """Normalize ``s`` into a kebab-case identifier string."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _collect_module(py: Path) -> ModuleInfo | None:
    """Parse ``py`` and return gathered navmap metadata if it is a module."""
    mod = _module_name(py)
    if not mod:
        return None
    navmap_dict, exports = _parse_py(py)
    sections, anchors = _scan_inline_markers(py)

    # Normalize sections to kebab-case and ensure symbol lists are unique & stable
    nav_sections = []
    for sec in navmap_dict.get("sections", []):
        sid = _kebab(str(sec.get("id", ""))) if sec else ""
        symbols = list(dict.fromkeys((sec or {}).get("symbols", [])))
        nav_sections.append({"id": sid, "symbols": symbols})
    navmap_dict["sections"] = nav_sections

    # Stable, deduped exports
    nav_exports = list(dict.fromkeys(navmap_dict.get("exports", exports)))
    exports = nav_exports

    return ModuleInfo(
        module=mod,
        path=py,
        exports=exports,
        sections=sections,
        anchors=anchors,
        navmap_dict=navmap_dict,
    )


def _discover_py_files() -> list[Path]:
    """Return every Python source file under ``src`` sorted lexicographically."""
    return sorted(p for p in SRC.rglob("*.py") if p.is_file())


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
    files = _discover_py_files()
    data: dict[str, Any] = {
        "commit": _git_sha(),
        "policy_version": "1",
        "link_mode": LINK_MODE,
        "modules": {},
    }

    for py in files:
        info = _collect_module(py)
        if not info:
            continue

        # Build link bundle
        links = {"source": f"vscode://file/{_rel(info.path)}"}  # VS Code URL scheme
        if LINK_MODE in ("github", "both"):
            links["github"] = _gh_link(info.path, None, None)

        # Per-symbol meta with module defaults inherited
        module_meta = {
            k: v
            for k, v in info.navmap_dict.items()
            if k in ("owner", "stability", "since", "deprecated_in") and v is not None
        }

        symbols_meta = {
            name: dict(meta) for name, meta in (info.navmap_dict.get("symbols") or {}).items()
        }
        if module_meta:
            for name in list(symbols_meta.keys()):
                for k, v in module_meta.items():
                    symbols_meta[name].setdefault(k, v)
        if not symbols_meta:
            for name in info.exports:
                fields: dict[str, Any] = {}
                for k, v in module_meta.items():
                    fields[k] = v
                symbols_meta[name] = fields

        entry = {
            "path": _rel(info.path),
            "exports": list(dict.fromkeys(info.exports)),
            "sections": info.navmap_dict.get("sections", []),
            "section_lines": info.sections,
            "anchors": info.anchors,
            "links": links,
            "meta": symbols_meta,
            "module_meta": module_meta,
            "tags": info.navmap_dict.get("tags", []),
            "synopsis": info.navmap_dict.get("synopsis", ""),
            "see_also": info.navmap_dict.get("see_also", []),
            "deps": info.navmap_dict.get("deps", []),
        }
        data["modules"][info.module] = entry

    # Write
    out = json_path or INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main() -> int:
    """Main.

    Returns
    -------
    int
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> main(...)
    """
    build_index()
    print(f"navmap built → {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
