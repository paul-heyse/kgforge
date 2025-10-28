#!/usr/bin/env python
#!/usr/bin/env python3
"""Validate NavMap conformance against policy and perform a round-trip check.

Enforces:
- __navmap__ exists when module has exports
- first section id == 'public-api'
- section IDs are kebab-case
- section symbols are valid Python identifiers and have nearby [nav:anchor]
- per-exported-symbol meta: owner present; stability in {stable,beta,experimental,deprecated}
- PEP 440 validation for since/deprecated_in; if both present then deprecated_in >= since
- round-trip: a freshly built JSON matches inline markers (section_lines & anchors)

Exit codes:
  0 = OK
  1 = violations found or round-trip mismatch
  2 = runtime/installation error (e.g., missing dependency)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
INDEX = REPO / "site" / "_build" / "navmap" / "navmap.json"

# Regexes
SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]\s*$")
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]\s*$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
STABILITY = {"stable", "beta", "experimental", "deprecated"}

# Optional PEP 440 validation
try:
    from packaging.version import InvalidVersion, Version  # type: ignore
except Exception:  # pragma: no cover
    Version = None  # type: ignore
    InvalidVersion = Exception  # type: ignore


def _read_text(py: Path) -> list[str]:
    """Return file contents as a list of lines, or an empty list on failure."""
    try:
        return py.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _scan_inline(py: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (sections, anchors) mapping to 1-based line numbers."""
    sections: dict[str, int] = {}
    anchors: dict[str, int] = {}
    for i, line in enumerate(_read_text(py), 1):
        m = SECTION_RE.match(line)
        if m:
            sections[m.group(1)] = i
        m = ANCHOR_RE.match(line)
        if m:
            anchors[m.group(1)] = i
    return sections, anchors


def _parse_navmap_dict(py: Path) -> dict[str, Any]:
    """Parse a module's __navmap__ by a quick-and-safe literal eval.

    This checker only needs fields existence; deep evaluation is unnecessary.
    """
    import ast

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__navmap__" for t in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, dict):
                        result = value
                except Exception:
                    pass
    return result


def _parse_all(py: Path) -> list[str]:
    """Return the literal ``__all__`` sequence when it can be safely evaluated."""
    import ast

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                try:
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        vals: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                vals.append(elt.value)
                            elif isinstance(elt, ast.Name) and IDENT_RE.match(elt.id):
                                vals.append(elt.id)
                            else:
                                return []
                        return vals
                except Exception:
                    return []
    return []


def _exports_for(py: Path, nav: dict[str, Any]) -> list[str]:
    """Derive the export list from ``__navmap__`` or ``__all__`` definitions."""
    nav_exports = nav.get("exports")
    if isinstance(nav_exports, list) and all(isinstance(x, str) for x in nav_exports):
        return list(dict.fromkeys(nav_exports))
    all_list = _parse_all(py)
    if all_list:
        return list(dict.fromkeys(all_list))
    return []


def _module_path(py: Path) -> str | None:
    """Return the dotted module path for ``py`` within ``src/`` if possible."""
    try:
        rel = py.relative_to(SRC)
    except Exception:
        return None
    if rel.suffix != ".py":
        return None
    return ".".join(rel.with_suffix("").parts)


def _validate_pep440(field_val: Any) -> str | None:
    """Validate PEP 440 version strings and report an error message when invalid."""
    if field_val is None or field_val == "":
        return None
    if Version is None:
        return "packaging not installed (cannot validate PEP 440)"
    try:
        Version(str(field_val))
        return None
    except InvalidVersion:
        return f"non-PEP440 version: {field_val!r}"


def _inspect(py: Path) -> list[str]:
    """Validate a module at ``py`` and return collected violation messages."""
    errs: list[str] = []
    nav = _parse_navmap_dict(py)
    sections_inline, anchors_inline = _scan_inline(py)
    exports = _exports_for(py, nav)

    # If module exports anything, __navmap__ must exist
    if exports and not nav:
        errs.append(f"{py}: module exports symbols but has no __navmap__")
        return errs

    # Sections: first must be public-api; kebab-case ids; anchors present for listed symbols
    sections = nav.get("sections", [])
    if sections:
        first = sections[0].get("id")
        if first != "public-api":
            errs.append(f"{py}: first navmap section must have id 'public-api'")
        for sec in sections:
            sid = sec.get("id", "")
            if sid and not SLUG_RE.match(sid):
                errs.append(f"{py}: section id '{sid}' is not kebab-case")
            for sym in sec.get("symbols", []):
                if not IDENT_RE.match(sym):
                    errs.append(f"{py}: invalid symbol name '{sym}' in section '{sid}'")
                elif sym not in anchors_inline:
                    errs.append(f"{py}: missing [nav:anchor] for section symbol '{sym}'")

    # Exports: if nav declares, it must match the actual exports setwise
    nav_exports = nav.get("exports", [])
    if isinstance(nav_exports, list):
        if set(x for x in nav_exports if isinstance(x, str)) != set(exports):
            errs.append(f"{py}: __navmap__['exports'] does not match __all__/exports set")

    # Per-export meta requirements (owner, stability, since/dep PEP 440)
    meta: dict[str, Any] = nav.get("symbols", {}) or {}
    for name in sorted(exports):
        m = meta.get(name, {})
        stab = m.get("stability")
        owner = m.get("owner")
        if stab not in STABILITY:
            errs.append(f"{py}: symbol '{name}' missing/invalid stability (got {stab!r})")
        if not owner:
            errs.append(f"{py}: symbol '{name}' missing owner (e.g., '@team')")
        # Versions
        since = m.get("since")
        deprec = m.get("deprecated_in")
        e1 = _validate_pep440(since)
        if e1:
            errs.append(f"{py}: symbol '{name}' since invalid: {e1}")
        e2 = _validate_pep440(deprec)
        if e2:
            errs.append(f"{py}: symbol '{name}' deprecated_in invalid: {e2}")
        if Version and since and deprec:
            try:
                if Version(str(deprec)) < Version(str(since)):
                    errs.append(f"{py}: symbol '{name}' deprecated_in ({deprec}) < since ({since})")
            except InvalidVersion:
                pass

    return errs


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
    
    errors: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        errors.extend(_inspect(py))

    if errors:
        print("\n".join(errors))
        return 1

    # Round-trip check: compare freshly built JSON to inline markers
    try:
        # Local import to avoid creating a hard dependency in module scope
        from tools.navmap.build_navmap import build_index  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"navmap check: unable to import build_navmap for round-trip ({e})", file=sys.stderr)
        return 1

    index = build_index(json_path=INDEX)  # also refreshes file on disk
    rt_errs: list[str] = []
    for mod, entry in (index.get("modules") or {}).items():
        p = REPO / entry.get("path", "")
        lines = _read_text(p)
        # Sections at recorded lines
        for sid, lineno in (entry.get("section_lines") or {}).items():
            if lineno < 1 or lineno > len(lines) or not SECTION_RE.match(lines[lineno - 1]):
                rt_errs.append(f"{p}: round-trip mismatch for section '{sid}' at line {lineno}")
        # Anchors at recorded lines
        for sym, lineno in (entry.get("anchors") or {}).items():
            if lineno < 1 or lineno > len(lines) or not ANCHOR_RE.match(lines[lineno - 1]):
                rt_errs.append(f"{p}: round-trip mismatch for anchor '{sym}' at line {lineno}")

    if rt_errs:
        print("\n".join(rt_errs))
        return 1

    print("navmap check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
