"""Overview of build symbol index.

This module bundles build symbol index logic for the kgfoundry stack. It
groups related helpers so downstream packages can import a single
cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from griffe import Object as GriffeObject
else:
    GriffeObject = object  # type: ignore[misc, assignment]

griffe_module = importlib.import_module("griffe")
loader_module: ModuleType | None
try:
    loader_module = importlib.import_module("griffe.loader")
except ModuleNotFoundError:
    loader_module = None
Object = cast(Any, griffe_module.Object)
default_loader = griffe_module.GriffeLoader
GriffeLoader = cast(
    Any,
    loader_module.GriffeLoader if loader_module is not None else default_loader,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS_BUILD = ROOT / "docs" / "_build"

from tools.detect_pkg import detect_packages, detect_primary  # noqa: E402

SRC = ROOT / "src"
ENV_PKGS = os.environ.get("DOCS_PKG")
LINK_MODE = os.environ.get("DOCS_LINK_MODE", "both").lower()
GITHUB_ORG = os.environ.get("DOCS_GITHUB_ORG")
GITHUB_REPO = os.environ.get("DOCS_GITHUB_REPO")
GITHUB_SHA = os.environ.get("DOCS_GITHUB_SHA")

NAVMAP_CANDIDATES = [
    DOCS_BUILD / "navmap.json",
    DOCS_BUILD / "navmap" / "navmap.json",
    ROOT / "site" / "_build" / "navmap" / "navmap.json",
]

loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


@dataclass(slots=True)
class NavLookup:
    """Model the NavLookup.

    Represent the navlookup data structure used throughout the project.
    The class encapsulates behaviour behind a well-defined interface for
    collaborating components. Instances are typically created by
    factories or runtime orchestrators documented nearby.
    """

    symbol_meta: dict[str, dict[str, Any]]
    module_meta: dict[str, dict[str, Any]]
    sections: dict[str, str]


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from docs._scripts.build_symbol_index import iter_packages
    >>> result = iter_packages()
    >>> result  # doctest: +ELLIPSIS
    """
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    packages = detect_packages()
    return packages or [detect_primary()]


def safe_attr(node: object, attr: str, default: object | None = None) -> object | None:
    """Compute safe attr.

    Carry out the safe attr operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.
    attr : str
        Description for ``attr``.
    default : object | None
        Optional parameter default ``None``. Description for ``default``.

    Returns
    -------
    object | None
        Description of return value.

    Examples
    --------
    >>> from docs._scripts.build_symbol_index import safe_attr
    >>> result = safe_attr(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        return getattr(node, attr)
    except Exception:
        return default


def _module_for(path: str | None, kind: str) -> str | None:
    """Return the module path for ``path`` given the object ``kind``."""
    if not path:
        return None
    if kind in {"module", "package"}:
        return path
    if "." in path:
        return path.rsplit(".", 1)[0]
    return path


def _package_for(module: str | None, path: str | None) -> str | None:
    """Return the top-level package for a module or object path."""
    target = module or path
    if not target:
        return None
    return target.split(".", 1)[0]


def _canonical_path(node: object) -> str | None:
    """Return the canonical path for ``node`` if available."""
    canonical = safe_attr(node, "canonical_path")
    if isinstance(canonical, Object):
        return canonical.path
    if canonical is None:
        return None
    return str(canonical)


def _string_signature(node: object) -> str | None:
    """Return a printable signature for callable ``node`` objects."""
    signature = safe_attr(node, "signature")
    if signature is None:
        return None
    return str(signature)


def _normalize_lineno(value: object | None) -> int | None:
    """Normalize a ``lineno``-like value into an ``int`` when possible."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _join_symbol(module: str, symbol: str) -> str:
    """Return a fully qualified symbol path from ``module`` and ``symbol``."""
    if not symbol:
        return module
    if symbol.startswith(module):
        return symbol
    if "." in symbol:
        return symbol
    return f"{module}.{symbol}"


def _load_navmap() -> NavLookup:
    """Load the NavMap index if present and return lookup tables."""
    for candidate in NAVMAP_CANDIDATES:
        try:
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return _index_navmap(data)
        except json.JSONDecodeError:
            continue
    return NavLookup(symbol_meta={}, module_meta={}, sections={})


def _clean_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``meta`` with ``None`` entries removed."""
    if not isinstance(meta, Mapping):
        return {}
    return {key: value for key, value in meta.items() if value is not None}


def _record_module_defaults(
    module_name: str,
    payload: Mapping[str, object],
    module_meta: MutableMapping[str, dict[str, Any]],
    symbol_meta: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract module-level defaults and prime symbol metadata with them."""
    defaults = _clean_meta(cast(Mapping[str, object] | None, payload.get("module_meta")))
    module_meta[module_name] = defaults
    if defaults:
        symbol_meta.setdefault(module_name, dict(defaults))
    return defaults


def _record_symbol_meta(
    module_name: str,
    per_symbol_meta: Mapping[str, object] | None,
    symbol_meta: MutableMapping[str, dict[str, Any]],
) -> None:
    """Merge per-symbol metadata into the index."""
    if not isinstance(per_symbol_meta, Mapping):
        return
    for name, meta in per_symbol_meta.items():
        if not isinstance(name, str) or not isinstance(meta, Mapping):
            continue
        fq_name = _join_symbol(module_name, name)
        symbol_meta[fq_name] = _clean_meta(meta)


def _record_sections(
    module_name: str,
    sections_payload: Sequence[object] | None,
    sections: MutableMapping[str, str],
) -> None:
    """Associate section ids with fully qualified symbol names."""
    if sections_payload is None:
        return
    for section_obj in sections_payload:
        if not isinstance(section_obj, Mapping):
            continue
        section_id = section_obj.get("id")
        if not isinstance(section_id, str):
            continue
        symbols = section_obj.get("symbols")
        if not isinstance(symbols, Sequence):
            continue
        for symbol in symbols:
            if not isinstance(symbol, str):
                continue
            fq_name = _join_symbol(module_name, symbol)
            sections[fq_name] = section_id


def _index_navmap(data: Mapping[str, object]) -> NavLookup:
    """Index NavMap metadata for symbol enrichment."""
    symbol_meta: dict[str, dict[str, Any]] = {}
    module_meta: dict[str, dict[str, Any]] = {}
    sections: dict[str, str] = {}

    modules = data.get("modules")
    if not isinstance(modules, Mapping):
        return NavLookup(symbol_meta, module_meta, sections)

    for module_name, payload in modules.items():
        if not isinstance(module_name, str) or not isinstance(payload, Mapping):
            continue

        _record_module_defaults(module_name, payload, module_meta, symbol_meta)
        _record_symbol_meta(
            module_name, cast(Mapping[str, object] | None, payload.get("meta")), symbol_meta
        )
        _record_sections(
            module_name,
            cast(Sequence[object] | None, payload.get("sections")),
            sections,
        )

    return NavLookup(symbol_meta=symbol_meta, module_meta=module_meta, sections=sections)


def _load_test_map() -> dict[str, Any]:
    """Return the optional test map produced earlier in the docs pipeline."""
    test_map_path = DOCS_BUILD / "test_map.json"
    if test_map_path.exists():
        try:
            data = json.loads(test_map_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return {}
    return {}


def _current_sha() -> str:
    """Resolve the Git SHA used for GitHub permalinks."""
    if GITHUB_SHA:
        return GITHUB_SHA
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover - fallback for detached states
        return "HEAD"


def _github_link(file_rel: str, start: int | None, end: int | None) -> str | None:
    """Return a commit-stable GitHub permalink if GitHub metadata is
    configured.
    """
    if not (GITHUB_ORG and GITHUB_REPO):
        return None
    sha = _current_sha()
    fragment = ""
    if start and end and end >= start:
        fragment = f"#L{start}-L{end}"
    elif start:
        fragment = f"#L{start}"
    return f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}/blob/{sha}/{file_rel}{fragment}"


def _source_links(file_rel: str | None, start: int | None, end: int | None) -> dict[str, str]:
    """Build the source link bundle (editor/github) for a symbol row."""
    if not file_rel:
        return {}

    links: dict[str, str] = {}
    abs_path = (ROOT / file_rel).resolve()
    start_line = start or 1

    if LINK_MODE in ("editor", "both"):
        links["editor"] = f"vscode://file/{abs_path}:{start_line}:1"

    if LINK_MODE in ("github", "both"):
        gh = _github_link(file_rel, start, end)
        if gh:
            links["github"] = gh

    return links


def _meta_value(
    symbol_meta: Mapping[str, object] | None,
    module_defaults: Mapping[str, object] | None,
    key: str,
) -> object | None:
    """Return the metadata value for ``key`` using symbol overrides then module
    defaults.
    """
    if symbol_meta and key in symbol_meta and symbol_meta[key] is not None:
        return symbol_meta[key]
    if module_defaults and key in module_defaults and module_defaults[key] is not None:
        return module_defaults[key]
    return None


def _doc_first_paragraph(node: object) -> str:
    """Return the first paragraph of a node's docstring."""
    doc = safe_attr(node, "docstring")
    if doc is None:
        return ""
    value = getattr(doc, "value", None)
    if isinstance(value, str):
        text = value.strip()
        if text:
            first = text.split("\n\n", 1)[0]
            return first.strip()
    return ""


def _kind_name(node: object) -> str | None:
    """Return the normalized kind string for a node."""
    kind_obj = safe_attr(node, "kind")
    if isinstance(kind_obj, str):
        return kind_obj
    value = safe_attr(kind_obj, "value")
    return value if isinstance(value, str) else None


def _relative_file(node: object) -> str | None:
    """Return a node's relative filename if available."""
    file_rel_obj = safe_attr(node, "relative_package_filepath") or safe_attr(
        node, "relative_filepath"
    )
    if isinstance(file_rel_obj, Path):
        return str(file_rel_obj)
    return str(file_rel_obj) if isinstance(file_rel_obj, str) else None


def _normalize_tests(value: object) -> list[str]:
    """Normalize the ``tested_by`` payload into a list of string labels."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _row_section(
    nav: NavLookup,
    module: str | None,
    path: str,
    canonical: str | None,
    kind: str,
) -> str | None:
    """Resolve the section id for a symbol."""
    direct = nav.sections.get(path)
    if direct:
        return direct
    if canonical:
        canonical_section = nav.sections.get(canonical)
        if canonical_section:
            return canonical_section
    if module:
        module_section = nav.sections.get(module)
        if module_section:
            return module_section
    if kind == "module":
        return "module"
    if kind == "class":
        return "class"
    return None


def _iter_members(node: object) -> Iterable[GriffeObject]:
    """Yield member nodes from a griffe object."""
    members_attr = safe_attr(node, "members")
    if not isinstance(members_attr, Mapping):
        return ()
    try:
        return tuple(cast(Iterable[GriffeObject], members_attr.values()))
    except Exception:  # pragma: no cover - fallback when values() unavailable
        return ()


def _row_from_node(
    node: GriffeObject,
    nav: NavLookup,
    test_map: Mapping[str, object],
) -> dict[str, Any] | None:
    """Construct a row describing ``node`` if it has a valid path."""
    path = getattr(node, "path", None)
    if not isinstance(path, str):
        return None

    kind = _kind_name(node)
    if kind is None:
        return None

    module = _module_for(path, kind)
    package = _package_for(module, path)
    file_rel = _relative_file(node)

    lineno = _normalize_lineno(safe_attr(node, "lineno"))
    endlineno = _normalize_lineno(safe_attr(node, "endlineno"))

    canonical = _canonical_path(node)
    symbol_meta = nav.symbol_meta.get(path)
    if symbol_meta is None and canonical:
        symbol_meta = nav.symbol_meta.get(canonical)
    module_defaults = nav.module_meta.get(module or "")

    tested_by = _normalize_tests(test_map.get(path))
    if not tested_by and canonical:
        tested_by = _normalize_tests(test_map.get(canonical))

    row: dict[str, Any] = {
        "path": path,
        "canonical_path": canonical,
        "kind": kind,
        "package": package,
        "module": module,
        "file": file_rel,
        "lineno": lineno,
        "endlineno": endlineno,
        "doc": _doc_first_paragraph(node),
        "signature": _string_signature(node),
        "is_async": bool(safe_attr(node, "is_async")),
        "is_property": kind == "property",
        "owner": _meta_value(symbol_meta, module_defaults, "owner"),
        "stability": _meta_value(symbol_meta, module_defaults, "stability"),
        "since": _meta_value(symbol_meta, module_defaults, "since"),
        "deprecated_in": _meta_value(symbol_meta, module_defaults, "deprecated_in"),
        "section": _row_section(nav, module, path, canonical, kind),
        "tested_by": tested_by,
        "source_link": _source_links(file_rel, lineno, endlineno),
    }
    return row


def _collect_rows(nav: NavLookup, test_map: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Traverse packages and return enriched symbol rows."""
    rows: dict[str, dict[str, Any]] = {}

    def _walk(node: GriffeObject) -> None:
        row = _row_from_node(node, nav, test_map)
        if row is not None:
            path_value = row.get("path")
            if isinstance(path_value, str):
                rows[path_value] = row
        for member in _iter_members(node):
            _walk(member)

    for pkg in iter_packages():
        root = loader.load(pkg)
        _walk(root)

    return [rows[key] for key in sorted(rows)]


def _build_reverse_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build reverse lookup tables keyed by file and module."""
    by_file: dict[str, set[str]] = defaultdict(set)
    by_module: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        path = row.get("path")
        if not isinstance(path, str):
            continue

        file_rel = row.get("file")
        if isinstance(file_rel, str):
            by_file[file_rel].add(path)

        module = row.get("module")
        if isinstance(module, str):
            by_module[module].add(path)

    by_file_sorted = {k: sorted(v) for k, v in sorted(by_file.items())}
    by_module_sorted = {k: sorted(v) for k, v in sorted(by_module.items())}

    return by_file_sorted, by_module_sorted


def _write_json_if_changed(path: Path, data: object) -> bool:
    """Write ``data`` to ``path`` if the serialized content changed."""
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == serialized:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return True


def main() -> int:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    int
        Description of return value.

    Examples
    --------
    >>> from docs._scripts.build_symbol_index import main
    >>> result = main()
    >>> result  # doctest: +ELLIPSIS
    """
    nav_lookup = _load_navmap()
    test_map = _load_test_map()

    rows = _collect_rows(nav_lookup, test_map)
    by_file, by_module = _build_reverse_maps(rows)

    symbols_path = DOCS_BUILD / "symbols.json"
    by_file_path = DOCS_BUILD / "by_file.json"
    by_module_path = DOCS_BUILD / "by_module.json"

    wrote_symbols = _write_json_if_changed(symbols_path, rows)
    wrote_by_file = _write_json_if_changed(by_file_path, by_file)
    wrote_by_module = _write_json_if_changed(by_module_path, by_module)

    status = []
    status.append(
        f"{'Updated' if wrote_symbols else 'Unchanged'} {symbols_path} ({len(rows)} entries)"
    )
    status.append(f"{'Updated' if wrote_by_file else 'Unchanged'} {by_file_path}")
    status.append(f"{'Updated' if wrote_by_module else 'Unchanged'} {by_module_path}")

    print("; ".join(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
