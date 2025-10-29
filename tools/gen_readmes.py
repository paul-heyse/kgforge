"""Overview of gen readmes.

This module bundles gen readmes logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.griffe_utils import resolve_griffe  # noqa: E402

_griffe_module, _griffe_object_type, GriffeLoader = resolve_griffe()


class DocstringLike(Protocol):
    """Minimal interface of ``griffe`` docstrings used by this module."""

    value: str | None


class KindLike(Protocol):
    """Subset of the ``griffe`` kind enumeration accessed during rendering."""

    value: str


class BaseLike(Protocol):
    """Shape of base classes referenced when classifying exceptions."""

    full: str | None
    name: str | None


class GriffeObjectLike(Protocol):
    """Structural type describing the ``griffe`` objects we inspect."""

    path: str
    name: str
    docstring: DocstringLike | None
    kind: KindLike | None
    members: Mapping[str, GriffeObjectLike] | None
    relative_package_filepath: str | None
    lineno: int | None
    endlineno: int | None
    bases: Sequence[BaseLike] | None
    is_package: bool | None


from tools.detect_pkg import detect_packages, detect_primary  # noqa: E402

SRC = ROOT / "src"
NAVMAP_PATH = ROOT / "site" / "_build" / "navmap" / "navmap.json"
TESTMAP_PATH = ROOT / "docs" / "_build" / "test_map.json"

DEFAULT_SYNOPSIS = "Package synopsis not yet documented."
MIN_REMOTE_PARTS: Final[int] = 2
README_WRAP_COLUMN: Final[int] = 80
README_BADGE_INDENT: Final[int] = 4


def detect_repo() -> tuple[str, str]:
    """Compute detect repo.

    Carry out the detect repo operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    Tuple[str, str]
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import detect_repo
    >>> result = detect_repo()
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        remote = ""

    override_owner = os.environ.get("DOCS_GITHUB_ORG")
    override_repo = os.environ.get("DOCS_GITHUB_REPO")
    if override_owner and override_repo:
        return override_owner, override_repo

    if remote.endswith(".git"):
        remote = remote[:-4]
    path = ""
    if remote.startswith("git@"):
        _, remainder = remote.split("@", 1)
        path = remainder.split(":", 1)[1]
    else:
        parsed_remote = urlparse(remote)
        if parsed_remote.path:
            path = parsed_remote.path.lstrip("/")

    if path:
        parts = path.split("/")
        if len(parts) >= MIN_REMOTE_PARTS:
            return parts[0], parts[1]

    return override_owner or "your-org", override_repo or "your-repo"


def git_sha() -> str:
    """Compute git sha.

    Carry out the git sha operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import git_sha
    >>> result = git_sha()
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")


OWNER, REPO = detect_repo()
SHA = git_sha()


def gh_url(rel_path: str, start: int, end: int | None) -> str:
    """Compute gh url.

    Carry out the gh url operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    rel_path : str
        Description for ``rel_path``.
    start : int
        Description for ``start``.
    end : int | None
        Description for ``end``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import gh_url
    >>> result = gh_url(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    fragment = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel_path}{fragment}"


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import iter_packages
    >>> result = iter_packages()
    >>> result  # doctest: +ELLIPSIS
    """
    env_pkgs = os.environ.get("DOCS_PKG")
    if env_pkgs:
        return [pkg.strip() for pkg in env_pkgs.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


def summarize(node: GriffeObjectLike) -> str:
    """Compute summarize.

    Carry out the summarize operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import summarize
    >>> result = summarize(...)
    >>> result  # doctest: +ELLIPSIS
    """
    doc = node.docstring
    if doc is None or not doc.value:
        return ""
    raw = doc.value.strip()
    if not raw:
        return ""
    # Find the first non-empty line before attempting to split into sentences.
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if not first_line:
        return ""
    match = re.search(r"(?<=[.!?])\s", first_line)
    if match:
        return first_line[: match.start()].strip()
    return first_line


def is_public(node: GriffeObjectLike) -> bool:
    """Compute is public.

    Carry out the is public operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import is_public
    >>> result = is_public(...)
    >>> result  # doctest: +ELLIPSIS
    """
    return not node.name.startswith("_")


def get_open_link(node: GriffeObjectLike, readme_dir: Path) -> str | None:
    """Compute get open link.

    Carry out the get open link operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.
    readme_dir : Path
        Description for ``readme_dir``.

    Returns
    -------
    str | None
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import get_open_link
    >>> result = get_open_link(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    rel_path = node.relative_package_filepath
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        relative = abs_path.relative_to(readme_dir).as_posix()
    except ValueError:
        return None
    start = int(node.lineno or 1)
    return f"./{relative}:{start}:1"


def get_view_link(node: GriffeObjectLike) -> str | None:
    """Compute get view link.

    Carry out the get view link operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Returns
    -------
    str | None
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import get_view_link
    >>> result = get_view_link(...)
    >>> result  # doctest: +ELLIPSIS
    """
    rel_path = node.relative_package_filepath
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        rel = abs_path.relative_to(ROOT)
    except ValueError:
        return None
    start = int(node.lineno or 1)
    end = node.endlineno
    return gh_url(str(rel).replace("\\", "/"), start, end)


def iter_public_members(node: GriffeObjectLike) -> list[GriffeObjectLike]:
    """Compute iter public members.

    Carry out the iter public members operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Returns
    -------
    collections.abc.Iterable
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import iter_public_members
    >>> result = iter_public_members(...)
    >>> result  # doctest: +ELLIPSIS
    """
    members = node.members
    if not members:
        return []
    public = [member for member in members.values() if is_public(member)]

    def _member_key(member: GriffeObjectLike) -> str:
        """Member key.

        Parameters
        ----------
        member : GriffeObjectLike
            Description.

        Returns
        -------
        str
            Description.

        Raises
        ------
        Exception
            Description.

        Examples
        --------
        >>> _member_key(...)
        """
        if member.path:
            return member.path
        return member.name

    return sorted(public, key=_member_key)


def _load_json(path: Path) -> dict[str, Any]:
    """Load json.

    Parameters
    ----------
    path : Path
        Description.

    Returns
    -------
    dict[str, Any]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _load_json(...)
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


NAVMAP = _load_json(NAVMAP_PATH)
TEST_MAP = _load_json(TESTMAP_PATH)


@dataclass(frozen=True)
class Config:
    """Model the Config.

    Represent the config data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    packages: list[str]
    link_mode: str  # github | editor | both
    editor: str  # vscode | relative
    fail_on_metadata_miss: bool
    dry_run: bool
    verbose: bool
    run_doctoc: bool


@dataclass(frozen=True)
class Badges:
    """Model the Badges.

    Represent the badges data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    stability: str | None = None
    owner: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    tested_by: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class NavMatch:
    """Computed data for a symbol while scanning the NavMap."""

    symbol_meta: dict[str, Any]
    defaults: dict[str, Any]
    matches_symbol_list: bool
    matches_prefix: bool


def _module_defaults(module: Mapping[str, Any]) -> dict[str, Any]:
    """Return module-level defaults for badge metadata."""
    module_meta = module.get("module_meta")
    if isinstance(module_meta, Mapping):
        return dict(module_meta)
    defaults: dict[str, Any] = {}
    for key in ("owner", "stability", "since", "deprecated_in"):
        value = module.get(key)
        if value is not None:
            defaults[key] = value
    return defaults


def _section_for_symbol(module: Mapping[str, Any], symbol: str) -> str | None:
    """Return the section identifier for ``symbol`` when present."""
    sections = module.get("sections")
    if not isinstance(sections, Sequence):
        return None
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        section_id = section.get("id")
        symbols = section.get("symbols")
        if (
            isinstance(section_id, str)
            and isinstance(symbols, Sequence)
            and any(entry == symbol for entry in symbols)
        ):
            return section_id
    return None


def _module_match_for_symbol(
    module_id: str,
    module: Mapping[str, Any],
    qname: str,
    symbol: str,
) -> NavMatch:
    """Return per-module metadata for ``symbol``."""
    symbol_meta: dict[str, Any] = {}
    meta = module.get("meta")
    if isinstance(meta, Mapping):
        candidate = meta.get(qname) or meta.get(symbol)
        if isinstance(candidate, Mapping):
            symbol_meta = dict(candidate)
            if "section" not in symbol_meta:
                section_id = _section_for_symbol(module, symbol)
                if section_id:
                    symbol_meta = {**symbol_meta, "section": section_id}

    defaults = _module_defaults(module)

    symbol_entries = module.get("symbols")
    matches_symbol_list = False
    if isinstance(symbol_entries, Sequence):
        matches_symbol_list = any(entry == symbol for entry in symbol_entries)

    matches_prefix = qname.startswith(module_id)

    return NavMatch(
        symbol_meta=symbol_meta,
        defaults=defaults,
        matches_symbol_list=matches_symbol_list,
        matches_prefix=matches_prefix,
    )


def parse_config() -> Config:
    """Compute parse config.

    Carry out the parse config operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    Config
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import parse_config
    >>> result = parse_config()
    >>> result  # doctest: +ELLIPSIS
    """
    parser = argparse.ArgumentParser(description="Generate per-package README files.")
    parser.add_argument("--packages", default=os.getenv("DOCS_PKG", ""))
    parser.add_argument(
        "--link-mode",
        default=os.getenv("DOCS_LINK_MODE", "both"),
        choices=["github", "editor", "both"],
    )
    parser.add_argument(
        "--editor",
        default=os.getenv("DOCS_EDITOR", "vscode"),
        choices=["vscode", "relative"],
    )
    parser.add_argument("--fail-on-metadata-miss", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--run-doctoc", action="store_true", default=False)
    args = parser.parse_args()

    packages = (
        [pkg.strip() for pkg in args.packages.split(",") if pkg.strip()]
        if args.packages
        else iter_packages()
    )
    return Config(
        packages=packages,
        link_mode=args.link_mode,
        editor=args.editor,
        fail_on_metadata_miss=args.fail_on_metadata_miss,
        dry_run=args.dry_run,
        verbose=args.verbose,
        run_doctoc=args.run_doctoc,
    )


def _lookup_nav(qname: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return symbol metadata and module defaults from the NavMap.

    The NavMap JSON generated by ``tools/navmap/build_navmap.py`` has the
    structure::

        {
            "modules": {
                "package.module": {
                    "meta": {"package.module.symbol": {...}},
                    "module_meta": {...},
                    "sections": [
                        {"id": "storage", "symbols": ["symbol"]},
                    ],
                }
            }
        }

    ``meta`` entries are per-symbol overrides; ``module_meta`` supplies default
    values that cascade to every symbol in the module.  We normalise the lookup
    so ``badges_for`` can merge overrides with defaults seamlessly.
    """
    modules = NAVMAP.get("modules", {}) if isinstance(NAVMAP, Mapping) else {}
    if not isinstance(modules, Mapping):
        return {}, {}

    symbol = qname.split(".")[-1]
    best_defaults: dict[str, Any] = {}
    for module_id, module in modules.items():
        if not isinstance(module_id, str) or not isinstance(module, Mapping):
            continue
        match = _module_match_for_symbol(module_id, module, qname, symbol)
        if match.symbol_meta:
            return match.symbol_meta, match.defaults
        if match.defaults and (match.matches_symbol_list or match.matches_prefix):
            return {}, match.defaults
        if match.defaults and match.matches_prefix:
            best_defaults = match.defaults
    if best_defaults:
        return {}, best_defaults
    return {}, {}


def badges_for(qname: str) -> Badges:
    """Compute badges for.

    Carry out the badges for operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    qname : str
        Description for ``qname``.

    Returns
    -------
    Badges
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import badges_for
    >>> result = badges_for(...)
    >>> result  # doctest: +ELLIPSIS
    """
    symbol_meta, defaults = _lookup_nav(qname)
    merged = {**defaults, **symbol_meta}
    tests: list[dict[str, Any]] = []
    if isinstance(TEST_MAP, dict):
        recorded = TEST_MAP.get(qname) or TEST_MAP.get(qname.split(".")[-1])
        if isinstance(recorded, list):
            tests = [entry for entry in recorded if isinstance(entry, dict)][:3]
    return Badges(
        stability=merged.get("stability"),
        owner=merged.get("owner"),
        section=merged.get("section"),
        since=merged.get("since"),
        deprecated_in=merged.get("deprecated_in"),
        tested_by=tests,
    )


def _format_test_badge(entries: list[dict[str, Any]] | None) -> str | None:
    """Format the ``tested-by`` badge snippet when entries exist."""
    if not entries:
        return None
    formatted: list[str] = []
    for entry in entries:
        file = entry.get("file")
        lines = entry.get("lines")
        if not file:
            continue
        if isinstance(lines, list) and lines:
            formatted.append(f"{file}:{lines[0]}")
        else:
            formatted.append(file)
    if not formatted:
        return None
    return "`tested-by: " + ", ".join(formatted) + "`"


def _badge_parts(badge: Badges) -> list[str]:
    """Return textual fragments that make up the badge line."""
    attributes = [
        ("stability", "stability"),
        ("owner", "owner"),
        ("section", "section"),
        ("since", "since"),
        ("deprecated_in", "deprecated"),
    ]
    parts = [f"`{label}:{value}`" for attr, label in attributes if (value := getattr(badge, attr))]
    test_badge = _format_test_badge(badge.tested_by)
    if test_badge:
        parts.append(test_badge)
    return parts


def _wrap_badge_parts(parts: Sequence[str]) -> list[str]:
    """Wrap badge fragments to respect the configured line width."""
    wrapped: list[str] = []
    current: list[str] = []
    current_len = 0
    available = README_WRAP_COLUMN - README_BADGE_INDENT
    for part in parts:
        part_len = len(part) + (1 if current else 0)
        if current and current_len + part_len > available:
            wrapped.append(" ".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += part_len
    if current:
        wrapped.append(" ".join(current))
    return wrapped


def format_badges(qname: str, base_length: int = 0) -> str:
    """Compute format badges.

    Carry out the format badges operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    qname : str
        Description for ``qname``.
    base_length : int | None
        Optional parameter default ``0``. Description for ``base_length``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import format_badges
    >>> result = format_badges(...)
    >>> result  # doctest: +ELLIPSIS
    """
    badge = badges_for(qname)
    parts = _badge_parts(badge)
    if not parts:
        return ""
    badge_line = " ".join(parts)
    if not base_length or base_length + 1 + len(badge_line) <= README_WRAP_COLUMN:
        return " " + badge_line
    wrapped = _wrap_badge_parts(parts)
    return "\n    " + "\n    ".join(wrapped)


def editor_link(abs_path: Path, lineno: int, editor_mode: str) -> str | None:
    """Compute editor link.

    Carry out the editor link operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    abs_path : Path
        Description for ``abs_path``.
    lineno : int
        Description for ``lineno``.
    editor_mode : str
        Description for ``editor_mode``.

    Returns
    -------
    str | None
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import editor_link
    >>> result = editor_link(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    if editor_mode == "vscode":
        return f"vscode://file/{abs_path}:{lineno}:1"
    if editor_mode == "relative":
        try:
            rel = abs_path.relative_to(ROOT)
        except ValueError:
            rel = abs_path
        return f"./{rel.as_posix()}:{lineno}:1"
    return None


def _is_exception(node: GriffeObjectLike) -> bool:
    """Is exception.

    Parameters
    ----------
    node : Object
        Description.

    Returns
    -------
    bool
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _is_exception(...)
    """
    kind = node.kind.value if node.kind else ""
    if kind != "class":
        return False
    if node.name.endswith(("Error", "Exception")):
        return True
    for base in node.bases or []:
        base_name = base.full or base.name
        if base_name and base_name.endswith(("Error", "Exception")):
            return True
    return False


KINDS = {"module", "package", "class", "function"}


def bucket_for(node: GriffeObjectLike) -> str:
    """Compute bucket for.

    Carry out the bucket for operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import bucket_for
    >>> result = bucket_for(...)
    >>> result  # doctest: +ELLIPSIS
    """
    kind = node.kind.value if node.kind else ""
    if kind in {"module", "package"}:
        return "Modules"
    if kind == "class":
        return "Exceptions" if _is_exception(node) else "Classes"
    if kind == "function":
        return "Functions"
    return "Other"


def render_line(node: GriffeObjectLike, readme_dir: Path, cfg: Config) -> str | None:
    """Compute render line.

    Carry out the render line operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.
    readme_dir : Path
        Description for ``readme_dir``.
    cfg : Config
        Description for ``cfg``.

    Returns
    -------
    str | None
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import render_line
    >>> result = render_line(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    qname = node.path
    summary = summarize(node)

    open_link = get_open_link(node, readme_dir) if cfg.link_mode in {"editor", "both"} else None
    view_link = get_view_link(node) if cfg.link_mode in {"github", "both"} else None

    if cfg.link_mode in {"editor", "both"} and node.relative_package_filepath:
        base = SRC if SRC.exists() else ROOT
        abs_path = (base / node.relative_package_filepath).resolve()
        direct = editor_link(abs_path, int(node.lineno or 1), cfg.editor)
        if direct:
            open_link = direct

    if not (open_link or view_link):
        return None

    line = f"- **`{qname}`**"
    if summary:
        line += f" — {summary}"
    badge_text = format_badges(qname, len(line))
    if badge_text:
        line += badge_text

    links: list[str] = []
    if open_link:
        links.append(f"[open]({open_link})")
    if view_link:
        links.append(f"[view]({view_link})")
    tail = f" → {' | '.join(links)}" if links else ""
    return line + tail + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Compute write if changed.

    Carry out the write if changed operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
        Description for ``path``.
    content : str
        Description for ``content``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import write_if_changed
    >>> result = write_if_changed(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    rendered = content.rstrip() + f"\n<!-- agent:readme v1 sha:{SHA} content:{digest} -->\n"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if previous == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def write_readme(node: GriffeObjectLike, cfg: Config) -> bool:
    """Compute write readme.

    Carry out the write readme operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.
    cfg : Config
        Description for ``cfg``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.gen_readmes import write_readme
    >>> result = write_readme(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"

    buckets: dict[str, list[str]] = {
        name: [] for name in ("Modules", "Classes", "Functions", "Exceptions", "Other")
    }
    children = [
        child for child in iter_public_members(node) if child.kind and child.kind.value in KINDS
    ]

    for child in sorted(children, key=lambda child: child.path):
        line = render_line(child, pkg_dir, cfg)
        if line:
            buckets[bucket_for(child)].append(line)

    lines: list[str] = [f"# `{node.path}`\n\n"]
    synopsis = summarize(node) or DEFAULT_SYNOPSIS
    lines.append(f"{synopsis}\n\n")
    lines.extend(
        [
            "<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n",
            "<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n\n",
        ]
    )

    for section in ("Modules", "Classes", "Functions", "Exceptions", "Other"):
        items = buckets.get(section, [])
        if items:
            lines.append(f"## {section}\n\n")
            lines.extend(items)
            lines.append("\n")

    content = "".join(lines).rstrip() + "\n"
    if cfg.dry_run:
        print(f"[dry-run] would write {readme}")
        return False
    changed = write_if_changed(readme, content)
    if changed:
        print(f"Wrote {readme}")
        _maybe_run_doctoc(readme, cfg)
    return changed


def _maybe_run_doctoc(readme: Path, cfg: Config) -> None:
    """Run DocToc when enabled via ``--run-doctoc``."""
    if not cfg.run_doctoc:
        return
    doctoc = shutil.which("doctoc")
    if not doctoc:
        print(f"Info: doctoc not installed; skipping TOC update for {readme}")
        return
    result = subprocess.run(
        [doctoc, str(readme)],
        check=False,
        capture_output=True,
        text=True,
    )
    if cfg.verbose:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        print(
            f"Warning: doctoc exited with code {result.returncode} for {readme}",
            file=sys.stderr,
        )


def _collect_missing_metadata(node: GriffeObjectLike, missing: set[str]) -> None:
    """Collect missing metadata.

    Parameters
    ----------
    node : Object
        Description.
    missing : set[str]
        Description.

    Returns
    -------
    None
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _collect_missing_metadata(...)
    """
    for child in iter_public_members(node):
        kind = child.kind.value if child.kind else ""
        if kind in KINDS:
            qname = child.path
            badge = badges_for(qname)
            if not badge.stability or not badge.owner:
                missing.add(qname)
        if kind in {"module", "package"}:
            _collect_missing_metadata(child, missing)


def _ensure_packages_selected(packages: Sequence[str]) -> None:
    """Exit when no packages are available for processing."""
    if packages:
        return
    message = "No packages detected; set DOCS_PKG or add packages under src/."
    raise SystemExit(message)


def _warn_missing_inputs() -> None:
    """Emit warnings when auxiliary metadata files are missing."""
    if not NAVMAP_PATH.exists():
        print(f"Warning: NavMap not found at {NAVMAP_PATH}; badges will be empty")
    if not TESTMAP_PATH.exists():
        print(f"Warning: Test map not found at {TESTMAP_PATH}; tested-by badges will be empty")


def _process_module(module: GriffeObjectLike, cfg: Config, missing_meta: set[str]) -> bool:
    """Render README files for ``module`` and its package members."""
    changed = False
    if cfg.fail_on_metadata_miss:
        _collect_missing_metadata(module, missing_meta)
    changed |= write_readme(module, cfg)

    members = module.members
    if not members:
        return changed

    for member in members.values():
        if not member.is_package:
            continue
        if cfg.fail_on_metadata_miss:
            _collect_missing_metadata(member, missing_meta)
        changed |= write_readme(member, cfg)
    return changed


def _report_duration(start: float, changed_any: bool) -> None:
    """Print a timing summary when verbose mode is enabled."""
    duration = time.time() - start
    print(f"completed in {duration:.2f}s; changed={changed_any}")


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Raises
    ------
    SystemExit
        Raised when validation fails.

    Examples
    --------
    >>> from tools.gen_readmes import main
    >>> main()  # doctest: +ELLIPSIS
    """
    cfg = parse_config()
    _ensure_packages_selected(cfg.packages)
    _warn_missing_inputs()

    loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])
    missing_meta: set[str] = set()
    changed_any = False
    start = time.time()

    for pkg in cfg.packages:
        module = loader.load(pkg)
        changed_any |= _process_module(module, cfg, missing_meta)

    if cfg.fail_on_metadata_miss and missing_meta:
        print(
            "ERROR: Missing owner/stability for public symbols:\n  - "
            + "\n  - ".join(sorted(missing_meta))
        )
        raise SystemExit(2)

    if cfg.verbose:
        _report_duration(start, changed_any)


if __name__ == "__main__":
    main()
