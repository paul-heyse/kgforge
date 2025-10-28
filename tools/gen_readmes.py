"""Generate enriched package READMEs from static API metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from griffe import Object

try:
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

from detect_pkg import detect_packages, detect_primary

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NAVMAP_PATH = ROOT / "site" / "_build" / "navmap" / "navmap.json"
TESTMAP_PATH = ROOT / "docs" / "_build" / "test_map.json"


def detect_repo() -> tuple[str, str]:
    """Compute detect repo.

    Carry out the detect repo operation.

    Returns
    -------
    Tuple[str, str]
        Description of return value.
    """
    
    
    
    
    





    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        remote = ""

    owner_override = os.environ.get("DOCS_GITHUB_ORG")
    repo_override = os.environ.get("DOCS_GITHUB_REPO")
    if owner_override and repo_override:
        return owner_override, repo_override

    if remote.endswith(".git"):
        remote = remote[:-4]
    path = ""
    if remote.startswith("git@"):
        _, remainder = remote.split("@", 1)
        path = remainder.split(":", 1)[1]
    elif remote.startswith("https://"):
        path = remote.split("https://", 1)[1]

    if path:
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]

    return owner_override or "your-org", repo_override or "your-repo"


def git_sha() -> str:
    """Compute git sha.

    Carry out the git sha operation.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    





    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")


OWNER, REPO = detect_repo()
SHA = git_sha()


def gh_url(rel: str, start: int, end: int | None) -> str:
    """Compute gh url.

    Carry out the gh url operation.

    Parameters
    ----------
    rel : str
        Description for ``rel``.
    start : int
        Description for ``start``.
    end : int | None
        Description for ``end``.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    





    fragment = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{fragment}"


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    





    env_pkgs = os.environ.get("DOCS_PKG")
    if env_pkgs:
        return [pkg.strip() for pkg in env_pkgs.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


def summarize(node: Object) -> str:
    """Compute summarize.

    Carry out the summarize operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    





    doc = getattr(node, "docstring", None)
    if doc and getattr(doc, "value", None):
        return doc.value.strip().splitlines()[0].strip().rstrip(".")
    return ""


def is_public(node: Object) -> bool:
    """Compute is public.

    Carry out the is public operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.

    Returns
    -------
    bool
        Description of return value.
    """
    
    
    
    
    





    name = getattr(node, "name", "")
    return not name.startswith("_")


def get_open_link(node: Object, readme_dir: Path) -> str | None:
    """Compute get open link.

    Carry out the get open link operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    readme_dir : Path
        Description for ``readme_dir``.

    Returns
    -------
    str | None
        Description of return value.
    """
    
    
    
    
    





    rel_path = getattr(node, "relative_package_filepath", None)
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        relative = abs_path.relative_to(readme_dir).as_posix()
    except ValueError:
        return None
    start = int(getattr(node, "lineno", 1) or 1)
    return f"./{relative}:{start}:1"


def get_view_link(node: Object, readme_dir: Path) -> str | None:
    """Compute get view link.

    Carry out the get view link operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    readme_dir : Path
        Description for ``readme_dir``.

    Returns
    -------
    str | None
        Description of return value.
    """
    
    
    
    
    





    rel_path = getattr(node, "relative_package_filepath", None)
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        rel = abs_path.relative_to(ROOT)
    except ValueError:
        return None
    start = int(getattr(node, "lineno", 1) or 1)
    end = getattr(node, "endlineno", None)
    return gh_url(str(rel).replace("\\", "/"), start, end)


def iter_public_members(node: Object) -> Iterable[Object]:
    """Compute iter public members.

    Carry out the iter public members operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.

    Returns
    -------
    Iterable[Object]
        Description of return value.
    """
    
    
    
    
    





    members = getattr(node, "members", {})
    return sorted([m for m in members.values() if is_public(m)], key=lambda child: child.name)


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
    """Represent Config.

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
    >>> Config(...)
    """

    packages: list[str]
    link_mode: str  # github | editor | both
    editor: str  # vscode | relative
    fail_on_metadata_miss: bool
    dry_run: bool
    verbose: bool


@dataclass(frozen=True)
class Badges:
    """Represent Badges.

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
    >>> Badges(...)
    """

    stability: str | None = None
    owner: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    tested_by: list[dict[str, Any]] = field(default_factory=list)


def parse_config() -> Config:
    """Parse config.

    Returns
    -------
    Config
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> parse_config(...)
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
    parser.add_argument(
        "--fail-on-metadata-miss",
        action="store_true",
        default=False,
        help="Exit with a non-zero status if any public symbol lacks owner/stability metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report files that would be written without modifying them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose diagnostics.",
    )

    args = parser.parse_args()
    pkgs = (
        [pkg.strip() for pkg in args.packages.split(",") if pkg.strip()]
        if args.packages
        else iter_packages()
    )
    return Config(
        packages=pkgs,
        link_mode=args.link_mode,
        editor=args.editor,
        fail_on_metadata_miss=args.fail_on_metadata_miss,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def _lookup_nav(qname: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return symbol metadata and module-level defaults for a qualified name."""
    modules = NAVMAP.get("modules", {}) if isinstance(NAVMAP, dict) else {}
    symbol_name = qname.split(".")[-1]
    for entry in modules.values():
        if not isinstance(entry, dict):
            continue
        meta_table = entry.get("meta", {})
        if not isinstance(meta_table, dict):
            continue
        symbol_meta = meta_table.get(qname) or meta_table.get(symbol_name) or {}
        if not isinstance(symbol_meta, dict):
            symbol_meta = {}
        if symbol_meta:
            section_id = None
            for section in entry.get("sections", []) or []:
                symbols = section.get("symbols") if isinstance(section, dict) else None
                if isinstance(symbols, list) and symbol_name in symbols:
                    section_id = section.get("id")
                    break
            if section_id and "section" not in symbol_meta:
                symbol_meta = dict(symbol_meta)
                symbol_meta["section"] = section_id

            module_defaults: dict[str, Any] = {}
            module_meta = entry.get("module_meta")
            if isinstance(module_meta, dict):
                module_defaults = module_meta
            else:
                for key in ("owner", "stability", "since", "deprecated_in"):
                    if key in entry:
                        module_defaults[key] = entry[key]

            return symbol_meta, module_defaults
    return {}, {}


def badges_for(qname: str) -> Badges:
    """Compute badges for.

    Carry out the badges for operation.

    Parameters
    ----------
    qname : str
        Description for ``qname``.

    Returns
    -------
    Badges
        Description of return value.
    """
    
    
    
    
    





    meta, defaults = _lookup_nav(qname)
    merged = {**defaults, **meta}
    tests: list[dict[str, Any]] = []
    if isinstance(TEST_MAP, dict):
        recorded = TEST_MAP.get(qname)
        if recorded is None:
            recorded = TEST_MAP.get(qname.split(".")[-1])
        if isinstance(recorded, list):
            tests = [t for t in recorded if isinstance(t, dict)][:3]
    return Badges(
        stability=merged.get("stability"),
        owner=merged.get("owner"),
        section=merged.get("section"),
        since=merged.get("since"),
        deprecated_in=merged.get("deprecated_in"),
        tested_by=tests,
    )


def _format_test_badge(entries: list[dict[str, Any]]) -> str | None:
    """Format test badge.

    Parameters
    ----------
    entries : list[dict[str, Any]]
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
    >>> _format_test_badge(...)
    """
    if not entries:
        return None
    formatted: list[str] = []
    for item in entries:
        file = item.get("file")
        lines = item.get("lines")
        if not file:
            continue
        if isinstance(lines, list) and lines:
            formatted.append(f"{file}:{lines[0]}")
        else:
            formatted.append(file)
    if not formatted:
        return None
    return "`tested-by:" + ", ".join(formatted) + "`"


def format_badges(qname: str) -> str:
    """Compute format badges.

    Carry out the format badges operation.

    Parameters
    ----------
    qname : str
        Description for ``qname``.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    





    badges = badges_for(qname)
    tags: list[str] = []
    if badges.stability:
        tags.append(f"`stability:{badges.stability}`")
    if badges.owner:
        tags.append(f"`owner:{badges.owner}`")
    if badges.section:
        tags.append(f"`section:{badges.section}`")
    if badges.since:
        tags.append(f"`since:{badges.since}`")
    if badges.deprecated_in:
        tags.append(f"`deprecated:{badges.deprecated_in}`")
    test_tag = _format_test_badge(badges.tested_by)
    if test_tag:
        tags.append(test_tag)
    return (" " + " ".join(tags)) if tags else ""


def editor_link(abs_path: Path, lineno: int, editor_mode: str) -> str | None:
    """Compute editor link.

    Carry out the editor link operation.

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
    """
    
    
    
    
    





    if editor_mode == "vscode":
        return f"vscode://file/{abs_path}:{lineno}:1"
    # For 'relative', fall back to caller logic.
    return None


def _is_exception(node: Object) -> bool:
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
    kind = getattr(getattr(node, "kind", None), "value", "")
    if kind != "class":
        return False
    name = getattr(node, "name", "")
    if name.endswith(("Error", "Exception")):
        return True
    for base in getattr(node, "bases", []) or []:
        base_name = getattr(base, "full", None) or getattr(base, "name", None)
        if isinstance(base_name, str) and base_name.endswith(("Error", "Exception")):
            return True
    return False


KINDS = {"module", "package", "class", "function"}


def bucket_for(node: Object) -> str:
    """Bucket for.

    Parameters
    ----------
    node : Object
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
    >>> bucket_for(...)
    """
    kind = getattr(getattr(node, "kind", None), "value", "")
    if kind in {"module", "package"}:
        return "Modules"
    if kind == "class":
        return "Exceptions" if _is_exception(node) else "Classes"
    if kind == "function":
        return "Functions"
    return "Other"


def render_line(node: Object, readme_dir: Path, cfg: Config) -> str | None:
    """Compute render line.

    Carry out the render line operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    readme_dir : Path
        Description for ``readme_dir``.
    cfg : Config
        Description for ``cfg``.

    Returns
    -------
    str | None
        Description of return value.
    """
    
    
    
    
    





    qname = getattr(node, "path", "")
    summary = summarize(node)

    open_link = get_open_link(node, readme_dir) if cfg.link_mode in {"editor", "both"} else None
    view_link = get_view_link(node, readme_dir) if cfg.link_mode in {"github", "both"} else None

    if cfg.link_mode in {"editor", "both"}:
        rel_path = getattr(node, "relative_package_filepath", None)
        if rel_path:
            base = SRC if SRC.exists() else ROOT
            abs_path = (base / rel_path).resolve()
            override = editor_link(abs_path, int(getattr(node, "lineno", 1) or 1), cfg.editor)
            if override:
                open_link = override

    if not (open_link or view_link):
        return None

    parts = [f"- **`{qname}`**"]
    if summary:
        parts.append(f"— {summary}")
    badges = format_badges(qname)
    if badges:
        parts.append(badges)

    links: list[str] = []
    if open_link:
        links.append(f"[open]({open_link})")
    if view_link:
        links.append(f"[view]({view_link})")
    tail = f" → {' | '.join(links)}" if links else ""
    return " ".join(parts).replace("  ", " ").strip() + tail + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Compute write if changed.

    Carry out the write if changed operation.

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
    """
    
    
    
    
    





    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    rendered = content.rstrip() + f"\n<!-- agent:readme v1 sha:{SHA} content:{digest} -->\n"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if previous == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def write_readme(node: Object, cfg: Config) -> bool:
    """Compute write readme.

    Carry out the write readme operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    cfg : Config
        Description for ``cfg``.

    Returns
    -------
    bool
        Description of return value.
    """
    
    
    
    
    





    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"

    buckets = {
        "Modules": [],
        "Classes": [],
        "Functions": [],
        "Exceptions": [],
        "Other": [],
    }

    children = [
        child
        for child in iter_public_members(node)
        if getattr(getattr(child, "kind", None), "value", "") in KINDS
    ]

    for child in sorted(children, key=lambda c: getattr(c, "path", "")):
        line = render_line(child, pkg_dir, cfg)
        if line:
            buckets[bucket_for(child)].append(line)

    lines: list[str] = [f"# `{node.path}`\n\n"]
    synopsis = summarize(node)
    if synopsis:
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
    return changed


def _collect_missing_metadata(node: Object, missing: set[str]) -> None:
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
        kind = getattr(getattr(child, "kind", None), "value", "")
        if kind in KINDS:
            qname = getattr(child, "path", "")
            badge = badges_for(qname)
            if not badge.stability or not badge.owner:
                missing.add(qname)
        if kind in {"module", "package"}:
            _collect_missing_metadata(child, missing)


def main() -> None:
    """Main.

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
    >>> main(...)
    """
    cfg = parse_config()
    if not cfg.packages:
        raise SystemExit("No packages detected; set DOCS_PKG or add source packages under src/.")

    if cfg.verbose:
        print(
            f"packages={cfg.packages} link_mode={cfg.link_mode} editor={cfg.editor} "
            f"fail_on_metadata_miss={cfg.fail_on_metadata_miss} dry_run={cfg.dry_run}"
        )

    loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])
    missing_meta: set[str] = set()
    changed_any = False
    start = time.time()

    for pkg in cfg.packages:
        root = loader.load(pkg)
        if cfg.fail_on_metadata_miss:
            _collect_missing_metadata(root, missing_meta)

        changed_any |= write_readme(root, cfg)

        for member in root.members.values():
            if getattr(member, "is_package", False):
                if cfg.fail_on_metadata_miss:
                    _collect_missing_metadata(member, missing_meta)
                changed_any |= write_readme(member, cfg)

    if cfg.fail_on_metadata_miss and missing_meta:
        print(
            "ERROR: Missing owner/stability for public symbols:\n  - "
            + "\n  - ".join(sorted(missing_meta))
        )
        raise SystemExit(2)

    if cfg.verbose:
        duration = time.time() - start
        print(f"completed in {duration:.2f}s; changed={changed_any}")


if __name__ == "__main__":
    main()
