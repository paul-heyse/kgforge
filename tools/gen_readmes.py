"""Generate per-package README files with metadata badges and deep links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from griffe import Object

try:
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover
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
    elif remote.startswith("https://"):
        path = remote.split("https://", 1)[1]

    if path:
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]

    return override_owner or "your-org", override_repo or "your-repo"


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


def gh_url(rel_path: str, start: int, end: int | None) -> str:
    """Compute gh url.

    Carry out the gh url operation.

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
    """
    
    
    
    
    
    
    fragment = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel_path}{fragment}"


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
    
    
    
    
    
    
    return not getattr(node, "name", "").startswith("_")


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


def get_view_link(node: Object) -> str | None:
    """Compute get view link.

    Carry out the get view link operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.

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
    public = [m for m in members.values() if is_public(m)]
    return sorted(public, key=lambda child: getattr(child, "path", child.name))


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
    tested_by: list[dict[str, Any]] = None


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
    parser.add_argument("--fail-on-metadata-miss", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
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
    )


def _lookup_nav(qname: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lookup nav.

    Parameters
    ----------
    qname : str
        Description.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _lookup_nav(...)
    """
    modules = NAVMAP.get("modules", {}) if isinstance(NAVMAP, dict) else {}
    symbol = qname.split(".")[-1]
    for module in modules.values():
        if not isinstance(module, dict):
            continue
        meta = module.get("meta")
        if not isinstance(meta, dict):
            continue
        symbol_meta = meta.get(qname) or meta.get(symbol) or {}
        if not isinstance(symbol_meta, dict):
            symbol_meta = {}
        if symbol_meta:
            section_id = None
            for section in module.get("sections", []) or []:
                if (
                    isinstance(section, dict)
                    and symbol in section.get("symbols", [])
                    and isinstance(section.get("id"), str)
                ):
                    section_id = section["id"]
                    break
            if section_id and "section" not in symbol_meta:
                symbol_meta = {**symbol_meta, "section": section_id}

            defaults: dict[str, Any] = {}
            module_meta = module.get("module_meta")
            if isinstance(module_meta, dict):
                defaults = module_meta
            else:
                for key in ("owner", "stability", "since", "deprecated_in"):
                    if key in module:
                        defaults[key] = module[key]

            return symbol_meta, defaults
    return {}, {}


def badges_for(qname: str) -> Badges:
    """Badges for.

    Parameters
    ----------
    qname : str
        Description.

    Returns
    -------
    Badges
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> badges_for(...)
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
    """Format test badge.

    Parameters
    ----------
    entries : list[dict[str, Any]] | None
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
    return "`tested-by:" + ", ".join(formatted) + "`"


def format_badges(qname: str) -> str:
    """Format badges.

    Parameters
    ----------
    qname : str
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
    >>> format_badges(...)
    """
    badge = badges_for(qname)
    parts: list[str] = []
    if badge.stability:
        parts.append(f"`stability:{badge.stability}`")
    if badge.owner:
        parts.append(f"`owner:{badge.owner}`")
    if badge.section:
        parts.append(f"`section:{badge.section}`")
    if badge.since:
        parts.append(f"`since:{badge.since}`")
    if badge.deprecated_in:
        parts.append(f"`deprecated:{badge.deprecated_in}`")
    test_badge = _format_test_badge(badge.tested_by)
    if test_badge:
        parts.append(test_badge)
    return (" " + " ".join(parts)) if parts else ""


def editor_link(abs_path: Path, lineno: int, editor_mode: str) -> str | None:
    """Editor link.

    Parameters
    ----------
    abs_path : Path
        Description.
    lineno : int
        Description.
    editor_mode : str
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
    >>> editor_link(...)
    """
    if editor_mode == "vscode":
        return f"vscode://file/{abs_path}:{lineno}:1"
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
    """Render line.

    Parameters
    ----------
    node : Object
        Description.
    readme_dir : Path
        Description.
    cfg : Config
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
    >>> render_line(...)
    """
    qname = getattr(node, "path", "")
    summary = summarize(node)

    open_link = get_open_link(node, readme_dir) if cfg.link_mode in {"editor", "both"} else None
    view_link = get_view_link(node) if cfg.link_mode in {"github", "both"} else None

    if cfg.link_mode in {"editor", "both"}:
        rel_path = getattr(node, "relative_package_filepath", None)
        if rel_path:
            base = SRC if SRC.exists() else ROOT
            abs_path = (base / rel_path).resolve()
            direct = editor_link(abs_path, int(getattr(node, "lineno", 1) or 1), cfg.editor)
            if direct:
                open_link = direct

    if not (open_link or view_link):
        return None

    parts = [f"- **`{qname}`**"]
    if summary:
        parts.append(f"— {summary}")
    badge_text = format_badges(qname)
    if badge_text:
        parts.append(badge_text)

    links: list[str] = []
    if open_link:
        links.append(f"[open]({open_link})")
    if view_link:
        links.append(f"[view]({view_link})")
    tail = f" → {' | '.join(links)}" if links else ""
    return " ".join(parts).strip() + tail + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Write if changed.

    Parameters
    ----------
    path : Path
        Description.
    content : str
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
    >>> write_if_changed(...)
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
    """Write readme.

    Parameters
    ----------
    node : Object
        Description.
    cfg : Config
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
    >>> write_readme(...)
    """
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"

    buckets = {name: [] for name in ("Modules", "Classes", "Functions", "Exceptions", "Other")}
    children = [
        child
        for child in iter_public_members(node)
        if getattr(getattr(child, "kind", None), "value", "") in KINDS
    ]

    for child in sorted(children, key=lambda child: getattr(child, "path", "")):
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
        raise SystemExit("No packages detected; set DOCS_PKG or add packages under src/.")

    loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])
    missing_meta: set[str] = set()
    changed_any = False
    start = time.time()

    for pkg in cfg.packages:
        module = loader.load(pkg)
        if cfg.fail_on_metadata_miss:
            _collect_missing_metadata(module, missing_meta)
        changed_any |= write_readme(module, cfg)

        for member in module.members.values():
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
