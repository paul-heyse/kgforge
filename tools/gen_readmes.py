"""Generate per-package README files with metadata badges and deep links."""

from __future__ import annotations

"""Generate deterministic, metadata-rich package README files.

The README generator walks the object tree discovered by Griffe, enriches
each public symbol with metadata from the NavMap/TestMap JSON artifacts,
and writes per-package ``README.md`` files.  The generated documents follow a
strict structure so downstream automation (agents, tooling, DocToc) can reason
about the content reliably.

High level workflow:

* discover packages to document (CLI flags, environment variables, detection)
* load NavMap/TestMap metadata for badges and "tested-by" annotations
* render each package into a deterministic markdown structure
* optionally invoke DocToc to populate the table of contents markers

The module also exposes helpers that the unit tests exercise directly so we can
guarantee determinism, badge ordering, and link formatting behaviour.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
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

DEFAULT_SYNOPSIS = "Package synopsis not yet documented."


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
    """Return the first sentence of ``node``'s docstring.

    Griffe surfaces docstrings via ``Object.docstring.value``.  The README
    synopsis needs the first complete sentence – not merely the first line – so
    we trim surrounding whitespace, drop leading blank lines, and capture the
    first sentence boundary.  Punctuation is preserved to keep the natural
    language flow intact.
    """

    doc = getattr(node, "docstring", None)
    if not doc or not getattr(doc, "value", None):
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
    """Configuration for README generation."""

    packages: list[str]
    link_mode: str  # github | editor | both
    editor: str  # vscode | relative
    fail_on_metadata_miss: bool
    dry_run: bool
    verbose: bool
    run_doctoc: bool


@dataclass(frozen=True)
class Badges:
    """Container for metadata badges rendered next to each symbol."""

    stability: str | None = None
    owner: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    tested_by: list[dict[str, Any]] = field(default_factory=list)


def parse_config() -> Config:
    """Parse CLI arguments and environment variables into :class:`Config`."""
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
                    "meta": {
                        "package.module.symbol": {...}
                    },
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
    modules = NAVMAP.get("modules", {}) if isinstance(NAVMAP, dict) else {}
    if not isinstance(modules, dict):
        return {}, {}

    symbol = qname.split(".")[-1]
    best_defaults: dict[str, Any] = {}
    for module_id, module in modules.items():
        if not isinstance(module_id, str):
            continue
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

            if symbol_meta:
                return symbol_meta, defaults

            if defaults and (qname.startswith(module_id) or symbol in (module.get("symbols") or [])):
                return {}, defaults

            if defaults and qname.startswith(module_id):
                best_defaults = defaults
    if best_defaults:
        return {}, best_defaults
    return {}, {}


def badges_for(qname: str) -> Badges:
    """Return :class:`Badges` describing README metadata for ``qname``.

    The ``docs/_build/test_map.json`` structure maps fully-qualified symbol
    names to a list of test descriptors::

        {
            "package.module.symbol": [
                {"file": "tests/unit/test_symbol.py", "lines": [10, 12]},
                ...
            ]
        }

    Only the first three entries are rendered to keep the output compact.
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
    return "`tested-by: " + ", ".join(formatted) + "`"


def format_badges(qname: str, base_length: int = 0) -> str:
    """Format metadata badges for ``qname`` with optional wrapping.

    ``base_length`` represents the length of the rendered line prefix (symbol
    name + summary).  When the combined prefix and badge text would exceed 80
    characters we emit a newline and indent continuation lines with four spaces
    to keep badges readable.
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
    if not parts:
        return ""
    badge_line = " ".join(parts)
    if base_length and base_length + 1 + len(badge_line) > 80:
        wrapped: list[str] = []
        current: list[str] = []
        current_len = 0
        available = 80 - 4
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
        return "\n    " + "\n    ".join(wrapped)
    return " " + badge_line


def editor_link(abs_path: Path, lineno: int, editor_mode: str) -> str | None:
    """Generate an ``[open]`` link for the configured editor mode."""

    if editor_mode == "vscode":
        return f"vscode://file/{abs_path}:{lineno}:1"
    if editor_mode == "relative":
        try:
            rel = abs_path.relative_to(ROOT)
        except ValueError:
            rel = abs_path
        return f"./{rel.as_posix()}:{lineno}:1"
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
    """Render a markdown bullet for ``node``.

    Example
    -------
    >>> line = render_line(node, Path("src/pkg"), cfg)
    >>> line.startswith("- **`pkg.symbol`**")
    True
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
    """Write (or update) the README for ``node``.

    The function returns ``True`` when the markdown on disk changed which allows
    callers to decide whether to run expensive follow-up tooling such as DocToc.
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
    """CLI entry point for README generation."""
    cfg = parse_config()
    if not cfg.packages:
        raise SystemExit("No packages detected; set DOCS_PKG or add packages under src/.")

    if not NAVMAP_PATH.exists():
        print(f"Warning: NavMap not found at {NAVMAP_PATH}; badges will be empty")
    if not TESTMAP_PATH.exists():
        print(
            f"Warning: Test map not found at {TESTMAP_PATH}; tested-by badges will be empty"
        )

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
