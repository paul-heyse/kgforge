"""
Generate package-level README.md files that:
- List top-level classes/functions
- Provide deep links to their start line
- Are compatible with local editor links or GitHub permalinks
After generation, run 'doctoc src/<pkg>' to update TOCs.
"""

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

try:
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

from detect_pkg import detect_packages, detect_primary

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LINK_MODE = os.environ.get("DOCS_LINK_MODE", "editor").lower()  # "editor" | "github"
EDITOR = os.environ.get("DOCS_EDITOR", "vscode")
ENV_PKGS = os.environ.get("DOCS_PKG")


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")


OWNER = os.environ.get("DOCS_GITHUB_ORG", "your-org")
REPO = os.environ.get("DOCS_GITHUB_REPO", "your-repo")
SHA = git_sha()


def gh_url(rel: str, start: int, end: int | None) -> str:
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{rng}"


def editor_url(abs_path: Path, start: int) -> str | None:
    if EDITOR == "vscode":
        return f"vscode://file/{abs_path}:{start}:1"
    if EDITOR == "pycharm":
        return f"pycharm://open?file={abs_path}&line={start}"
    return None


def iter_packages():
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def summarize(node) -> str:
    doc = getattr(node, "docstring", None)
    if doc and getattr(doc, "value", None):
        summary = doc.value.strip().splitlines()[0].strip()
        return summary.rstrip(".")
    return ""


def is_public(node) -> bool:
    name = getattr(node, "name", "")
    return not name.startswith("_")


def get_source_url(node) -> str | None:
    rel_path = getattr(node, "relative_package_filepath", None)
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    start = int(getattr(node, "lineno", 1) or 1)
    end = getattr(node, "endlineno", None)
    if LINK_MODE == "github":
        rel = abs_path.relative_to(ROOT)
        return gh_url(str(rel).replace("\\", "/"), start, end)
    return editor_url(abs_path, start)


def get_relative_url(node, readme_dir: Path) -> str | None:
    rel_path = getattr(node, "relative_package_filepath", None)
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        relative = abs_path.relative_to(readme_dir)
    except ValueError:
        return None
    start = int(getattr(node, "lineno", 1) or 1)
    end = getattr(node, "endlineno", None)
    anchor = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"{relative.as_posix()}{anchor}"


def iter_public_members(node) -> Iterable:
    members = getattr(node, "members", {})
    return sorted([m for m in members.values() if is_public(m)], key=lambda child: child.name)


def render_member(node, *, indent: int, lines: list[str], readme_dir: Path) -> None:
    url = get_source_url(node)
    rel_url = get_relative_url(node, readme_dir)
    if not url and not rel_url:
        return
    summary = summarize(node)
    bullet = " " * indent + "- "
    label = f"**`{node.path}`**"
    text = f"{bullet}{label}"
    if summary:
        text += f" — {summary}"
    links: list[str] = []
    if url:
        link_label = "source" if LINK_MODE == "github" else "open"
        links.append(f"[{link_label}]({url})")
    if rel_url:
        links.append(f"[view]({rel_url})")
    if links:
        text += " → " + " | ".join(links)
    text += "\n"
    lines.append(text)

    kind = getattr(node, "kind", None)
    if getattr(kind, "value", None) in {"module", "package"}:
        for child in iter_public_members(node):
            child_kind = getattr(child.kind, "value", "")
            if child_kind not in {"module", "package", "class", "function"}:
                continue
            render_member(child, indent=indent + 2, lines=lines, readme_dir=readme_dir)


def write_readme(node):
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"
    lines = [
        f"# `{node.path}`\n\n",
        "<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n",
        "<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n\n",
        "## API\n",
    ]

    for child in iter_public_members(node):
        child_kind = getattr(child.kind, "value", "")
        if child_kind not in {"module", "package", "class", "function"}:
            continue
        render_member(child, indent=0, lines=lines, readme_dir=pkg_dir)

    readme.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {readme}")


for pkg in iter_packages():
    module = loader.load(pkg)
    write_readme(module)
    for member in module.members.values():
        if member.is_package:
            write_readme(member)
