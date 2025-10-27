"""Generate package-level README.md files for public APIs.

- List top-level classes/functions
- Provide deep links to their start line
- Are compatible with local editor links or GitHub permalinks
After generation, run 'doctoc src/<pkg>' to update TOCs.
"""

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from griffe import Object

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


def detect_repo() -> tuple[str, str]:
    """Infer the GitHub owner/repo from the current git remote."""
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        remote = ""
    owner = os.environ.get("DOCS_GITHUB_ORG")
    repo = os.environ.get("DOCS_GITHUB_REPO")
    if owner and repo:
        return owner, repo
    if remote.endswith(".git"):
        remote = remote[:-4]
    path = ""
    if remote.startswith("git@"):
        _, remainder = remote.split("@", 1)
        path = remainder.split(":", 1)[1]
    elif remote.startswith("https://"):
        _, path = remote.split("https://", 1)
        path = path.split("/", 1)[1]
    if path:
        parts = path.split("/")
        try:
            owner_guess, repo_guess = parts[:2]
        except ValueError:
            pass
        else:
            return owner_guess, repo_guess
    return owner or "your-org", repo or "your-repo"


def git_sha() -> str:
    """Return the commit SHA used for GitHub permalinks."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")


OWNER, REPO = detect_repo()
SHA = git_sha()


def gh_url(rel: str, start: int, end: int | None) -> str:
    """Return a GitHub permalink for the given repository path.

    Parameters
    ----------
    rel : str
        Repository-relative path to the file.
    start : int
        Starting line number (1-based).
    end : int | None
        Optional ending line number (inclusive).

    Returns
    -------
    str
        GitHub URL pointing at the selected lines.
    """
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{rng}"


def iter_packages() -> list[str]:
    """Return the packages to document (respecting DOCS_PKG overrides)."""
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def summarize(node: Object) -> str:
    """Return the first sentence of the node docstring, if available."""
    doc = getattr(node, "docstring", None)
    if doc and getattr(doc, "value", None):
        summary = doc.value.strip().splitlines()[0].strip()
        return summary.rstrip(".")
    return ""


def is_public(node: Object) -> bool:
    """Return True when the node name does not start with an underscore."""
    name = getattr(node, "name", "")
    return not name.startswith("_")


def get_open_link(node: Object, readme_dir: Path) -> str | None:
    """Return a relative colon-style link for opening locally."""
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
    col = 1
    return f"./{relative}:{start}:{col}"


def get_view_link(node: Object, readme_dir: Path) -> str | None:
    """Return a GitHub permalink pinned to the current commit."""
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
    """Yield public members of a module or class (skipping private names)."""
    members = getattr(node, "members", {})
    return sorted([m for m in members.values() if is_public(m)], key=lambda child: child.name)


def render_member(node: Object, *, indent: int, lines: list[str], readme_dir: Path) -> None:
    """Append a formatted bullet for the given node to the README contents.

    Parameters
    ----------
    node : Object
        Symbol to render.
    indent : int
        Current bullet indentation width.
    lines : list[str]
        Accumulated README lines.
    readme_dir : Path
        Directory that will contain the README (used for relative links).
    """
    open_link = get_open_link(node, readme_dir)
    view_link = get_view_link(node, readme_dir)
    if not open_link and not view_link:
        return
    summary = summarize(node)
    bullet = " " * indent + "- "
    label = f"**`{node.path}`**"
    text = f"{bullet}{label}"
    if summary:
        text += f" — {summary}"
    if open_link:
        text += f" → [open]({open_link})"
        if view_link:
            text += f" | [view]({view_link})"
    elif view_link:
        text += f" → [view]({view_link})"
    text += "\n"
    lines.append(text)

    kind = getattr(node, "kind", None)
    if getattr(kind, "value", None) in {"module", "package"}:
        for child in iter_public_members(node):
            child_kind = getattr(child.kind, "value", "")
            if child_kind not in {"module", "package", "class", "function"}:
                continue
            render_member(child, indent=indent + 2, lines=lines, readme_dir=readme_dir)


def write_readme(node: Object) -> None:
    """Generate or overwrite the README for the given package node."""
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
