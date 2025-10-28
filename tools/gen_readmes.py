"""Gen Readmes utilities."""

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
    """Compute detect repo.

    Carry out the detect repo operation.

    Returns
    -------
    Tuple[str, str]
        Description of return value.
    """
    
    
    
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
    """Compute git sha.

    Carry out the git sha operation.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
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
    
    
    
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{rng}"


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


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
        summary = doc.value.strip().splitlines()[0].strip()
        return summary.rstrip(".")
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
    col = 1
    return f"./{relative}:{start}:{col}"


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


def render_member(node: Object, *, indent: int, lines: list[str], readme_dir: Path) -> None:
    """Compute render member.

    Carry out the render member operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    indent : int
        Description for ``indent``.
    lines : List[str]
        Description for ``lines``.
    readme_dir : Path
        Description for ``readme_dir``.
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
    """Compute write readme.

    Carry out the write readme operation.

    Parameters
    ----------
    node : Object
        Description for ``node``.
    """
    
    
    
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
