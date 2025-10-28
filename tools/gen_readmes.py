"""Provide utilities for module.

Auto-generated API documentation for the ``tools.gen_readmes`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.gen_readmes
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
    """Return detect repo.

    Auto-generated reference for the ``detect_repo`` callable defined in ``tools.gen_readmes``.
    
    Returns
    -------
    Tuple[str, str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.gen_readmes import detect_repo
    >>> result = detect_repo()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return git sha.

    Auto-generated reference for the ``git_sha`` callable defined in ``tools.gen_readmes``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.gen_readmes import git_sha
    >>> result = git_sha()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return gh url.

    Auto-generated reference for the ``gh_url`` callable defined in ``tools.gen_readmes``.
    
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
    
    Examples
    --------
    >>> from tools.gen_readmes import gh_url
    >>> result = gh_url(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{rng}"


def iter_packages() -> list[str]:
    """Return iter packages.

    Auto-generated reference for the ``iter_packages`` callable defined in ``tools.gen_readmes``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.gen_readmes import iter_packages
    >>> result = iter_packages()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    if ENV_PKGS:
        return [pkg.strip() for pkg in ENV_PKGS.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def summarize(node: Object) -> str:
    """Return summarize.

    Auto-generated reference for the ``summarize`` callable defined in ``tools.gen_readmes``.
    
    Parameters
    ----------
    node : Object
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
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    doc = getattr(node, "docstring", None)
    if doc and getattr(doc, "value", None):
        summary = doc.value.strip().splitlines()[0].strip()
        return summary.rstrip(".")
    return ""


def is_public(node: Object) -> bool:
    """Return is public.

    Auto-generated reference for the ``is_public`` callable defined in ``tools.gen_readmes``.
    
    Parameters
    ----------
    node : Object
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
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    name = getattr(node, "name", "")
    return not name.startswith("_")


def get_open_link(node: Object, readme_dir: Path) -> str | None:
    """Return get open link.

    Auto-generated reference for the ``get_open_link`` callable defined in ``tools.gen_readmes``.
    
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
    
    Examples
    --------
    >>> from tools.gen_readmes import get_open_link
    >>> result = get_open_link(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return get view link.

    Auto-generated reference for the ``get_view_link`` callable defined in ``tools.gen_readmes``.
    
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
    
    Examples
    --------
    >>> from tools.gen_readmes import get_view_link
    >>> result = get_view_link(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return iter public members.

    Auto-generated reference for the ``iter_public_members`` callable defined in ``tools.gen_readmes``.
    
    Parameters
    ----------
    node : Object
        Description for ``node``.
    
    Returns
    -------
    Iterable[Object]
        Description of return value.
    
    Examples
    --------
    >>> from tools.gen_readmes import iter_public_members
    >>> result = iter_public_members(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    members = getattr(node, "members", {})
    return sorted([m for m in members.values() if is_public(m)], key=lambda child: child.name)


def render_member(node: Object, *, indent: int, lines: list[str], readme_dir: Path) -> None:
    """Return render member.

    Auto-generated reference for the ``render_member`` callable defined in ``tools.gen_readmes``.
    
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
    
    Examples
    --------
    >>> from tools.gen_readmes import render_member
    >>> render_member(..., ..., ..., ...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return write readme.

    Auto-generated reference for the ``write_readme`` callable defined in ``tools.gen_readmes``.
    
    Parameters
    ----------
    node : Object
        Description for ``node``.
    
    Examples
    --------
    >>> from tools.gen_readmes import write_readme
    >>> write_readme(...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.gen_readmes
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
