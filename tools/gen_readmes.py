"""
Generate package-level README.md files that:
- List top-level classes/functions
- Provide deep links to their start line
- Are compatible with local editor links or GitHub permalinks
After generation, run 'doctoc src/<pkg>' to update TOCs.
"""

import os
import subprocess
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


def gh_url(rel, start, end):
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel}{rng}"


def editor_url(abs_path, start):
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


def write_readme(node):
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"
    lines = [f"# `{node.path}`\n\n", "## API\n"]
    for child in node.members.values():
        if child.kind.value in {"class", "function"} and child.lineno:
            abs_path = ((SRC if SRC.exists() else ROOT) / child.relative_package_filepath).resolve()
            if LINK_MODE == "github":
                rel = abs_path.relative_to(ROOT)
                url = gh_url(str(rel), child.lineno, getattr(child, "endlineno", None))
            else:
                url = editor_url(abs_path, child.lineno)
            lines.append(f"- **`{child.path}`** → [source]({url})\n")
    readme.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {readme}")


for pkg in iter_packages():
    module = loader.load(pkg)
    write_readme(module)
    for member in module.members.values():
        if member.is_package:
            write_readme(member)
