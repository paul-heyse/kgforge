# -*- coding: utf-8 -*-
"""
Agent-first Sphinx configuration
- Static API docs via AutoAPI (no imports)
- Markdown via MyST
- Per-symbol "open in editor" links (linkcode + Griffe)
- Viewable source with line numbers (viewcode)
- JSON builder for machine-readable output
This file is robust to different repo shapes (src/PKG or PKG at root).
"""
import os, sys, subprocess
from pathlib import Path

# --- Project metadata (override via env if you like)
project = os.environ.get("PROJECT_NAME", "KGForge")
author  = os.environ.get("PROJECT_AUTHOR", "KGForge Maintainers")

# --- Paths
DOCS_DIR = Path(__file__).resolve().parent
ROOT     = DOCS_DIR.parent
SRC_DIR  = ROOT / "src"
TOOLS_DIR = ROOT / "tools"

def _detect_pkg():
    # pick the first package dir that has an __init__.py, preferring src/
    candidates = []
    if SRC_DIR.exists():
        candidates += [p.parent.name for p in SRC_DIR.glob("*/__init__.py")]
    candidates += [p.parent.name for p in ROOT.glob("*/__init__.py") if p.parent.name != "docs"]
    if not candidates:
        raise RuntimeError("No Python package found under src/ or project root")
    # Prefer lowercase names if any; else first found
    lowers = [c for c in candidates if c.islower()]
    return (lowers[0] if lowers else candidates[0])

PKG = os.environ.get("DOCS_PKG", _detect_pkg())

# Sphinx searches sys.path when rendering cross-refs; we do not import the package for API parsing.
sys.path.insert(0, str(ROOT))
if TOOLS_DIR.exists() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from detect_pkg import detect_packages, detect_primary
    _PACKAGES = []
    if os.environ.get("DOCS_PKG"):
        _PACKAGES = [pkg.strip() for pkg in os.environ["DOCS_PKG"].split(",") if pkg.strip()]
    if not _PACKAGES:
        _PACKAGES = detect_packages()
    if not _PACKAGES:
        _PACKAGES = [detect_primary()]
except Exception:
    _PACKAGES = [PKG]

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "autoapi.extension",            # static API docs (no import)
    "sphinxcontrib.mermaid",
]

# Use whatever theme you prefer; pydata_sphinx_theme is widely used.
html_theme = os.environ.get("SPHINX_THEME", "pydata_sphinx_theme")

# MyST (Markdown) features
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# AutoAPI: scan source directory (or root if no src/ folder)
autoapi_type = "python"
autoapi_dirs = [str(SRC_DIR if SRC_DIR.exists() else ROOT)]
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_options = [
    "members", "undoc-members", "show-inheritance", "special-members", "imported-members"
]
autoapi_ignore = ["*/__init__.py"]

# Show type hints nicely
autodoc_typehints = "description"

# Cross-link to Python stdlib docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Show line numbers in rendered source pages (Sphinx >= 7.2)
viewcode_line_numbers = True

suppress_warnings = ["myst.header", "misc.restructuredtext"]

# Ensure JSON builder can serialize lru_cache wrappers
from sphinxcontrib.serializinghtml import jsonimpl  # type: ignore

_json_default = jsonimpl.json.JSONEncoder.default


def _json_safe_default(self, obj):  # pragma: no cover - builder patch
    if obj.__class__.__name__ == "_lru_cache_wrapper":
        return repr(obj)
    return _json_default(self, obj)


jsonimpl.json.JSONEncoder.default = _json_safe_default

# --- Build deep links per symbol without importing your code (use Griffe)
from griffe import GriffeLoader

_loader = GriffeLoader(search_paths=[str(SRC_DIR if SRC_DIR.exists() else ROOT)])
_MODULE_CACHE = {}
PKG = _PACKAGES[0]

def _get_root(module: str | None):
    top = (module.split(".", 1)[0] if module else PKG)
    if top not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[top] = _loader.load(top)
        except Exception:
            _MODULE_CACHE[top] = None
    return _MODULE_CACHE[top]

def _lookup(module: str, fullname: str):
    """Return (abs_path, start, end) of a symbol via Griffe; None if not found."""
    root = _get_root(module)
    if root is None:
        return None
    node = root
    if module:
        parts = module.split(".")
        # skip the top-level package name (already in root)
        for part in parts[1:]:
            node = node.members.get(part)
            if node is None:
                return None
    for part in (fullname or "").split("."):
        if not part:
            continue
        node = node.members.get(part)
        if node is None:
            return None
    file_rel = getattr(node, "relative_package_filepath", None)
    start = getattr(node, "lineno", None)
    end = getattr(node, "endlineno", None)
    if not (file_rel and start):
        return None
    base = SRC_DIR if SRC_DIR.exists() else ROOT
    abs_path = base / file_rel
    return str(abs_path.resolve()), int(start), int(end or start)

# Link modes:
#   DOCS_LINK_MODE=editor (default): vscode://file/<abs>:line:col
#   DOCS_LINK_MODE=github: https://github.com/<org>/<repo>/blob/<sha>/<rel>#Lstart-Lend
LINK_MODE = os.environ.get("DOCS_LINK_MODE", "editor").lower()
EDITOR = os.environ.get("DOCS_EDITOR", "vscode")

def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")

def _github_url(relpath, start, end):
    org  = os.environ.get("DOCS_GITHUB_ORG", "your-org")
    repo = os.environ.get("DOCS_GITHUB_REPO", "your-repo")
    sha  = _git_sha()
    rng  = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{org}/{repo}/blob/{sha}/{relpath}{rng}"

def linkcode_resolve(domain, info):
    if domain != "py" or not info.get("module"):
        return None
    res = _lookup(info["module"], info.get("fullname", ""))
    if not res:
        return None
    abs_path, start, end = res
    if LINK_MODE == "github":
        # compute relative path from repo root
        rel = os.path.relpath(abs_path, ROOT)
        return _github_url(rel, start, end)
    # default: open in editor
    if EDITOR == "vscode":
        return f"vscode://file/{abs_path}:{start}:1"
    elif EDITOR == "pycharm":
        return f"pycharm://open?file={abs_path}&line={start}"
    return None
