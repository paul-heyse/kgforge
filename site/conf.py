"""Agent-first Sphinx configuration.

- Static API docs via AutoAPI (no imports)
- Markdown via MyST
- Per-symbol "open in editor" links (linkcode + Griffe)
- Viewable source with line numbers (viewcode)
- JSON builder for machine-readable output

This file is robust to different repo shapes (``src/<pkg>`` or ``<pkg>`` at root).
"""

import collections
import inspect
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from astroid import builder as astroid_builder
from astroid import manager as astroid_manager
from astroid.nodes import Module as AstroidModule
from autoapi import _parser as autoapi_parser  # type: ignore[attr-defined]
from griffe import Module as GriffeModule
from sphinxcontrib.serializinghtml import jsonimpl  # type: ignore[attr-defined]

# --- Project metadata (override via env if you like)
project = os.environ.get("PROJECT_NAME", "kgfoundry")
author = os.environ.get("PROJECT_AUTHOR", "kgfoundry Maintainers")

# --- Paths
DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"


def _detect_pkg() -> str:
    """Return the primary package name, preferring ``src/`` packages."""
    candidates: list[str] = []
    if SRC_DIR.exists():
        candidates += [p.parent.name for p in SRC_DIR.glob("*/__init__.py")]
    candidates += [p.parent.name for p in ROOT.glob("*/__init__.py") if p.parent.name != "docs"]
    if not candidates:
        message = "No Python package found under src/ or project root"
        raise RuntimeError(message)
    # Prefer lowercase names if any; else first found
    lowers = [c for c in candidates if c.islower()]
    return lowers[0] if lowers else candidates[0]


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
    "autoapi.extension",  # static API docs (no import)
    "sphinxcontrib.mermaid",
    "numpydoc",
    "numpydoc_validation",
    "sphinx_gallery.gen_gallery",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# Treat missing references as hard failures so documentation stays healthy.
nitpicky = True

# Enforce strict NumPy validation across the codebase.
numpydoc_validation_checks = {
    "GL01",
    "SS01",
    "ES01",
    "RT01",
    "PR01",
}

# Use whatever theme you prefer; pydata_sphinx_theme is widely used.
html_theme = os.environ.get("SPHINX_THEME", "pydata_sphinx_theme")

# MyST (Markdown) features
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# AutoAPI: scan each detected package directory so module names match import paths
autoapi_type = "python"
_AUTOAPI_DIRS: list[str] = []
for _pkg in _PACKAGES:
    _candidate = SRC_DIR / _pkg.replace(".", "/")
    if _candidate.exists():
        _AUTOAPI_DIRS.append(str(_candidate))
if not _AUTOAPI_DIRS:
    _AUTOAPI_DIRS = [str(SRC_DIR if SRC_DIR.exists() else ROOT)]
autoapi_dirs = _AUTOAPI_DIRS
autoapi_root = "autoapi"
autoapi_output_dir = "autoapi"
autoapi_add_toctree_entry = True
autoapi_python_module_prefix = "src."
autoapi_generate_module_toc = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "special-members",
    "imported-members",
    "private-members",
]
# Include package __init__ modules so AutoAPI emits package-level toctrees.
autoapi_ignore: list[str] = ["tools/navmap/check_navmap.py"]


def _autoapi_parse_file(
    self: autoapi_parser.Parser, file_path: str, condition: Callable[[str], bool]
) -> AstroidModule:  # pragma: no cover - compatibility shim
    directory, filename = os.path.split(file_path)
    module_parts = []
    if filename not in {"__init__.py", "__init__.pyi"}:
        module_parts = [os.path.splitext(filename)[0]]
    module_parts = collections.deque(module_parts)
    while directory and condition(directory):
        directory, module_part = os.path.split(directory)
        if module_part:
            module_parts.appendleft(module_part)

    module_name = ".".join(module_parts)
    manager = astroid_manager.AstroidManager()
    node = astroid_builder.AstroidBuilder(manager).file_build(file_path, module_name)
    return self.parse(node)


if "manager" in inspect.signature(astroid_builder.AstroidBuilder.__init__).parameters:
    autoapi_parser.Parser._parse_file = _autoapi_parse_file

# Show type hints nicely
autodoc_typehints = "description"

# Cross-link to Python stdlib docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "duckdb": ("https://duckdb.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "typer": ("https://typer.tiangolo.com/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
}

# Show line numbers in rendered source pages (Sphinx >= 7.2)
viewcode_line_numbers = True

suppress_warnings = ["myst.header", "misc.restructuredtext"]

EXAMPLES_DIR = (DOCS_DIR.parent / "examples").resolve()

sphinx_gallery_conf = {
    "examples_dirs": str(EXAMPLES_DIR),
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "doc_module": tuple(_PACKAGES),
    "run_stale_examples": False,
    "plot_gallery": False,
    "backreferences_dir": "gen_modules/backrefs",
}

# Ensure JSON builder can serialize lru_cache wrappers

_json_default = jsonimpl.json.JSONEncoder.default


def _json_safe_default(self: jsonimpl.json.JSONEncoder, obj: object) -> object:  # pragma: no cover
    if obj.__class__.__name__ == "_lru_cache_wrapper":
        return repr(obj)
    return _json_default(self, obj)


jsonimpl.json.JSONEncoder.default = _json_safe_default

# --- Build deep links per symbol without importing your code (use Griffe)
try:  #  griffe >=0.45 exposes loader module; fallback for current layout.
    from griffe.loader import GriffeLoader
except ImportError:  # pragma: no cover - compatibility shim
    from griffe import GriffeLoader  # type: ignore[attr-defined]

_loader = GriffeLoader(search_paths=[str(SRC_DIR if SRC_DIR.exists() else ROOT)])
_MODULE_CACHE: dict[str, GriffeModule | None] = {}
PKG = _PACKAGES[0]

# Ensure sphinx-gallery backref directory exists to avoid missing-path errors
(DOCS_DIR / "gen_modules" / "backrefs").mkdir(parents=True, exist_ok=True)


def _get_root(module: str | None) -> GriffeModule | None:
    top = module.split(".", 1)[0] if module else PKG
    if top not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[top] = _loader.load(top)
        except Exception:
            _MODULE_CACHE[top] = None
    return _MODULE_CACHE[top]


def _lookup(module: str, fullname: str) -> tuple[str, int, int] | None:
    """Return ``(abs_path, start, end)`` for the requested symbol."""
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


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return os.environ.get("DOCS_GITHUB_SHA", "main")


def _github_url(relpath: str, start: int, end: int | None) -> str:
    org = os.environ.get("DOCS_GITHUB_ORG", "your-org")
    repo = os.environ.get("DOCS_GITHUB_REPO", "your-repo")
    sha = _git_sha()
    rng = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{org}/{repo}/blob/{sha}/{relpath}{rng}"


def linkcode_resolve(domain: str, info: Mapping[str, str | None]) -> str | None:
    """Resolve Sphinx linkcode directives to editor or GitHub URLs."""
    module_name = info.get("module")
    if domain != "py" or not module_name:
        return None
    fullname = info.get("fullname", "")
    res = _lookup(module_name, fullname)
    if not res:
        return None
    abs_path, start, end = res
    if LINK_MODE == "github":
        rel = os.path.relpath(abs_path, ROOT)
        return _github_url(rel, start, end)
    if EDITOR == "vscode":
        return f"vscode://file/{abs_path}:{start}:1"
    if EDITOR == "pycharm":
        return f"pycharm://open?file={abs_path}&line={start}"
    return None


nitpick_ignore = set()

nitpick_ignore.update(
    {
        ("py:class", "optional"),
        ("py:class", "Optional"),
    }
)

nitpick_ignore_regex = [
    ("py:obj", r"pydantic\\.BaseModel"),
    ("py:class", r"pydantic\\.BaseModel"),
    ("py:class", r"kgfoundry\\.kgfoundry_common\\..*"),
    ("py:class", r"kgfoundry\\.kgfoundry_common\\.models\\.Id"),
]
