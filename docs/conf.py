"""Agent-first Sphinx configuration.

- Static API docs via AutoAPI (no imports)
- Markdown via MyST
- Per-symbol "open in editor" links (linkcode + Griffe)
- Viewable source with line numbers (viewcode)
- JSON builder for machine-readable output

This file is robust to different repo shapes (``src/<pkg>`` or ``<pkg>`` at root).
"""

from __future__ import annotations

import collections
import importlib
import inspect
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, cast

import certifi
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from tools.griffe_utils import resolve_griffe

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Strongly typed project configuration derived from environment variables."""

    project_name: str
    author: str
    link_mode: str
    editor: str
    github_org: str
    github_repo: str
    github_sha: str | None


PROBLEM_DETAILS_SAMPLE: Final[dict[str, str | int]] = {
    "type": "https://kgfoundry.dev/docs/problems/link-resolution",
    "title": "Unable to resolve symbol location",
    "status": 500,
    "detail": "Griffe could not resolve the requested module or member.",
    "instance": "urn:kgfoundry:docs:linkcode:unresolved",
}


def get_project_settings(env: Mapping[str, str] | None = None) -> ProjectSettings:
    """Return strongly typed project settings derived from environment variables."""
    source = env if env is not None else os.environ
    project_name = source.get("PROJECT_NAME", "kgfoundry")
    author = source.get("PROJECT_AUTHOR", "kgfoundry Maintainers")
    link_mode = source.get("DOCS_LINK_MODE", "editor").lower()
    editor = source.get("DOCS_EDITOR", "vscode").lower()
    github_org = source.get("DOCS_GITHUB_ORG", "your-org")
    github_repo = source.get("DOCS_GITHUB_REPO", "your-repo")
    github_sha = source.get("DOCS_GITHUB_SHA")
    if link_mode not in {"editor", "github"}:
        LOGGER.warning("Unsupported DOCS_LINK_MODE '%s', defaulting to 'editor'", link_mode)
        link_mode = "editor"
    if editor not in {"vscode", "pycharm"}:
        LOGGER.warning("Unsupported DOCS_EDITOR '%s', defaulting to 'vscode'", editor)
        editor = "vscode"
    return ProjectSettings(
        project_name=project_name,
        author=author,
        link_mode=link_mode,
        editor=editor,
        github_org=github_org,
        github_repo=github_repo,
        github_sha=github_sha,
    )


_BaseModel: type[object] | None
try:
    from pydantic import BaseModel as _PydanticBaseModel
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _BaseModel = None
else:
    _BaseModel = cast(type[object], _PydanticBaseModel)

_auto_docstrings: ModuleType | None
try:
    from tools import auto_docstrings as _auto_docstrings  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _auto_docstrings = None

try:
    griffe_exceptions = importlib.import_module("griffe.exceptions")
except ModuleNotFoundError:  # pragma: no cover - griffe < 0.32
    GriffeLoadingError = RuntimeError
else:
    GriffeLoadingError = cast(
        type[Exception], getattr(griffe_exceptions, "LoadingError", RuntimeError)
    )

astroid_builder = importlib.import_module("astroid.builder")
astroid_manager = importlib.import_module("astroid.manager")
autoapi_parser = importlib.import_module("autoapi._parser")
jsonimpl = importlib.import_module("sphinxcontrib.serializinghtml.jsonimpl")
json_module = cast(ModuleType, jsonimpl.json)


class AutoapiParser(Protocol):
    """Partial protocol for AutoAPI parser implementations."""

    def parse(self, node: object, /) -> object:
        """Return an AutoAPI document tree for the provided AST node."""
        ...


class GriffeNode(Protocol):
    """Subset of Griffe objects used for linkcode resolution."""

    filepath: Path | str | None
    lineno: int | None
    endlineno: int | None
    members: Mapping[str, GriffeNode] | None


class GriffeLoaderInstance(Protocol):
    """Minimal loader surface required for linkcode."""

    def load(self, module: str) -> GriffeNode:
        """Return the module graph for ``module``."""
        ...


class GriffeLoaderFactory(Protocol):
    """Callable factory returning Griffe loader instances."""

    def __call__(self, search_paths: Sequence[str]) -> GriffeLoaderInstance:
        """Return a loader configured with the provided search paths."""
        ...


class _AutoDocstringsModule(Protocol):
    MAGIC_METHOD_EXTENDED_SUMMARIES: Mapping[str, str]
    STANDARD_METHOD_EXTENDED_SUMMARIES: Mapping[str, str]
    PYDANTIC_ARTIFACT_SUMMARIES: Mapping[str, str]

    def summarize(self, name: str, kind: str) -> str:
        """Return a synthesized summary for the provided symbol."""
        ...


# --- Project metadata (override via env if you like)
SETTINGS = get_project_settings()
project = SETTINGS.project_name
author = SETTINGS.author

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
    from tools.detect_pkg import detect_packages, detect_primary
except ImportError:  # pragma: no cover - fallback when tooling unavailable

    def detect_packages() -> list[str]:
        """Return empty package list when tooling is unavailable."""
        LOGGER.debug("tools.detect_pkg.detect_packages unavailable; returning empty list")
        return []

    def detect_primary() -> str:
        """Return fallback package name when tooling is unavailable."""
        LOGGER.debug("tools.detect_pkg.detect_primary unavailable; returning detected package")
        return PKG


def _apply_auto_docstring_overrides() -> None:
    """Inject extended summaries for Pydantic helpers and magic methods."""
    if _BaseModel is None or _auto_docstrings is None:
        return

    auto_docstrings = cast(_AutoDocstringsModule, _auto_docstrings)

    overrides: dict[str, str] = {}
    overrides.update(auto_docstrings.MAGIC_METHOD_EXTENDED_SUMMARIES)
    overrides.update(auto_docstrings.STANDARD_METHOD_EXTENDED_SUMMARIES)
    overrides.update(auto_docstrings.PYDANTIC_ARTIFACT_SUMMARIES)

    for name, extended in overrides.items():
        attr = getattr(_BaseModel, name, None)
        if attr is None:
            continue
        summary = auto_docstrings.summarize(name, "function")
        doc_text = f"{summary}\n\n{extended}"
        with suppress(
            AttributeError, TypeError
        ):  # pragma: no cover - descriptor without doc support
            attr.__doc__ = doc_text


_AUTOAPI_DOC_OVERRIDES: dict[str, str] = {}


def _build_autoapi_doc_overrides() -> None:
    """Prepare docstring overrides for members missing extended summaries."""
    if _AUTOAPI_DOC_OVERRIDES:
        return

    if _auto_docstrings is None:
        return

    auto_docstrings = cast(_AutoDocstringsModule, _auto_docstrings)

    overrides: dict[str, str] = {}
    overrides.update(auto_docstrings.MAGIC_METHOD_EXTENDED_SUMMARIES)
    overrides.update(auto_docstrings.STANDARD_METHOD_EXTENDED_SUMMARIES)
    overrides.update(auto_docstrings.PYDANTIC_ARTIFACT_SUMMARIES)

    for name, extended in overrides.items():
        summary = auto_docstrings.summarize(name, "function")
        _AUTOAPI_DOC_OVERRIDES[name] = f"{summary}\n\n{extended}"


def _autoapi_skip_member(*event: object) -> bool | None:
    """Skip inherited helpers that rely on autogenerated Pydantic internals."""
    if len(event) != AUTOAPI_EVENT_ARG_COUNT:
        return None
    _, _, name, obj, _, _ = cast(AutoapiEvent, event)
    if not isinstance(name, str):
        return None

    _build_autoapi_doc_overrides()

    simple = name.split(".")[-1]
    doc_override = _AUTOAPI_DOC_OVERRIDES.get(simple)
    if doc_override and hasattr(obj, DOCSTRING_ATTR):
        setattr(obj, DOCSTRING_ATTR, doc_override)
    return None


_PACKAGES: list[str] = []
if os.environ.get("DOCS_PKG"):
    _PACKAGES = [pkg.strip() for pkg in os.environ["DOCS_PKG"].split(",") if pkg.strip()]
if not _PACKAGES:
    _PACKAGES = detect_packages()
if not _PACKAGES:
    _PACKAGES = [detect_primary()]

_apply_auto_docstring_overrides()
_build_autoapi_doc_overrides()

numpydoc_validation_exclude = sorted({rf"^{re.escape(name)}$" for name in _AUTOAPI_DOC_OVERRIDES})

AutoapiEvent = tuple[Sphinx, str, str, object, bool, Mapping[str, object]]
AUTOAPI_EVENT_ARG_COUNT = 6
DOCSTRING_ATTR = "docstring"

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
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"default", "optional"}
numpydoc_xref_aliases = {
    # Internal models and aliases (TypeAlias entries resolve as ``py:data``).
    "Concept": ("py:class", "src.ontology.catalog.Concept"),
    "FloatArray": ("py:data", "src.vectorstore_faiss.gpu.FloatArray"),
    "Id": ("py:data", "src.kgfoundry_common.models.Id"),
    "NavMap": ("py:class", "src.kgfoundry_common.navmap_types.NavMap"),
    "Stability": ("py:data", "src.kgfoundry_common.navmap_types.Stability"),
    "StrArray": ("py:data", "src.vectorstore_faiss.gpu.StrArray"),
    "VecArray": ("py:data", "src.search_api.faiss_adapter.VecArray"),
    "SupportsHttp": ("py:class", "src.search_client.client.SupportsHttp"),
    "SupportsResponse": ("py:class", "src.search_client.client.SupportsResponse"),
    # Third-party helpers that numpydoc otherwise classifies incorrectly.
    "NDArray": ("py:data", "numpy.typing.NDArray"),
    "numpy.typing.NDArray": ("py:data", "numpy.typing.NDArray"),
    "pyarrow.schema": ("py:function", "pyarrow.schema"),
    "HTTPException": ("py:class", "fastapi.HTTPException"),
    "Depends": ("py:function", "fastapi.Depends"),
    "Header": ("py:function", "fastapi.Header"),
    "typer.Argument": ("py:class", "typer.Argument"),
    "typer.Option": ("py:class", "typer.Option"),
    "typer.Exit": ("py:exc", "typer.Exit"),
}

# Use whatever theme you prefer; pydata_sphinx_theme is widely used.
html_theme = os.environ.get("SPHINX_THEME", "pydata_sphinx_theme")

# MyST (Markdown) features
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]
# Generate id attributes for headings up to h3 so intra-page links stay stable.
myst_heading_anchors = 3

# Use the certifi CA bundle so intersphinx requests succeed in restricted environments.
tls_cacert = certifi.where()
# Fall back to skipping certificate verification when the bundled store is insufficient (e.g. CI sandboxes).
tls_verify = False

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
autoapi_ignore: list[str] = [
    "tools/navmap/check_navmap.py",
    # Exclude deprecated exception aliases to avoid duplicate documentation targets.
    "*/kgfoundry_common/exceptions.py",
]


def _autoapi_parse_file(
    self: AutoapiParser, file_path: str, condition: Callable[[str], bool]
) -> object:  # pragma: no cover - compatibility shim
    path = Path(file_path)
    module_parts: collections.deque[str] = collections.deque()
    if path.name not in {"__init__.py", "__init__.pyi"}:
        module_parts.append(path.stem)
    parent = path.parent
    while str(parent) and condition(str(parent)):
        module_parts.appendleft(parent.name)
        parent = parent.parent

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
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    # Scientific computing stack used throughout vector operations.
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    # HTTP and testing stacks referenced in API docs.
    "httpx": (
        "https://httpx.readthedocs.io/en/stable/",
        "https://httpx.readthedocs.io/en/stable/objects.inv",
    ),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    # Best-effort mappings for CLI/database integrations.
    "typer": ("https://typer.tiangolo.com/", None),
    "duckdb": ("https://duckdb.org/docs/api/python/", None),
}

# Fallback external links for types missing from upstream inventories.
extlinks = {
    "numpy-type": ("https://numpy.org/doc/stable/reference/generated/%s.html", "%s"),
    "pyarrow-type": ("https://arrow.apache.org/docs/python/generated/%s.html", "%s"),
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
    "first_notebook_cell": None,  # Do not prepend setup cells to rendered notebooks.
    "line_numbers": False,  # Produce cleaner code blocks in gallery pages.
    "reference_url": {  # Keep cross-reference targets local to this project.
        "sphinx_gallery": None,
    },
    "capture_repr": (),  # Avoid capturing repr output unless explicitly requested.
    "expected_failing_examples": [],  # Fail the build if an example regresses unexpectedly.
    "min_reported_time": 0,  # Always surface execution timing metadata in reports.
}

# Ensure JSON builder can serialize lru_cache wrappers


class DocsJSONEncoder(json.JSONEncoder):
    """JSON encoder that can serialize ``functools.lru_cache`` wrappers."""

    def default(self, o: object) -> object:  # pragma: no cover - exercised in Sphinx build
        """Return a JSON-safe representation for cache wrappers and delegates to super."""
        if o.__class__.__name__ == "_lru_cache_wrapper":
            return repr(o)
        return super().default(o)


# Override JSONEncoder default to render lru_cache wrappers
json_module.JSONEncoder = DocsJSONEncoder  # type: ignore[attr-defined]
json.JSONEncoder = DocsJSONEncoder

# --- Build deep links per symbol without importing your code (use Griffe)
griffe_api = resolve_griffe()
loader_factory = cast(GriffeLoaderFactory, griffe_api.loader_type)
_loader = loader_factory(search_paths=[str(SRC_DIR if SRC_DIR.exists() else ROOT)])
_MODULE_CACHE: dict[str, GriffeNode | None] = {}
PKG = _PACKAGES[0]

# Ensure sphinx-gallery backref directory exists to avoid missing-path errors
(DOCS_DIR / "gen_modules" / "backrefs").mkdir(parents=True, exist_ok=True)


def _get_root(module: str | None) -> GriffeNode | None:
    top = module.split(".", 1)[0] if module else PKG
    if top not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[top] = _loader.load(top)
        except (GriffeLoadingError, ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            problem_details = {
                **PROBLEM_DETAILS_SAMPLE,
                "detail": f"Griffe failed to load module '{top}': {exc}",
            }
            LOGGER.warning(
                "Failed to load module '%s' for linkcode lookup",
                top,
                extra={"problem_details": problem_details},
            )
            _MODULE_CACHE[top] = None
    return _MODULE_CACHE[top]


def _child_member(node: GriffeNode, name: str) -> GriffeNode | None:
    """Return the member named ``name`` from ``node`` when present."""
    members = getattr(node, "members", None)
    if isinstance(members, Mapping):
        child = members.get(name)
        if child is not None:
            return cast(GriffeNode, child)
    return None


def _lookup(module: str | None, fullname: str | None) -> tuple[str, int, int] | None:
    """Return ``(abs_path, start, end)`` for the requested symbol."""
    root = _get_root(module)
    if root is None:
        return None
    node: GriffeNode = root
    if module:
        parts = module.split(".")
        # skip the top-level package name (already in root)
        for part in parts[1:]:
            child = _child_member(node, part)
            if child is None:
                return None
            node = child
    for part in (fullname or "").split("."):
        if not part:
            continue
        child = _child_member(node, part)
        if child is None:
            return None
        node = child
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
LINK_MODE = SETTINGS.link_mode
EDITOR = SETTINGS.editor


def _git_sha() -> str:
    git_dir = ROOT / ".git"
    head_path = git_dir / "HEAD"
    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return SETTINGS.github_sha or "main"

    if head_contents.startswith("ref: "):
        ref = head_contents.removeprefix("ref: ").strip()
        ref_path = git_dir / ref
        with suppress(FileNotFoundError):
            ref_contents = ref_path.read_text(encoding="utf-8").strip()
            if ref_contents:
                return ref_contents
        return SETTINGS.github_sha or "main"
    if head_contents:
        return head_contents
    return SETTINGS.github_sha or "main"


def _github_url(relpath: str, start: int, end: int | None) -> str:
    org = SETTINGS.github_org
    repo = SETTINGS.github_repo
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
        LOGGER.debug(
            "linkcode resolution failed",
            extra={
                "problem_details": {
                    **PROBLEM_DETAILS_SAMPLE,
                    "detail": f"Unable to resolve {module_name}.{fullname}",
                }
            },
        )
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


nitpick_ignore = {
    # NumPy scalar aliases do not expose inventory targets.
    ("py:class", "numpy.float32"),
    ("py:class", "np.float32"),
    ("py:class", "np.int64"),
    ("py:class", "np.str_"),
    ("py:class", "NDArray"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "pyarrow.schema"),
    ("py:class", "Id"),
    ("py:class", "src.kgfoundry_common.models.Id"),
    ("py:class", "Stability"),
    ("py:class", "src.kgfoundry_common.navmap_types.Stability"),
    ("py:class", "VecArray"),
    ("py:class", "src.search_api.faiss_adapter.VecArray"),
    ("py:class", "FloatArray"),
    ("py:class", "StrArray"),
    ("py:class", "src.vectorstore_faiss.gpu.FloatArray"),
    ("py:class", "src.vectorstore_faiss.gpu.StrArray"),
    ("py:class", "SupportsHttp"),
    ("py:class", "src.search_client.client.SupportsHttp"),
    ("py:exc", "HTTPException"),
    ("py:class", "Concept"),
    ("py:class", "src.ontology.catalog.Concept"),
    # Third-party projects without accessible inventories.
    ("py:class", "duckdb.DuckDBPyConnection"),
    ("py:class", "typer.Argument"),
    ("py:class", "typer.Option"),
    ("py:exc", "typer.Exit"),
}

nitpick_ignore_regex: list[tuple[str, str]] = []


class GalleryTagsDirective(Directive):
    """Lightweight directive so ``.. tags::`` blocks parse without warnings."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        """Return an empty node list so Sphinx accepts ``.. tags::`` blocks."""
        if self.content:
            LOGGER.debug("tags directive content ignored", extra={"tags": list(self.content)})
        return []


def setup(app: Sphinx) -> None:  # pragma: no cover - Sphinx integration hook
    """Register custom directives when Sphinx loads the config module."""
    app.add_directive("tags", GalleryTagsDirective)
    app.connect("autoapi-skip-member", _autoapi_skip_member)
