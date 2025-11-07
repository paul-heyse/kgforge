"""Core Tree-sitter utilities for the code-intelligence indexer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import tree_sitter as ts
from tools import get_logger
from tree_sitter import Language, Parser

from codeintel.errors import LanguageNotSupportedError, ManifestError, QuerySyntaxError
from codeintel.observability import record_parse

if TYPE_CHECKING:
    from tree_sitter import Tree

_logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LANG_MANIFEST = ROOT / "build" / "languages.json"


@dataclass(frozen=True)
class LangSpec:
    """Language specification from the manifest."""

    name: str
    """Canonical language name (e.g., 'python')."""
    package: str
    """Importable package name (e.g., 'tree_sitter_python')."""
    module: str
    """Module name (usually same as package)."""
    version: str
    """Package version string."""


@cache
def _load_manifest() -> dict[str, LangSpec]:
    """Load and parse the language manifest.

    Returns
    -------
    dict[str, LangSpec]
        Mapping from canonical language name to language specification.

    Raises
    ------
    ManifestError
        If the manifest file is missing or malformed.
    """
    if not LANG_MANIFEST.exists():
        message = (
            f"Language manifest not found at {LANG_MANIFEST}. "
            "Run 'python -m codeintel.build_languages' to generate it."
        )
        _logger.error("Language manifest not found", extra={"manifest_path": str(LANG_MANIFEST)})
        raise ManifestError(message)
    try:
        data = json.loads(LANG_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"Failed to read or parse language manifest '{LANG_MANIFEST}': {exc}"
        _logger.exception("Failed to parse manifest", extra={"manifest_path": str(LANG_MANIFEST)})
        raise ManifestError(message) from exc
    languages: dict[str, LangSpec] = {}
    for name, item in data.get("languages", {}).items():
        languages[name] = LangSpec(
            name=name,
            package=item.get("package", ""),
            module=item.get("module", item.get("package", "")),
            version=item.get("version", "unknown"),
        )
    if not languages:
        message = f"Language manifest '{LANG_MANIFEST}' contains no languages."
        _logger.error("Empty manifest", extra={"manifest_path": str(LANG_MANIFEST)})
        raise ManifestError(message)
    _logger.debug("Loaded language manifest", extra={"languages": list(languages.keys())})
    return languages


# Internal attribute mapping (short names for Langs dataclass attributes)
_LANG_ATTR_MAP: Final[dict[str, str]] = {
    "python": "py",
    "json": "json",
    "yaml": "yaml",
    "toml": "toml",
    "markdown": "md",
}


def _get_language_attr(name: str) -> str:
    """Get the internal attribute name for a language.

    Parameters
    ----------
    name : str
        Canonical language name.

    Returns
    -------
    str
        Internal attribute name (e.g., 'py' for 'python').
    """
    return _LANG_ATTR_MAP.get(name, name)


@cache
def _get_language_names() -> tuple[str, ...]:
    """Get the list of supported language names from the manifest.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of canonical language names.
    """
    manifest = _load_manifest()
    return tuple(sorted(manifest.keys()))


LANGUAGE_NAMES: Final[tuple[str, ...]] = _get_language_names()
"""Canonical language names loaded from the manifest."""


LANGUAGE_ALIAS: Final[dict[str, str]] = {name: _get_language_attr(name) for name in LANGUAGE_NAMES}
"""Mapping from canonical names to internal attribute names."""

_QUERY_ATTR = "Query"
_QUERY_CURSOR_ATTR = "QueryCursor"
_QUERY_CLS = cast("type[Any]", getattr(ts, _QUERY_ATTR))
_QUERY_CURSOR_CLS = cast("type[Any]", getattr(ts, _QUERY_CURSOR_ATTR))


class Langs:
    """Container for the Tree-sitter languages used by the code-intel indexer.

    This class provides dynamic attribute access to Tree-sitter language instances
    loaded from the language manifest. Languages are accessed via short attribute
    names (e.g., ``langs.py`` for Python, ``langs.json`` for JSON) rather than
    canonical names.

    Attributes are dynamically determined from the language manifest, but common
    attributes include: ``py`` (python), ``json``, ``yaml``, ``toml``, ``md`` (markdown).

    Languages are stored internally and accessed via attribute lookup (e.g., ``langs.py``).
    This design allows the indexer to work with multiple language grammars without
    requiring explicit imports or hardcoded language names.

    Parameters
    ----------
    **languages : Language
        Language instances keyed by internal attribute name (e.g., 'py', 'json').
        These are typically loaded from the manifest via :func:`load_langs`.
    """

    def __init__(self, **languages: Language) -> None:
        self.__dict__.update(languages)

    def __getattr__(self, name: str) -> Language:
        """Raise :class:`AttributeError` with a helpful message when missing.

        Parameters
        ----------
        name : str
            Internal attribute name (e.g., 'py', 'json', 'yaml').

        Raises
        ------
        AttributeError
            Always raised to indicate the requested language alias is unavailable.
        """
        if name.startswith("_"):
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)
        available = [k for k in self.__dict__ if not k.startswith("_")]
        msg = f"Language attribute '{name}' not found. Available: {available}"
        raise AttributeError(msg)


@cache
def _load_language(package: str) -> Language:
    """Load a Tree-sitter language from its Python package.

    Parameters
    ----------
    package : str
        Importable package name that exposes a ``language`` factory returning a
        pointer to a ``TSLanguage``.

    Returns
    -------
    Language
        Instantiated Tree-sitter ``Language`` ready for parsing.

    Raises
    ------
    ManifestError
        If the package is missing or does not implement the expected ``language``
        callable.
    """
    try:
        module = import_module(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - configuration error
        message = (
            f"Tree-sitter package '{package}' is not installed. Run 'scripts/bootstrap.sh' "
            "to sync dependencies."
        )
        _logger.exception("Package not found", extra={"package_name": package})
        raise ManifestError(message) from exc
    try:
        factory = module.language
    except AttributeError as exc:
        message = f"Tree-sitter package '{package}' does not expose a 'language()' factory."
        _logger.exception("Invalid package", extra={"package_name": package})
        raise ManifestError(message) from exc
    return Language(factory(), name=package)


def load_langs() -> Langs:
    """Instantiate the Tree-sitter languages used by the code-intel indexer.

    Languages are loaded from the manifest file (`build/languages.json`), which serves
    as the single source of truth for language configuration.

    Returns
    -------
    Langs
        Container holding all Tree-sitter grammars specified in the manifest.

    Raises
    ------
    ManifestError
        If the manifest is missing, malformed, or any required Tree-sitter language
        wheel is missing or incompatible.
    """
    manifest = _load_manifest()
    languages: dict[str, Language] = {}
    for name, spec in manifest.items():
        attr = _get_language_attr(name)
        try:
            languages[attr] = _load_language(spec.package)
        except ManifestError as exc:
            message = (
                f"Failed to load Tree-sitter language '{name}' "
                f"(attr='{attr}', package='{spec.package}') from manifest."
            )
            _logger.exception(
                "Failed to load language",
                extra={"language_name": name, "language_module": spec.module},
            )
            raise ManifestError(message) from exc
    _logger.info(
        "Loaded languages", extra={"count": len(languages), "languages": list(manifest.keys())}
    )
    return Langs(**languages)


def get_language(langs: Langs, language: str) -> Language:
    """Return the Tree-sitter language associated with ``language``.

    Parameters
    ----------
    langs : Langs
        Bundle of pre-loaded Tree-sitter languages.
    language : str
        Canonical language identifier (for example ``"python"``).

    Returns
    -------
    Language
        The requested Tree-sitter language instance.

    Raises
    ------
    LanguageNotSupportedError
        If the language name is not recognised.
    """
    try:
        attr = LANGUAGE_ALIAS[language]
    except KeyError as exc:
        available = ", ".join(sorted(LANGUAGE_NAMES))
        message = f"Language '{language}' not supported. Available: {available}"
        _logger.warning(
            "Unsupported language", extra={"requested": language, "available": list(LANGUAGE_NAMES)}
        )
        raise LanguageNotSupportedError(
            message, extensions={"requested": language, "available": list(LANGUAGE_NAMES)}
        ) from exc
    return getattr(langs, attr)


def parse_bytes(lang: Language, data: bytes) -> Tree:
    """Parse a byte buffer with the supplied Tree-sitter language.

    Parameters
    ----------
    lang : Language
        Instantiated Tree-sitter grammar.
    data : bytes
        UTF-8 encoded source code to parse.

    Returns
    -------
    Tree
        Parsed syntax tree for the provided source buffer.
    """
    start = time.monotonic()
    parser = Parser()
    cast("Any", parser).language = lang
    tree = parser.parse(data)
    duration = time.monotonic() - start

    # Record parse metrics
    record_parse(lang.name, len(data), duration)

    _logger.debug(
        "Parsed file",
        extra={"language": lang.name, "size_bytes": len(data), "duration_s": round(duration, 4)},
    )
    return tree


def run_query(lang: Language, query_src: str, tree: Tree, data: bytes) -> list[dict[str, Any]]:
    """Execute a Tree-sitter query and emit structured captures.

    Parameters
    ----------
    lang : Language
        Tree-sitter grammar used to compile the query.
    query_src : str
        S-expression describing captures of interest.
    tree : Tree
        Parsed syntax tree produced by :func:`parse_bytes`.
    data : bytes
        Original source buffer associated with ``tree``.

    Returns
    -------
    list[dict[str, Any]]
        Capture metadata including byte ranges, points, and extracted text.

    Raises
    ------
    QuerySyntaxError
        If the query string has invalid syntax or references undefined node types.
    """
    try:
        query = _QUERY_CLS(lang, query_src)
    except Exception as exc:
        message = f"Failed to compile query: {exc}"
        _logger.exception("Query compilation failed", extra={"language": lang.name})
        raise QuerySyntaxError(message, extensions={"language": lang.name}) from exc

    cursor = _QUERY_CURSOR_CLS(query)
    out: list[dict[str, Any]] = []
    for match_id, captures in cursor.matches(tree.root_node):
        for capture_name, nodes in captures.items():
            out.extend(
                {
                    "capture": capture_name,
                    "match_id": match_id,
                    "kind": node.type,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "start_point": {
                        "row": node.start_point[0],
                        "column": node.start_point[1],
                    },
                    "end_point": {
                        "row": node.end_point[0],
                        "column": node.end_point[1],
                    },
                    "text": data[node.start_byte : node.end_byte].decode("utf-8", "ignore"),
                }
                for node in nodes
            )
    _logger.debug("Query executed", extra={"language": lang.name, "captures": len(out)})
    return out
