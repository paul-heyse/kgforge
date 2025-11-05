"""Core Tree-sitter utilities for the code-intelligence indexer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, cast

import tree_sitter as ts
from tree_sitter import Language, Parser

from kgfoundry_common.errors import ConfigurationError

if TYPE_CHECKING:
    from tree_sitter import Tree

LANG_PACKAGES: Final[dict[str, str]] = {
    "py": "tree_sitter_python",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
    "toml": "tree_sitter_toml",
    "md": "tree_sitter_markdown",
}

LANGUAGE_ALIAS: Final[dict[str, str]] = {
    "python": "py",
    "json": "json",
    "yaml": "yaml",
    "toml": "toml",
    "markdown": "md",
}

_QUERY_ATTR = "Query"
_QUERY_CURSOR_ATTR = "QueryCursor"
_QUERY_CLS = cast("type[Any]", getattr(ts, _QUERY_ATTR))
_QUERY_CURSOR_CLS = cast("type[Any]", getattr(ts, _QUERY_CURSOR_ATTR))


@dataclass(frozen=True)
class Langs:
    """Container for the Tree-sitter languages used by the code-intel indexer."""

    py: Language
    json: Language
    yaml: Language
    toml: Language
    md: Language


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
    ConfigurationError
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
        raise ConfigurationError(message, cause=exc) from exc
    try:
        factory = module.language
    except AttributeError as exc:
        message = f"Tree-sitter package '{package}' does not expose a 'language()' factory."
        raise ConfigurationError(message, cause=exc) from exc
    return Language(factory(), name=package)


def load_langs() -> Langs:
    """Instantiate the Tree-sitter languages used by the code-intel indexer.

    Returns
    -------
    Langs
        Container holding the Python, JSON, YAML, TOML, and Markdown grammars.

    Raises
    ------
    ConfigurationError
        If any required Tree-sitter language wheel is missing or incompatible.
    """
    languages: dict[str, Language] = {}
    for attr, package in LANG_PACKAGES.items():
        try:
            languages[attr] = _load_language(package)
        except ConfigurationError as exc:
            message = f"Failed to load Tree-sitter language '{attr}' from package '{package}'."
            raise ConfigurationError(message, cause=exc) from exc
    return Langs(**languages)


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
    parser = Parser()
    cast("Any", parser).language = lang
    return parser.parse(data)


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
    """
    query = _QUERY_CLS(lang, query_src)
    cursor = _QUERY_CURSOR_CLS(query)
    captures_map = cursor.captures(tree.root_node)
    out: list[dict[str, Any]] = []
    for capture_name, nodes in captures_map.items():
        out.extend(
            {
                "capture": capture_name,
                "kind": node.type,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "start_point": {"row": node.start_point[0], "column": node.start_point[1]},
                "end_point": {"row": node.end_point[0], "column": node.end_point[1]},
                "text": data[node.start_byte : node.end_byte].decode("utf-8", "ignore"),
            }
            for node in nodes
        )
    return out
