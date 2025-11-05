from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from codeintel.indexer import tscore
from tree_sitter import Language

if TYPE_CHECKING:
    from codeintel.indexer.tscore import Langs


@pytest.fixture(scope="module")
def langs() -> Langs:
    return tscore.load_langs()


def test_language_aliases_match_registered_languages(langs: Langs) -> None:
    for attr in tscore.LANGUAGE_ALIAS.values():
        language = getattr(langs, attr)
        assert isinstance(language, Language)


def test_parse_bytes_produces_tree(langs: Langs) -> None:
    tree = tscore.parse_bytes(langs.py, b"def foo():\n    return 1\n")
    assert tree.root_node.type == "module"
    assert [child.type for child in tree.root_node.children if child.is_named] == [
        "function_definition"
    ]


def test_run_query_captures_identifiers(langs: Langs) -> None:
    src = b"def foo(x):\n    return x\n"
    query = "(identifier) @id"
    tree = tscore.parse_bytes(langs.py, src)
    captures = tscore.run_query(langs.py, query, tree, src)
    assert any(hit["capture"] == "id" for hit in captures)
    assert all("text" in hit for hit in captures)
