from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from codeintel.mcp_server import tools

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "example.py").write_text(
        """
        def foo(x: int) -> int:
            return x + 1

        def bar(y):
            return foo(y)

        result = bar(41)
        """,
        encoding="utf-8",
    )
    (src / "broken.py").write_text("def baz(:\n    return 1\n", encoding="utf-8")
    return src


def test_run_ts_query_identifiers(sample_dir: Path) -> None:
    file_path = sample_dir / "example.py"
    query = "(identifier) @id"
    result = tools.run_ts_query(str(file_path), language="python", query=query)
    assert any(cap["text"] == "foo" for cap in result.captures)


def test_list_python_symbols(sample_dir: Path) -> None:
    symbols = tools.list_python_symbols(str(sample_dir))
    assert symbols, "Expected at least one symbol entry"
    names = {entry["name"] for file in symbols for entry in file["defs"] if entry["name"]}
    assert {"foo", "bar"}.issubset(names)


def test_list_calls(sample_dir: Path) -> None:
    calls = tools.list_calls(str(sample_dir), language="python")
    assert any(call["callee"] == "bar" for call in calls)


def test_list_errors(sample_dir: Path) -> None:
    errors = tools.list_errors(str(sample_dir / "broken.py"), language="python")
    assert errors, "Expected syntax errors to be reported"
