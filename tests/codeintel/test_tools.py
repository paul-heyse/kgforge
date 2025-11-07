from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from codeintel.mcp_server import tools

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_dir(repo_fixture: Path) -> Path:
    """Create sample directory within repository fixture.

    Parameters
    ----------
    repo_fixture : Path
        Repository fixture path.

    Returns
    -------
    Path
        Path to the created sample directory.
    """
    src = repo_fixture / "pkg"
    src.mkdir(exist_ok=True)
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


@pytest.mark.usefixtures("sample_dir")
def test_run_ts_query_identifiers() -> None:
    query = "(identifier) @id"
    result = tools.run_ts_query("pkg/example.py", language="python", query=query)
    assert any(cap["text"] == "foo" for cap in result.captures)


@pytest.mark.usefixtures("sample_dir")
def test_list_python_symbols() -> None:
    symbols = tools.list_python_symbols("pkg")
    assert symbols, "Expected at least one symbol entry"
    names = {entry["name"] for file in symbols for entry in file["defs"] if entry["name"]}
    assert {"foo", "bar"}.issubset(names)


@pytest.mark.usefixtures("sample_dir")
def test_list_calls() -> None:
    calls = tools.list_calls("pkg", language="python")
    assert any(call["callee"] == "bar" for call in calls)


@pytest.mark.usefixtures("sample_dir")
def test_list_errors() -> None:
    errors = tools.list_errors("pkg/broken.py", language="python")
    assert errors, "Expected syntax errors to be reported"
