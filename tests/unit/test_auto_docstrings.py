"""Tests for ``tools.auto_docstrings`` helpers."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "auto_docstrings.py"

spec = importlib.util.spec_from_file_location("tools.auto_docstrings", MODULE_PATH)
assert spec and spec.loader  # pragma: no cover - module must load for tests
auto_docstrings = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auto_docstrings
spec.loader.exec_module(auto_docstrings)

_required_sections = auto_docstrings._required_sections
annotation_to_text = auto_docstrings.annotation_to_text
build_docstring = auto_docstrings.build_docstring
build_examples = auto_docstrings.build_examples
module_name_for = auto_docstrings.module_name_for
parameters_for = auto_docstrings.parameters_for
extended_summary = auto_docstrings.extended_summary
describe_parameters = auto_docstrings.describe_parameters
describe_return = auto_docstrings.describe_return


@pytest.fixture()
def repo_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    src = repo / "src"
    docs_scripts = repo / "docs" / "_scripts"
    src.mkdir(parents=True)
    docs_scripts.mkdir(parents=True)

    monkeypatch.setattr(auto_docstrings, "REPO_ROOT", repo)
    monkeypatch.setattr(auto_docstrings, "SRC_ROOT", src)
    return repo


def test_module_name_for_src_package(repo_layout: Path) -> None:
    file_path = repo_layout / "src" / "kgfoundry_common" / "config.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")

    module = module_name_for(file_path)

    assert module == "kgfoundry_common.config"


def test_module_name_for_src_dunder_init(repo_layout: Path) -> None:
    file_path = repo_layout / "src" / "kgfoundry_common" / "__init__.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")

    module = module_name_for(file_path)

    assert module == "kgfoundry_common"


def test_module_name_for_non_src_files(repo_layout: Path) -> None:
    file_path = repo_layout / "docs" / "_scripts" / "render.py"
    file_path.write_text("", encoding="utf-8")

    module = module_name_for(file_path)

    assert module == "docs._scripts.render"


def _get_function(code: str) -> ast.FunctionDef:
    module = ast.parse(code)
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


@pytest.mark.parametrize(
    ("name", "expected_fragment"),
    [
        ("__iter__", "Yield each item"),
        ("__eq__", "Determine whether"),
        ("__pydantic_core_schema__", "schema object"),
        ("model_dump", "Serialise the model instance"),
    ],
)
def test_extended_summary_overrides(name: str, expected_fragment: str) -> None:
    """Ensure special members receive tailored extended summaries."""
    result = extended_summary("function", name, "pkg.module")

    assert expected_fragment in result


def test_build_docstring_appends_examples_for_public_function() -> None:
    node = _get_function(
        """
def do_stuff(value: int) -> str:
    return str(value)
"""
    )

    lines = build_docstring("function", node, "pkg.module")

    params = parameters_for(node)
    expected_tail = ["", *build_examples("pkg.module", "do_stuff", params, True)]

    assert lines[-(len(expected_tail) + 1) : -1] == expected_tail
    assert lines[-1] == '"""'


def test_build_docstring_skips_examples_for_private_function() -> None:
    node = _get_function(
        """
def _hidden(value: int) -> None:
    return None
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    assert "Examples" not in docstring


def test_required_sections_satisfied_by_generated_docstring() -> None:
    node = _get_function(
        """
def process(item: str, limit: int | None = None) -> str:
    return item
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    params = parameters_for(node)
    parameter_docs = describe_parameters(params)
    returns = annotation_to_text(node.returns)
    return_description = describe_return(returns)
    required = _required_sections("function", parameter_docs, return_description, [])

    for section in required:
        assert section in docstring


def test_build_docstring_infers_parameter_description() -> None:
    node = _get_function(
        """
def summarise(retrieved_items: list[str]) -> list[str]:
    return retrieved_items
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    assert "retrieved_items : List[str]" in docstring
    assert "Items retrieved from upstream operations. Accepts a list of str." in docstring


def test_build_docstring_infers_return_description() -> None:
    node = _get_function(
        """
def check_ready(status: bool) -> bool:
    return status
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    assert "Returns" in docstring
    assert "Boolean flag indicating success." in docstring


def test_build_docstring_omits_uninformative_sections() -> None:
    node = _get_function(
        """
def passthrough(value: Any) -> Any:
    return value
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    assert "Parameters" not in docstring
    assert "Returns" not in docstring


@pytest.mark.parametrize(
    "source, module_name, expected_lines",
    [
        (
            """
async def async_example(value: int) -> str:
    return str(value)
""",
            "pkg.sample",
            [
                ">>> from pkg.sample import async_example",
                ">>> result = async_example(...)",
                ">>> result  # doctest: +ELLIPSIS",
                "...",
            ],
        ),
        (
            """
def sync_example() -> None:
    pass
""",
            "pkg.sample",
            [
                ">>> sync_example()  # doctest: +ELLIPSIS",
            ],
        ),
    ],
)
def test_build_docstring_appends_examples(
    source: str, module_name: str, expected_lines: list[str]
) -> None:
    """Ensure Examples block is appended for functions and async functions."""
    node = ast.parse(source).body[0]
    doc_lines = build_docstring("function", node, module_name)

    assert "Examples" in doc_lines
    examples_index = doc_lines.index("Examples")
    assert doc_lines[examples_index + 1] == "--------"

    closing_index = len(doc_lines) - 1
    emitted_examples = doc_lines[examples_index + 2 : closing_index]
    for line in expected_lines:
        assert line in emitted_examples
