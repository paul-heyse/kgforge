from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import harvest_file
from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.semantics import build_semantic_schemas


def _write_module(tmp_path: Path) -> Path:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    file_path = src_dir / "sample.py"
    file_path.write_text(
        textwrap.dedent(
            """
            def sample(value, /, scale: float, *args: int, *, limit: int = 0, **kwargs: str) -> int:
                return value
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return file_path


def test_parameter_harvest_preserves_signature_kinds(tmp_path: Path) -> None:
    repo_root = tmp_path
    file_path = _write_module(repo_root)
    config = BuilderConfig()

    result = harvest_file(file_path, config, repo_root)
    symbol = next(entry for entry in result.symbols if entry.qname.endswith("sample"))

    kinds = [parameter.kind for parameter in symbol.parameters]
    assert kinds == [
        inspect._ParameterKind.POSITIONAL_ONLY,
        inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
        inspect._ParameterKind.VAR_POSITIONAL,
        inspect._ParameterKind.KEYWORD_ONLY,
        inspect._ParameterKind.VAR_KEYWORD,
    ]

    display_names = [parameter.display_name() for parameter in symbol.parameters]
    assert display_names == ["value", "scale", "*args", "limit", "**kwargs"]

    semantics = build_semantic_schemas(result, config)
    semantic = next(entry for entry in semantics if entry.symbol.qname.endswith("sample"))
    param_docs = {doc.name: doc for doc in semantic.schema.parameters}
    assert param_docs["args"].display_name == "*args"
    assert param_docs["kwargs"].display_name == "**kwargs"
    assert param_docs["args"].kind == "var_positional"
    assert param_docs["kwargs"].kind == "var_keyword"
    docstring = render_docstring(semantic.schema, config.ownership_marker)

    assert "value :" in docstring
    assert "*args :" in docstring
    assert "**kwargs :" in docstring
