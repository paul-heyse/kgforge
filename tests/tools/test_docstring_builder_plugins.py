from __future__ import annotations

import inspect
from pathlib import Path

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import ParameterHarvest, SymbolHarvest
from tools.docstring_builder.plugins import load_plugins
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc, ReturnDoc
from tools.docstring_builder.semantics import SemanticResult


def test_normalize_numpy_params_plugin(tmp_path: Path) -> None:
    config = BuilderConfig()
    repo_root = tmp_path
    manager = load_plugins(config, repo_root, only=["normalize_numpy_params"])
    try:
        parameter = ParameterHarvest(
            name="value",
            kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
            annotation="int",
            default=None,
        )
        symbol = SymbolHarvest(
            qname="pkg.module.func",
            module="pkg.module",
            kind="function",
            parameters=[parameter],
            return_annotation="int",
            docstring=None,
            owned=True,
            filepath=tmp_path / "pkg" / "module.py",
            lineno=1,
            end_lineno=2,
            col_offset=0,
            decorators=[],
            is_async=False,
            is_generator=False,
        )
        schema = DocstringSchema(
            summary="Summarise work.",
            parameters=[
                ParameterDoc(
                    name="value",
                    annotation="int",
                    description="todo describe value",
                    optional=False,
                    default=None,
                    display_name="value",
                    kind="positional_or_keyword",
                )
            ],
            returns=[ReturnDoc(annotation="int", description="todo", kind="returns")],
        )
        semantic = SemanticResult(symbol=symbol, schema=schema)
        transformed = manager.apply_transformers(symbol.filepath, [semantic])[0]
        parameter_description = transformed.schema.parameters[0].description
        assert parameter_description.startswith("Describe `value`")
        assert parameter_description.endswith(".")
        return_description = transformed.schema.returns[0].description
        assert return_description.startswith("Describe return value")
        assert return_description.endswith(".")
    finally:
        manager.finish()

