from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest
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
        assert parameter_description.startswith("Configure the value")
        assert parameter_description.endswith(".")
        return_description = transformed.schema.returns[0].description
        assert return_description.startswith("Describe return value")
        assert return_description.endswith(".")
    finally:
        manager.finish()


def test_dataclass_field_docs_plugin(tmp_path: Path) -> None:
    config = BuilderConfig()
    module_path = tmp_path / "pkg" / "module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Example:\n"
            "    count: int = 1\n"
            "    name: str | None = None\n"
        ),
        encoding="utf-8",
    )
    manager = load_plugins(config, tmp_path, only=["dataclass_field_docs"])
    try:
        symbol = SymbolHarvest(
            qname="pkg.module.Example",
            module="pkg.module",
            kind="class",
            parameters=[],
            return_annotation=None,
            docstring=None,
            owned=True,
            filepath=module_path,
            lineno=1,
            end_lineno=6,
            col_offset=0,
            decorators=["dataclass"],
            is_async=False,
            is_generator=False,
        )
        schema = DocstringSchema(summary="Summarise example.")
        semantic = SemanticResult(symbol=symbol, schema=schema)
        transformed = manager.apply_transformers(symbol.filepath, [semantic])[0]
        parameters = {parameter.name: parameter for parameter in transformed.schema.parameters}
        assert set(parameters) == {"count", "name"}
        count = parameters["count"]
        assert count.optional is True
        assert count.default == "1"
        assert "Defaults to``1``" not in count.description  # formatting check
        assert "Defaults to ``1``" in count.description
        name_param = parameters["name"]
        assert name_param.optional is True
        assert name_param.default == "None"
    finally:
        manager.finish()


def test_llm_summary_rewriter_apply(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = BuilderConfig()
    config.llm_summary_mode = "apply"
    manager = load_plugins(config, tmp_path, only=["llm_summary_rewriter"])
    try:
        symbol = SymbolHarvest(
            qname="pkg.module.func",
            module="pkg.module",
            kind="function",
            parameters=[],
            return_annotation=None,
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
        schema = DocstringSchema(summary="Returns the current value")
        semantic = SemanticResult(symbol=symbol, schema=schema)
        with caplog.at_level(logging.INFO):
            transformed = manager.apply_transformers(symbol.filepath, [semantic])[0]
        assert transformed.schema.summary.startswith("Return")
        assert "summary rewritten" in caplog.text
    finally:
        manager.finish()


def test_llm_summary_rewriter_dry_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = BuilderConfig()
    config.llm_summary_mode = "dry-run"
    manager = load_plugins(config, tmp_path, only=["llm_summary_rewriter"])
    try:
        symbol = SymbolHarvest(
            qname="pkg.module.func",
            module="pkg.module",
            kind="function",
            parameters=[],
            return_annotation=None,
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
        schema = DocstringSchema(summary="Provides access to the cache")
        semantic = SemanticResult(symbol=symbol, schema=schema)
        with caplog.at_level(logging.INFO):
            transformed = manager.apply_transformers(symbol.filepath, [semantic])[0]
        assert transformed.schema.summary == schema.summary
        assert "summary dry-run" in caplog.text
    finally:
        manager.finish()
