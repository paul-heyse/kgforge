from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import SymbolHarvest
from tools.docstring_builder.models import PluginExecutionError
from tools.docstring_builder.plugins import load_plugins
from tools.docstring_builder.plugins.dataclass_fields import DataclassFieldDocPlugin
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc
from tools.docstring_builder.semantics import SemanticResult


def _semantic_result(path: Path) -> SemanticResult:
    module_name = path.with_suffix("").name
    symbol = SymbolHarvest(
        qname=f"{module_name}.ModuleClass",
        module=module_name,
        kind="class",
        parameters=[],
        return_annotation=None,
        docstring=None,
        owned=True,
        filepath=path,
        lineno=1,
        end_lineno=None,
        col_offset=0,
        decorators=["dataclasses.dataclass"],
        is_async=False,
        is_generator=False,
    )
    schema = DocstringSchema(
        summary="Sample dataclass.",
        parameters=[
            ParameterDoc(
                name="name",
                annotation="str",
                description="",
                display_name="name",
                kind="positional_or_keyword",
            ),
            ParameterDoc(
                name="flag",
                annotation="bool",
                description="",
                display_name="flag",
                kind="keyword_only",
            ),
        ],
    )
    return SemanticResult(symbol=symbol, schema=schema)


def test_dataclass_plugin_produces_stable_descriptions(tmp_path: Path) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text(
        """
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ModuleClass:
    name: str
    size: int = 1
    flag: bool = field(default=False, metadata={"doc": "Toggle behaviour."})
""",
        encoding="utf-8",
    )

    config = BuilderConfig()
    manager = load_plugins(config, tmp_path, builtin=[DataclassFieldDocPlugin])
    result = _semantic_result(module_path)
    first = manager.apply_transformers(module_path, [result])[0]
    second = manager.apply_transformers(module_path, [result])[0]

    assert [parameter.name for parameter in first.schema.parameters] == [
        "name",
        "size",
        "flag",
    ]
    assert first.schema.parameters[2].description == "Toggle behaviour."
    assert first.schema.parameters == second.schema.parameters


class RecordingPlugin:
    name = "recording"
    stage = "transformer"

    def __init__(self) -> None:
        self.invocations = 0

    def on_start(self, context: object) -> None:  # pragma: no cover - no-op hook
        del context

    def on_finish(self, context: object) -> None:  # pragma: no cover - no-op hook
        del context

    def apply(self, context: object, result: SemanticResult) -> SemanticResult:
        del context
        self.invocations += 1
        return result


class UnusedPlugin(RecordingPlugin):
    name = "unused"


def test_plugin_only_and_disable_filters(tmp_path: Path) -> None:
    config = BuilderConfig()
    only_manager = load_plugins(
        config,
        tmp_path,
        only=["recording"],
        builtin=[RecordingPlugin, UnusedPlugin],
    )
    assert [plugin.name for plugin in only_manager.transformers] == ["recording"]
    assert "unused" in only_manager.skipped
    result = _semantic_result(tmp_path / "module.py")
    only_manager.apply_transformers(result.symbol.filepath, [result])
    plugin = cast(RecordingPlugin, only_manager.transformers[0])
    assert plugin.invocations == 1

    disable_manager = load_plugins(
        config,
        tmp_path,
        disable=["recording"],
        builtin=[RecordingPlugin],
    )
    assert disable_manager.transformers == []
    assert disable_manager.disabled == ["recording"]


class FailingPlugin:
    name = "failing"
    stage = "transformer"

    def on_start(self, context: object) -> None:  # pragma: no cover - no-op hook
        del context

    def on_finish(self, context: object) -> None:  # pragma: no cover - no-op hook
        del context

    def apply(self, context: object, result: SemanticResult) -> SemanticResult:
        del context, result
        message = "plugin failure"
        raise ValueError(message)


def test_plugin_failures_raise_plugin_execution_error(tmp_path: Path) -> None:
    config = BuilderConfig()
    manager = load_plugins(config, tmp_path, builtin=[FailingPlugin])
    result = _semantic_result(tmp_path / "module.py")
    with pytest.raises(PluginExecutionError) as excinfo:
        manager.apply_transformers(result.symbol.filepath, [result])
    assert isinstance(excinfo.value.__cause__, ValueError)
