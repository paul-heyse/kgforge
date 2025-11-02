"""Stub for the dataclass field documentation plugin."""

from __future__ import annotations

from pathlib import Path

from tools.docstring_builder.plugins.base import PluginContext, PluginStage, TransformerPlugin
from tools.docstring_builder.semantics import SemanticResult

class DataclassFieldDocPlugin(TransformerPlugin):
    name: str
    stage: PluginStage

    def __init__(self) -> None: ...

    def on_start(self, context: PluginContext, /) -> None: ...

    def on_finish(self, context: PluginContext, /) -> None: ...

    def apply(self, context: PluginContext, result: SemanticResult, /) -> SemanticResult: ...


def collect_dataclass_field_names(file_path: Path, module: str, /) -> dict[str, list[str]]: ...


__all__ = ["DataclassFieldDocPlugin", "collect_dataclass_field_names"]
