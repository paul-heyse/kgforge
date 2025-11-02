"""Stub for the parameter normalisation plugin."""

from __future__ import annotations

from tools.docstring_builder.plugins.base import PluginContext, PluginStage, TransformerPlugin
from tools.docstring_builder.schema import ParameterDoc, ReturnDoc
from tools.docstring_builder.semantics import SemanticResult

class NormalizeNumpyParamsPlugin(TransformerPlugin):
    name: str
    stage: PluginStage

    def on_start(self, context: PluginContext, /) -> None: ...

    def on_finish(self, context: PluginContext, /) -> None: ...

    def apply(self, context: PluginContext, result: SemanticResult, /) -> SemanticResult: ...

    @staticmethod
    def _normalize_parameter(parameter: ParameterDoc, /) -> ParameterDoc: ...

    @staticmethod
    def _normalize_return(entry: ReturnDoc, /) -> ReturnDoc: ...


__all__ = ["NormalizeNumpyParamsPlugin"]
