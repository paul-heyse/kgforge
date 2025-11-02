"""Stub for the LLM summary rewrite plugin."""

from __future__ import annotations

from tools.docstring_builder.plugins.base import PluginContext, PluginStage, TransformerPlugin
from tools.docstring_builder.semantics import SemanticResult

class LLMSummaryRewritePlugin(TransformerPlugin):
    name: str
    stage: PluginStage

    def on_start(self, context: PluginContext, /) -> None: ...
    def on_finish(self, context: PluginContext, /) -> None: ...
    def apply(self, context: PluginContext, result: SemanticResult, /) -> SemanticResult: ...

__all__ = ["LLMSummaryRewritePlugin"]
