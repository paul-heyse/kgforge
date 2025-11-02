"""Stubs for docstring builder plugin protocols and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar, runtime_checkable

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

PluginStage = Literal["harvester", "transformer", "formatter"]
DocstringPayload = HarvestResult | SemanticResult | DocstringEdit

InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)

@dataclass(slots=True)
class PluginContext:
    config: BuilderConfig
    repo_root: Path
    file_path: Path | None = ...

@runtime_checkable
class DocstringBuilderPlugin(Protocol[InputT_contra, OutputT_co]):
    name: str
    stage: PluginStage

    def on_start(self, context: PluginContext, /) -> None: ...
    def on_finish(self, context: PluginContext, /) -> None: ...
    def apply(self, context: PluginContext, payload: InputT_contra, /) -> OutputT_co: ...

@runtime_checkable
class HarvesterPlugin(DocstringBuilderPlugin[HarvestResult, HarvestResult], Protocol):
    stage: PluginStage

@runtime_checkable
class TransformerPlugin(DocstringBuilderPlugin[SemanticResult, SemanticResult], Protocol):
    stage: PluginStage

@runtime_checkable
class FormatterPlugin(DocstringBuilderPlugin[DocstringEdit, DocstringEdit], Protocol):
    stage: PluginStage

@runtime_checkable
class LegacyPluginProtocol(Protocol):
    name: str
    stage: PluginStage

    def run(self, context: PluginContext, payload: DocstringPayload, /) -> DocstringPayload: ...

class LegacyPluginAdapter(DocstringBuilderPlugin[DocstringPayload, DocstringPayload]):
    name: str
    stage: PluginStage

    def __init__(self, plugin: LegacyPluginProtocol, /) -> None: ...
    @classmethod
    def create(
        cls,
        plugin: LegacyPluginProtocol,
        /,
    ) -> HarvesterPlugin | TransformerPlugin | FormatterPlugin | LegacyPluginAdapter: ...
    @classmethod
    def wrap_harvester(cls, adapter: LegacyPluginAdapter, /) -> HarvesterPlugin: ...
    @classmethod
    def wrap_transformer(cls, adapter: LegacyPluginAdapter, /) -> TransformerPlugin: ...
    @classmethod
    def wrap_formatter(cls, adapter: LegacyPluginAdapter, /) -> FormatterPlugin: ...

__all__ = [
    "DocstringBuilderPlugin",
    "DocstringPayload",
    "FormatterPlugin",
    "HarvesterPlugin",
    "InputT_contra",
    "LegacyPluginAdapter",
    "LegacyPluginProtocol",
    "OutputT_co",
    "PluginContext",
    "PluginStage",
    "TransformerPlugin",
]
