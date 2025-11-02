"""Stubs for docstring builder plugin discovery and management."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, TypeVar

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.models import DocstringBuilderError
from tools.docstring_builder.plugins.base import (
    DocstringBuilderPlugin,
    DocstringPayload,
    FormatterPlugin,
    HarvesterPlugin,
    LegacyPluginAdapter,
    LegacyPluginProtocol,
    PluginContext,
    PluginStage,
    TransformerPlugin,
)
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")

PluginInstance = DocstringBuilderPlugin[object, object] | LegacyPluginProtocol
RegisteredPlugin = HarvesterPlugin | TransformerPlugin | FormatterPlugin | LegacyPluginAdapter
PluginFactory = Callable[[], PluginInstance]

class PluginConfigurationError(DocstringBuilderError): ...

@dataclass(slots=True)
class PluginManager:
    config: BuilderConfig
    repo_root: Path
    harvesters: list[HarvesterPlugin]
    transformers: list[TransformerPlugin]
    formatters: list[FormatterPlugin]
    available: list[str]
    disabled: list[str]
    skipped: list[str]

    def _context(self, file_path: Path | None = ...) -> PluginContext: ...
    def start(self) -> None: ...
    def finish(self) -> None: ...
    def apply_harvest(self, file_path: Path, result: HarvestResult, /) -> HarvestResult: ...
    def apply_transformers(
        self, file_path: Path, semantics: Iterable[SemanticResult], /
    ) -> list[SemanticResult]: ...
    def apply_formatters(
        self, file_path: Path, edits: Iterable[DocstringEdit], /
    ) -> list[DocstringEdit]: ...
    def enabled_plugins(self) -> list[str]: ...

def _run_harvest_pipeline(
    plugins: Iterable[HarvesterPlugin],
    context: PluginContext,
    payload: HarvestResult,
    /,
) -> HarvestResult: ...
def _run_transformer_pipeline(
    plugins: Iterable[TransformerPlugin],
    context: PluginContext,
    payload: SemanticResult,
    /,
) -> SemanticResult: ...
def _run_formatter_pipeline(
    plugins: Iterable[FormatterPlugin],
    context: PluginContext,
    payload: DocstringEdit,
    /,
) -> DocstringEdit: ...
def _invoke_apply(
    plugin: DocstringBuilderPlugin[PayloadT, ResultT],
    context: PluginContext,
    payload: PayloadT,
    /,
) -> ResultT: ...
def _instantiate_plugin(candidate: object, /) -> RegisteredPlugin: ...
def _materialize_candidate(candidate: object, /) -> object: ...
def _resolve_plugin_name(candidate: object, /) -> str: ...
def _resolve_plugin_name_strict(candidate: object, /) -> str: ...
def _ensure_plugin_instance(obj: object, /) -> RegisteredPlugin: ...
def _register_plugin(manager: PluginManager, plugin: RegisteredPlugin, /) -> None: ...
def _is_registered_plugin(candidate: object, /) -> TypeGuard[RegisteredPlugin]: ...
def _is_legacy_plugin(candidate: object, /) -> TypeGuard[LegacyPluginProtocol]: ...
def _is_valid_stage(value: object, /) -> TypeGuard[PluginStage]: ...
def _load_entry_points() -> list[object]: ...
def _builtin_candidates(builtin: Sequence[PluginFactory] | None, /) -> tuple[object, ...]: ...
def load_plugins(
    config: BuilderConfig,
    repo_root: Path,
    /,
    *,
    only: Sequence[str] | None = ...,
    disable: Sequence[str] | None = ...,
    builtin: Sequence[PluginFactory] | None = ...,
) -> PluginManager: ...

__all__ = [
    "DocstringPayload",
    "PluginConfigurationError",
    "PluginManager",
    "load_plugins",
]
