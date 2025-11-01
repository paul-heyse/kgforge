"""Typed plugin protocol definitions and compatibility adapters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar, runtime_checkable

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

PluginStage = Literal["harvester", "transformer", "formatter"]
type DocstringPayload = HarvestResult | SemanticResult | DocstringEdit

InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


@dataclass(slots=True, frozen=True)
class PluginContext:
    """Context supplied to plugins when they execute."""

    config: BuilderConfig
    repo_root: Path
    file_path: Path | None = None


@runtime_checkable
class DocstringBuilderPlugin(Protocol[InputT_contra, OutputT_co]):
    """Generic protocol implemented by all docstring builder plugins.

    Implementations receive a :class:`PluginContext` and stage-specific
    payload type before returning an output payload. Plugins SHOULD raise
    :class:`~tools.docstring_builder.models.PluginExecutionError` (or allow the
    manager to wrap unexpected exceptions into that error) so failures surface
    RFC 9457 Problem Details responses consistent with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """

    name: str
    stage: PluginStage

    def on_start(self, context: PluginContext) -> None:
        """Run plugin-specific setup before execution begins."""
        ...

    def on_finish(self, context: PluginContext) -> None:
        """Perform teardown after plugin execution completes."""
        ...

    def apply(self, context: PluginContext, payload: InputT_contra) -> OutputT_co:
        """Execute the plugin for ``payload`` and return the processed object."""
        ...


@runtime_checkable
class HarvesterPlugin(DocstringBuilderPlugin[HarvestResult, HarvestResult], Protocol):
    """Plugins operating on harvested module metadata."""

    stage: PluginStage

    def apply(self, context: PluginContext, payload: HarvestResult) -> HarvestResult:
        """Transform harvested metadata before semantic analysis."""
        ...


@runtime_checkable
class TransformerPlugin(
    DocstringBuilderPlugin[SemanticResult, SemanticResult],
    Protocol,
):
    """Plugins refining semantic analysis results."""

    stage: PluginStage

    def apply(self, context: PluginContext, payload: SemanticResult) -> SemanticResult:
        """Mutate semantic results before rendering."""
        ...


@runtime_checkable
class FormatterPlugin(DocstringBuilderPlugin[DocstringEdit, DocstringEdit], Protocol):
    """Plugins adjusting rendered docstring edits."""

    stage: PluginStage

    def apply(self, context: PluginContext, payload: DocstringEdit) -> DocstringEdit:
        """Amend rendered docstring edits prior to writing."""
        ...


@runtime_checkable
class LegacyPluginProtocol(Protocol):
    """Legacy plugin signature prior to the typed ``apply`` API."""

    stage: PluginStage
    name: str

    def run(self, context: PluginContext, payload: DocstringPayload) -> DocstringPayload:
        """Execute the legacy plugin implementation."""
        ...


class LegacyPluginAdapter(DocstringBuilderPlugin[DocstringPayload, DocstringPayload]):
    """Adapt legacy ``run`` plugins to the typed ``apply`` protocol."""

    name: str
    stage: PluginStage

    def __init__(self, plugin: LegacyPluginProtocol) -> None:
        stage = getattr(plugin, "stage", None)
        if stage not in {"harvester", "transformer", "formatter"}:
            message = f"Unsupported legacy plugin stage: {stage!r}"
            raise TypeError(message)
        self.stage = stage
        self._plugin = plugin
        self._warned = False
        self.name = getattr(plugin, "name", plugin.__class__.__name__)

    def on_start(self, context: PluginContext) -> None:
        """Invoke the legacy ``on_start`` hook if present."""
        hook = getattr(self._plugin, "on_start", None)
        if callable(hook):
            hook(context)

    def on_finish(self, context: PluginContext) -> None:
        """Invoke the legacy ``on_finish`` hook if present."""
        hook = getattr(self._plugin, "on_finish", None)
        if callable(hook):
            hook(context)

    def apply(self, context: PluginContext, payload: DocstringPayload) -> DocstringPayload:
        """Delegate to the legacy ``run`` implementation with a warning."""
        self._warn()
        return self._plugin.run(context, payload)

    def _warn(self) -> None:
        if self._warned:
            return
        warnings.warn(
            f"Plugin {self.name!r} uses the legacy run() API; update to apply().",
            DeprecationWarning,
            stacklevel=3,
        )
        self._warned = True


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
