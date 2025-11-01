"""Typed plugin protocol definitions and compatibility adapters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol, runtime_checkable

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

PluginStage = Literal["harvester", "transformer", "formatter"]
type DocstringPayload = HarvestResult | SemanticResult | DocstringEdit


@dataclass(slots=True, frozen=True)
class PluginContext:
    """Context supplied to plugins when they execute."""

    config: BuilderConfig
    repo_root: Path
    file_path: Path | None = None


@runtime_checkable
class DocstringBuilderPlugin(Protocol):
    """Protocol implemented by all docstring builder plugins."""

    name: ClassVar[str]
    stage: ClassVar[PluginStage]

    def on_start(self, context: PluginContext) -> None:
        """Run plugin-specific setup before execution begins."""
        ...

    def on_finish(self, context: PluginContext) -> None:
        """Perform teardown after plugin execution completes."""
        ...

    def apply(self, context: PluginContext, payload: DocstringPayload) -> DocstringPayload:
        """Execute the plugin for ``payload`` and return the processed object."""
        ...


@runtime_checkable
class HarvesterPlugin(DocstringBuilderPlugin, Protocol):
    """Plugins operating on harvested module metadata."""

    stage: ClassVar[Literal["harvester"]]

    def apply(  # type: ignore[override]
        self, context: PluginContext, payload: HarvestResult
    ) -> HarvestResult:  # pyrefly: ignore[bad-override]  # intentionally narrows type
        """Transform harvested metadata before semantic analysis."""
        ...


@runtime_checkable
class TransformerPlugin(DocstringBuilderPlugin, Protocol):
    """Plugins refining semantic analysis results."""

    stage: ClassVar[Literal["transformer"]]

    def apply(  # type: ignore[override]
        self, context: PluginContext, payload: SemanticResult
    ) -> SemanticResult:  # pyrefly: ignore[bad-override]  # intentionally narrows type
        """Mutate semantic results before rendering."""
        ...


@runtime_checkable
class FormatterPlugin(DocstringBuilderPlugin, Protocol):
    """Plugins adjusting rendered docstring edits."""

    stage: ClassVar[Literal["formatter"]]

    def apply(  # type: ignore[override]
        self, context: PluginContext, payload: DocstringEdit
    ) -> DocstringEdit:  # pyrefly: ignore[bad-override]  # intentionally narrows type
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


class LegacyPluginAdapter:
    """Adapt legacy ``run`` plugins to the typed ``apply`` protocol."""

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
    "FormatterPlugin",
    "HarvesterPlugin",
    "LegacyPluginAdapter",
    "LegacyPluginProtocol",
    "PluginContext",
    "PluginStage",
    "TransformerPlugin",
]
