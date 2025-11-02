"""Typed plugin protocol definitions and compatibility adapters."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast, runtime_checkable

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
        stage_candidate: object = getattr(plugin, "stage", None)
        valid_stages: dict[str, PluginStage] = {
            "harvester": "harvester",
            "transformer": "transformer",
            "formatter": "formatter",
        }
        if not isinstance(stage_candidate, str) or stage_candidate not in valid_stages:
            message = f"Unsupported legacy plugin stage: {stage_candidate!r}"
            raise TypeError(message)
        self.stage = valid_stages[stage_candidate]
        self._plugin = plugin
        self._warned = False
        name_attr: object = getattr(plugin, "name", None)
        if isinstance(name_attr, str):  # noqa: SIM108
            resolved_name = name_attr
        else:
            resolved_name = plugin.__class__.__name__
        self.name = resolved_name

    @classmethod
    def create(cls, plugin: LegacyPluginProtocol, /) -> _AnyLegacyAdapter:
        """Return a typed adapter for the legacy ``plugin`` instance."""
        adapter = cls(plugin)
        if adapter.stage == "harvester":
            return cls.wrap_harvester(adapter)
        if adapter.stage == "transformer":
            return cls.wrap_transformer(adapter)
        return cls.wrap_formatter(adapter)

    @classmethod
    def wrap_harvester(cls, adapter: LegacyPluginAdapter, /) -> _LegacyHarvesterAdapter:
        """Wrap ``adapter`` as a harvester plugin."""
        if adapter.stage != "harvester":
            message = f"Expected harvester plugin, received stage {adapter.stage!r}"
            raise TypeError(message)
        return _LegacyHarvesterAdapter(adapter)

    @classmethod
    def wrap_transformer(cls, adapter: LegacyPluginAdapter, /) -> _LegacyTransformerAdapter:
        """Wrap ``adapter`` as a transformer plugin."""
        if adapter.stage != "transformer":
            message = f"Expected transformer plugin, received stage {adapter.stage!r}"
            raise TypeError(message)
        return _LegacyTransformerAdapter(adapter)

    @classmethod
    def wrap_formatter(cls, adapter: LegacyPluginAdapter, /) -> _LegacyFormatterAdapter:
        """Wrap ``adapter`` as a formatter plugin."""
        if adapter.stage != "formatter":
            message = f"Expected formatter plugin, received stage {adapter.stage!r}"
            raise TypeError(message)
        return _LegacyFormatterAdapter(adapter)

    def on_start(self, context: PluginContext) -> None:
        """Invoke the legacy ``on_start`` hook if present."""
        hook: object = getattr(self._plugin, "on_start", None)
        if callable(hook):
            start_hook = cast(Callable[[PluginContext], object], hook)
            start_hook(context)

    def on_finish(self, context: PluginContext) -> None:
        """Invoke the legacy ``on_finish`` hook if present."""
        hook: object = getattr(self._plugin, "on_finish", None)
        if callable(hook):
            finish_hook = cast(Callable[[PluginContext], object], hook)
            finish_hook(context)

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


class _LegacyHarvesterAdapter(HarvesterPlugin):
    """Adapter that narrows ``apply`` to ``HarvestResult`` payloads."""

    stage: PluginStage = "harvester"

    def __init__(self, adapter: LegacyPluginAdapter) -> None:
        self._adapter = adapter
        self.name = adapter.name

    def on_start(self, context: PluginContext) -> None:
        self._adapter.on_start(context)

    def on_finish(self, context: PluginContext) -> None:
        self._adapter.on_finish(context)

    def apply(self, context: PluginContext, payload: HarvestResult) -> HarvestResult:
        result = self._adapter.apply(context, payload)
        if not isinstance(result, HarvestResult):
            message = (
                f"Legacy harvester {self.name!r} returned {type(result)!r}; expected HarvestResult"
            )
            raise TypeError(message)
        return result


class _LegacyTransformerAdapter(TransformerPlugin):
    """Adapter that narrows ``apply`` to ``SemanticResult`` payloads."""

    stage: PluginStage = "transformer"

    def __init__(self, adapter: LegacyPluginAdapter) -> None:
        self._adapter = adapter
        self.name = adapter.name

    def on_start(self, context: PluginContext) -> None:
        self._adapter.on_start(context)

    def on_finish(self, context: PluginContext) -> None:
        self._adapter.on_finish(context)

    def apply(self, context: PluginContext, payload: SemanticResult) -> SemanticResult:
        result = self._adapter.apply(context, payload)
        if not isinstance(result, SemanticResult):
            message = f"Legacy transformer {self.name!r} returned {type(result)!r}; expected SemanticResult"
            raise TypeError(message)
        return result


class _LegacyFormatterAdapter(FormatterPlugin):
    """Adapter that narrows ``apply`` to ``DocstringEdit`` payloads."""

    stage: PluginStage = "formatter"

    def __init__(self, adapter: LegacyPluginAdapter) -> None:
        self._adapter = adapter
        self.name = adapter.name

    def on_start(self, context: PluginContext) -> None:
        self._adapter.on_start(context)

    def on_finish(self, context: PluginContext) -> None:
        self._adapter.on_finish(context)

    def apply(self, context: PluginContext, payload: DocstringEdit) -> DocstringEdit:
        result = self._adapter.apply(context, payload)
        if not isinstance(result, DocstringEdit):
            message = (
                f"Legacy formatter {self.name!r} returned {type(result)!r}; expected DocstringEdit"
            )
            raise TypeError(message)
        return result


type _AnyLegacyAdapter = (
    LegacyPluginAdapter
    | _LegacyHarvesterAdapter
    | _LegacyTransformerAdapter
    | _LegacyFormatterAdapter
)


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
