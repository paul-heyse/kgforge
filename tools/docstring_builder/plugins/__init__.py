"""Plugin architecture for the docstring builder pipeline."""

from __future__ import annotations

import abc
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import ClassVar

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

ENTRY_POINT_GROUP = "kgfoundry.docstrings.plugins"


class PluginConfigurationError(RuntimeError):
    """Raised when plugin discovery or filtering fails."""


@dataclass(slots=True, frozen=True)
class PluginContext:
    """Context supplied to plugins when they execute."""

    config: BuilderConfig
    repo_root: Path
    file_path: Path | None = None


class BuilderPlugin(abc.ABC):
    """Base class for docstring builder plugins."""

    name: ClassVar[str]
    stage: ClassVar[str]

    def on_start(self, context: PluginContext) -> None:
        """Execute the setup hook before any files are processed."""
        del context

    def on_finish(self, context: PluginContext) -> None:
        """Execute the teardown hook after all files are processed."""
        del context


class HarvesterPlugin(BuilderPlugin):
    """Plugins that adjust :class:`HarvestResult` instances."""

    stage: ClassVar[str] = "harvester"

    @abc.abstractmethod
    def run(self, context: PluginContext, result: HarvestResult) -> HarvestResult:
        """Return a possibly modified harvest result."""


class TransformerPlugin(BuilderPlugin):
    """Plugins that refine :class:`SemanticResult` entries."""

    stage: ClassVar[str] = "transformer"

    @abc.abstractmethod
    def run(self, context: PluginContext, result: SemanticResult) -> SemanticResult:
        """Return a possibly modified semantic result."""


class FormatterPlugin(BuilderPlugin):
    """Plugins that adjust final :class:`DocstringEdit` entries."""

    stage: ClassVar[str] = "formatter"

    @abc.abstractmethod
    def run(self, context: PluginContext, edit: DocstringEdit) -> DocstringEdit:
        """Return a possibly modified docstring edit."""


@dataclass(slots=True)
class PluginManager:
    """Container coordinating plugin execution for each pipeline stage."""

    config: BuilderConfig
    repo_root: Path
    harvesters: list[HarvesterPlugin] = field(default_factory=list)
    transformers: list[TransformerPlugin] = field(default_factory=list)
    formatters: list[FormatterPlugin] = field(default_factory=list)
    available: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def _context(self, file_path: Path | None = None) -> PluginContext:
        return PluginContext(self.config, self.repo_root, file_path)

    def start(self) -> None:
        """Notify plugins that processing is about to begin."""
        context = self._context()
        with self._lock:
            for plugin in self._iter_plugins():
                plugin.on_start(context)

    def finish(self) -> None:
        """Notify plugins that processing has completed."""
        context = self._context()
        with self._lock:
            for plugin in reversed(list(self._iter_plugins())):
                plugin.on_finish(context)

    def apply_harvest(self, file_path: Path, result: HarvestResult) -> HarvestResult:
        """Run harvester plugins sequentially."""
        context = self._context(file_path)
        updated = result
        with self._lock:
            for plugin in self.harvesters:
                updated = plugin.run(context, updated)
        return updated

    def apply_transformers(
        self, file_path: Path, semantics: Iterable[SemanticResult]
    ) -> list[SemanticResult]:
        """Run transformer plugins sequentially for each semantic entry."""
        context = self._context(file_path)
        processed: list[SemanticResult] = []
        for entry in semantics:
            updated = entry
            with self._lock:
                for plugin in self.transformers:
                    updated = plugin.run(context, updated)
            processed.append(updated)
        return processed

    def apply_formatters(
        self, file_path: Path, edits: Iterable[DocstringEdit]
    ) -> list[DocstringEdit]:
        """Run formatter plugins sequentially for each docstring edit."""
        context = self._context(file_path)
        processed: list[DocstringEdit] = []
        for entry in edits:
            updated = entry
            with self._lock:
                for plugin in self.formatters:
                    updated = plugin.run(context, updated)
            processed.append(updated)
        return processed

    def enabled_plugins(self) -> list[str]:
        """Return the names of active plugins in execution order."""
        return [plugin.name for plugin in self._iter_plugins()]

    def _iter_plugins(self) -> Iterable[BuilderPlugin]:
        yield from self.harvesters
        yield from self.transformers
        yield from self.formatters


def _load_entry_point(entry_point: metadata.EntryPoint) -> type[BuilderPlugin]:
    loaded = entry_point.load()
    if isinstance(loaded, type) and issubclass(loaded, BuilderPlugin):
        return loaded
    if callable(loaded):  # pragma: no cover - third-party flexibility
        candidate = loaded()
        if isinstance(candidate, BuilderPlugin):
            return type(candidate)
    msg = f"Entry point {entry_point.name!r} did not return a BuilderPlugin"
    raise PluginConfigurationError(msg)


def _discover_entry_points() -> list[type[BuilderPlugin]]:
    discovered: list[type[BuilderPlugin]] = []
    for entry_point in metadata.entry_points().select(group=ENTRY_POINT_GROUP):
        try:
            discovered.append(_load_entry_point(entry_point))
        except PluginConfigurationError:
            raise
        except Exception as exc:  # pragma: no cover - best effort guard
            raise PluginConfigurationError(str(exc)) from exc
    return discovered


def _instantiate(plugin_type: type[BuilderPlugin]) -> BuilderPlugin:
    instance = plugin_type()
    if not isinstance(instance, BuilderPlugin):  # pragma: no cover - safety net
        msg = f"Plugin {plugin_type!r} is not a BuilderPlugin"
        raise PluginConfigurationError(msg)
    return instance


def load_plugins(
    config: BuilderConfig,
    repo_root: Path,
    *,
    only: Sequence[str] | None = None,
    disable: Sequence[str] | None = None,
    builtin: Sequence[type[BuilderPlugin]] | None = None,
) -> PluginManager:
    """Discover, filter, and instantiate plugins for the current run."""
    from tools.docstring_builder.plugins.dataclass_fields import DataclassFieldDocPlugin
    from tools.docstring_builder.plugins.llm_summary import LLMSummaryRewritePlugin
    from tools.docstring_builder.plugins.normalize_numpy_params import (
        NormalizeNumpyParamsPlugin,
    )

    builtin_types: list[type[BuilderPlugin]] = [
        DataclassFieldDocPlugin,
        LLMSummaryRewritePlugin,
        NormalizeNumpyParamsPlugin,
    ]
    if builtin is not None:
        builtin_types = list(builtin)

    discovered = {plugin_type.name: plugin_type for plugin_type in builtin_types}
    for plugin_type in _discover_entry_points():
        discovered.setdefault(plugin_type.name, plugin_type)

    only_set = {name.strip() for name in only or [] if name.strip()}
    disable_set = {name.strip() for name in disable or [] if name.strip()}

    missing_only = sorted(name for name in only_set if name not in discovered)
    missing_disable = sorted(name for name in disable_set if name not in discovered)
    if missing_only or missing_disable:
        missing = ", ".join([*missing_only, *missing_disable])
        message = f"Unknown plugin(s): {missing}"
        raise PluginConfigurationError(message)

    harvesters: list[HarvesterPlugin] = []
    transformers: list[TransformerPlugin] = []
    formatters: list[FormatterPlugin] = []
    skipped: list[str] = []

    for name, plugin_type in sorted(discovered.items()):
        if only_set and name not in only_set:
            skipped.append(name)
            continue
        if name in disable_set:
            skipped.append(name)
            continue
        plugin = _instantiate(plugin_type)
        if isinstance(plugin, HarvesterPlugin):
            harvesters.append(plugin)
        elif isinstance(plugin, TransformerPlugin):
            transformers.append(plugin)
        elif isinstance(plugin, FormatterPlugin):
            formatters.append(plugin)
        else:  # pragma: no cover - defensive guard
            skipped.append(name)

    manager = PluginManager(
        config=config,
        repo_root=repo_root,
        harvesters=harvesters,
        transformers=transformers,
        formatters=formatters,
        available=sorted(discovered.keys()),
        disabled=sorted(disable_set),
        skipped=sorted(skipped),
    )
    manager.start()
    return manager


__all__ = [
    "BuilderPlugin",
    "FormatterPlugin",
    "HarvesterPlugin",
    "PluginConfigurationError",
    "PluginContext",
    "PluginManager",
    "TransformerPlugin",
    "load_plugins",
]
