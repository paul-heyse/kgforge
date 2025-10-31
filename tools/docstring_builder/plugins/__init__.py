"""Plugin discovery and execution helpers for the docstring builder."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import cast

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.models import (
    DocstringBuilderError,
    PluginExecutionError,
)
from tools.docstring_builder.plugins.base import (
    DocstringBuilderPlugin,
    FormatterPlugin,
    HarvesterPlugin,
    LegacyPluginAdapter,
    LegacyPluginProtocol,
    PluginContext,
    PluginStage,
    TransformerPlugin,
)
from tools.docstring_builder.plugins.dataclass_fields import DataclassFieldDocPlugin
from tools.docstring_builder.plugins.llm_summary import LLMSummaryRewritePlugin
from tools.docstring_builder.plugins.normalize_numpy_params import (
    NormalizeNumpyParamsPlugin,
)
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

ENTRY_POINT_GROUP = "kgfoundry.docstrings.plugins"


class PluginConfigurationError(DocstringBuilderError):
    """Raised when plugin discovery or filtering fails."""


@dataclass(slots=True)
class PluginManager:
    """Coordinate plugin execution across builder stages."""

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
        return PluginContext(config=self.config, repo_root=self.repo_root, file_path=file_path)

    def start(self) -> None:
        """Invoke ``on_start`` for all registered plugins."""
        context = self._context()
        with self._lock:
            for plugin in self._iter_plugins():
                plugin.on_start(context)

    def finish(self) -> None:
        """Invoke ``on_finish`` for registered plugins in reverse order."""
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
                updated = _invoke_apply(plugin, context, updated)
        return updated

    def apply_transformers(
        self, file_path: Path, semantics: Iterable[SemanticResult]
    ) -> list[SemanticResult]:
        """Run transformer plugins sequentially for each semantic result."""
        context = self._context(file_path)
        processed: list[SemanticResult] = []
        for entry in semantics:
            updated = entry
            with self._lock:
                for plugin in self.transformers:
                    updated = _invoke_apply(plugin, context, updated)
            processed.append(updated)
        return processed

    def apply_formatters(
        self, file_path: Path, edits: Iterable[DocstringEdit]
    ) -> list[DocstringEdit]:
        """Run formatter plugins sequentially for each edit."""
        context = self._context(file_path)
        processed: list[DocstringEdit] = []
        for entry in edits:
            updated = entry
            with self._lock:
                for plugin in self.formatters:
                    updated = _invoke_apply(plugin, context, updated)
            processed.append(updated)
        return processed

    def enabled_plugins(self) -> list[str]:
        """Return the names of active plugins in execution order."""
        return [plugin.name for plugin in self._iter_plugins()]

    def _iter_plugins(self) -> Iterable[DocstringBuilderPlugin]:
        yield from self.harvesters
        yield from self.transformers
        yield from self.formatters


def _invoke_apply(
    plugin: DocstringBuilderPlugin,
    context: PluginContext,
    payload: HarvestResult | SemanticResult | DocstringEdit,
) -> HarvestResult | SemanticResult | DocstringEdit:
    try:
        result = plugin.apply(
            context,
            cast(HarvestResult | SemanticResult | DocstringEdit, payload),
        )
        return cast(HarvestResult | SemanticResult | DocstringEdit, result)
    except DocstringBuilderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Plugin {plugin.name!r} failed during {plugin.stage} execution"
        raise PluginExecutionError(message) from exc


def _instantiate_plugin(candidate: object) -> DocstringBuilderPlugin:
    if isinstance(candidate, LegacyPluginAdapter):
        return candidate
    if isinstance(candidate, type):
        instance = candidate()
        return _ensure_plugin_instance(instance)
    if callable(candidate):
        instance = candidate()
        return _ensure_plugin_instance(instance)
    if isinstance(candidate, DocstringBuilderPlugin):  # pragma: no cover - runtime check
        return candidate
    return _ensure_plugin_instance(candidate)


def _ensure_plugin_instance(obj: object) -> DocstringBuilderPlugin:
    name = getattr(obj, "name", obj.__class__.__name__)
    stage = getattr(obj, "stage", None)
    if stage not in {"harvester", "transformer", "formatter"}:
        message = f"Plugin {name!r} declares invalid stage {stage!r}"
        raise PluginConfigurationError(message)
    if isinstance(obj, DocstringBuilderPlugin):  # pragma: no cover - runtime check
        return cast(DocstringBuilderPlugin, obj)
    if hasattr(obj, "apply"):
        return cast(DocstringBuilderPlugin, obj)
    if hasattr(obj, "run"):
        legacy = cast(LegacyPluginProtocol, obj)
        return LegacyPluginAdapter(legacy)
    message = f"Plugin {name!r} must define apply() or run()"
    raise PluginConfigurationError(message)


def _register_plugin(manager: PluginManager, plugin: DocstringBuilderPlugin) -> None:
    stage: PluginStage = plugin.stage
    if stage == "harvester":
        manager.harvesters.append(cast(HarvesterPlugin, plugin))
    elif stage == "transformer":
        manager.transformers.append(cast(TransformerPlugin, plugin))
    elif stage == "formatter":
        manager.formatters.append(cast(FormatterPlugin, plugin))
    else:  # pragma: no cover - stage already validated earlier
        message = f"Unsupported plugin stage: {stage!r}"
        raise PluginConfigurationError(message)


def _load_entry_points() -> list[object]:
    loaded: list[object] = []
    for entry_point in metadata.entry_points().select(group=ENTRY_POINT_GROUP):
        try:
            loaded.append(entry_point.load())
        except Exception as exc:  # pragma: no cover - best effort guard
            message = f"Failed to load plugin entry point {entry_point.name!r}"
            raise PluginConfigurationError(message) from exc
    return loaded


def load_plugins(
    config: BuilderConfig,
    repo_root: Path,
    *,
    only: Sequence[str] | None = None,
    disable: Sequence[str] | None = None,
    builtin: Sequence[type[DocstringBuilderPlugin]] | None = None,
) -> PluginManager:
    """Discover, filter, and instantiate plugins for the current run."""
    builtin_types: list[object] = [
        DataclassFieldDocPlugin,
        LLMSummaryRewritePlugin,
        NormalizeNumpyParamsPlugin,
    ]
    if builtin is not None:
        builtin_types = list(builtin)

    discovered: dict[str, object] = {}
    for plugin_type in builtin_types:
        name = getattr(plugin_type, "name", plugin_type.__name__)
        discovered[name] = plugin_type

    for candidate in _load_entry_points():
        name = getattr(candidate, "name", getattr(candidate, "__name__", None))
        if not name:
            message = "Discovered plugin without a name attribute"
            raise PluginConfigurationError(message)
        discovered.setdefault(name, candidate)

    only_set = {entry.strip() for entry in only or [] if entry.strip()}
    disable_set = {entry.strip() for entry in disable or [] if entry.strip()}

    missing_only = sorted(name for name in only_set if name not in discovered)
    missing_disable = sorted(name for name in disable_set if name not in discovered)
    if missing_only or missing_disable:
        missing = ", ".join([*missing_only, *missing_disable])
        message = f"Unknown plugin(s): {missing}"
        raise PluginConfigurationError(message)

    manager = PluginManager(config=config, repo_root=repo_root)
    manager.available = sorted(discovered)

    for name in manager.available:
        if only_set and name not in only_set:
            manager.skipped.append(name)
            continue
        if name in disable_set:
            manager.disabled.append(name)
            manager.skipped.append(name)
            continue
        plugin = _instantiate_plugin(discovered[name])
        _register_plugin(manager, plugin)

    return manager


__all__ = [
    "PluginConfigurationError",
    "PluginManager",
    "load_plugins",
]
