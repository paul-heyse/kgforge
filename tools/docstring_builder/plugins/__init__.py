"""Plugin discovery and execution helpers for the docstring builder."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TypeGuard, TypeVar, cast

from tools._shared.logging import get_logger
from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.models import (
    DocstringBuilderError,
    PluginExecutionError,
)
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
from tools.docstring_builder.plugins.dataclass_fields import DataclassFieldDocPlugin
from tools.docstring_builder.plugins.llm_summary import LLMSummaryRewritePlugin
from tools.docstring_builder.plugins.normalize_numpy_params import (
    NormalizeNumpyParamsPlugin,
)
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult

ENTRY_POINT_GROUP = "kgfoundry.docstrings.plugins"

logging.getLogger(__name__).addHandler(logging.NullHandler())
_LOGGER = get_logger(__name__)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")

type PluginInstance = DocstringBuilderPlugin[object, object] | LegacyPluginProtocol

type RegisteredPlugin = (
    HarvesterPlugin | TransformerPlugin | FormatterPlugin | LegacyPluginAdapter
)
type PluginFactory = Callable[[], PluginInstance]


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
        with self._lock:
            return _run_harvest_pipeline(self.harvesters, context, result)

    def apply_transformers(
        self, file_path: Path, semantics: Iterable[SemanticResult]
    ) -> list[SemanticResult]:
        """Run transformer plugins sequentially for each semantic result."""
        context = self._context(file_path)
        processed: list[SemanticResult] = []
        for entry in semantics:
            with self._lock:
                processed.append(_run_transformer_pipeline(self.transformers, context, entry))
        return processed

    def apply_formatters(
        self, file_path: Path, edits: Iterable[DocstringEdit]
    ) -> list[DocstringEdit]:
        """Run formatter plugins sequentially for each edit."""
        context = self._context(file_path)
        processed: list[DocstringEdit] = []
        for entry in edits:
            with self._lock:
                processed.append(_run_formatter_pipeline(self.formatters, context, entry))
        return processed

    def enabled_plugins(self) -> list[str]:
        """Return the names of active plugins in execution order."""
        return [plugin.name for plugin in self._iter_plugins()]

    def _iter_plugins(self) -> Iterable[RegisteredPlugin]:
        yield from self.harvesters
        yield from self.transformers
        yield from self.formatters


def _run_harvest_pipeline(
    plugins: Iterable[HarvesterPlugin],
    context: PluginContext,
    payload: HarvestResult,
) -> HarvestResult:
    """Execute harvester plugins sequentially for ``payload``."""
    result = payload
    for plugin in plugins:
        result = _invoke_apply(plugin, context, result)
    return result


def _run_transformer_pipeline(
    plugins: Iterable[TransformerPlugin],
    context: PluginContext,
    payload: SemanticResult,
) -> SemanticResult:
    """Execute transformer plugins sequentially for ``payload``."""
    result = payload
    for plugin in plugins:
        result = _invoke_apply(plugin, context, result)
    return result


def _run_formatter_pipeline(
    plugins: Iterable[FormatterPlugin],
    context: PluginContext,
    payload: DocstringEdit,
) -> DocstringEdit:
    """Execute formatter plugins sequentially for ``payload``."""
    result = payload
    for plugin in plugins:
        result = _invoke_apply(plugin, context, result)
    return result


def _invoke_apply(
    plugin: DocstringBuilderPlugin[PayloadT, ResultT],
    context: PluginContext,
    payload: PayloadT,
) -> ResultT:
    """Invoke ``plugin.apply`` with structured error reporting."""
    try:
        return plugin.apply(context, payload)
    except DocstringBuilderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Plugin {plugin.name!r} failed during {plugin.stage} execution"
        file_path = str(context.file_path) if context.file_path is not None else None
        _LOGGER.exception(
            "Plugin execution failed",
            extra={
                "operation": "docstring_builder.plugins",
                "status": "error",
                "plugin": plugin.name,
                "stage": plugin.stage,
                "file_path": file_path,
            },
        )
        raise PluginExecutionError(message) from exc


def _instantiate_plugin(candidate: object) -> RegisteredPlugin:
    name = _resolve_plugin_name(candidate)
    try:
        instance = _materialize_candidate(candidate)
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Failed to instantiate plugin {name!r}"
        raise PluginConfigurationError(message) from exc
    return _ensure_plugin_instance(instance)


def _materialize_candidate(candidate: object) -> object:
    if _is_registered_plugin(candidate) or _is_legacy_plugin(candidate):
        return candidate
    if callable(candidate):
        factory = cast(Callable[[], PluginInstance], candidate)
        return factory()
    message = f"Unsupported plugin candidate {candidate!r}"
    raise TypeError(message)


def _resolve_plugin_name(candidate: object) -> str:
    name_attr: object = getattr(candidate, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    qualname_attr: object = getattr(candidate, "__name__", None)
    if isinstance(qualname_attr, str) and qualname_attr:
        return qualname_attr
    return candidate.__class__.__name__


def _resolve_plugin_name_strict(candidate: object) -> str:
    name_attr: object = getattr(candidate, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    qualname_attr: object = getattr(candidate, "__name__", None)
    if isinstance(qualname_attr, str) and qualname_attr:
        return qualname_attr
    message = "Discovered plugin without a name attribute"
    raise PluginConfigurationError(message)


def _ensure_plugin_instance(obj: object) -> RegisteredPlugin:
    name = obj.__class__.__name__
    name_attr: object = getattr(obj, "name", None)
    if isinstance(name_attr, str) and name_attr:
        name = name_attr
    stage_attr: object = getattr(obj, "stage", None)
    if not _is_valid_stage(stage_attr):
        message = f"Plugin {name!r} declares invalid stage {stage_attr!r}"
        raise PluginConfigurationError(message)
    if _is_registered_plugin(obj):  # pragma: no cover - runtime check
        return obj
    if _is_legacy_plugin(obj):
        try:
            return LegacyPluginAdapter.create(obj)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"Legacy plugin {name!r} is misconfigured"
            raise PluginConfigurationError(message) from exc
    message = f"Plugin {name!r} must define apply() or run()"
    raise PluginConfigurationError(message)


def _register_plugin(manager: PluginManager, plugin: RegisteredPlugin) -> None:
    if isinstance(plugin, HarvesterPlugin):
        manager.harvesters.append(plugin)
    elif isinstance(plugin, TransformerPlugin):
        manager.transformers.append(plugin)
    elif isinstance(plugin, FormatterPlugin):
        manager.formatters.append(plugin)
    else:  # pragma: no cover - defensive guard
        message = f"Unsupported plugin stage: {plugin.stage!r}"
        raise PluginConfigurationError(message)


def _is_registered_plugin(candidate: object) -> TypeGuard[RegisteredPlugin]:
    return isinstance(
        candidate,
        (HarvesterPlugin, TransformerPlugin, FormatterPlugin, LegacyPluginAdapter),
    )


def _is_legacy_plugin(candidate: object) -> TypeGuard[LegacyPluginProtocol]:
    return isinstance(candidate, LegacyPluginProtocol)


def _is_valid_stage(value: object) -> TypeGuard[PluginStage]:
    return value in {"harvester", "transformer", "formatter"}


def _load_entry_points() -> list[object]:
    loaded: list[object] = []
    for entry_point in metadata.entry_points().select(group=ENTRY_POINT_GROUP):
        try:
            candidate: object = entry_point.load()
            loaded.append(candidate)
        except Exception as exc:  # pragma: no cover - best effort guard
            message = f"Failed to load plugin entry point {entry_point.name!r}"
            raise PluginConfigurationError(message) from exc
    return loaded


def _builtin_candidates(builtin: Sequence[PluginFactory] | None) -> tuple[object, ...]:
    if builtin is None:
        return (
            DataclassFieldDocPlugin,
            LLMSummaryRewritePlugin,
            NormalizeNumpyParamsPlugin,
        )
    return tuple(builtin)


def load_plugins(
    config: BuilderConfig,
    repo_root: Path,
    *,
    only: Sequence[str] | None = None,
    disable: Sequence[str] | None = None,
    builtin: Sequence[PluginFactory] | None = None,
) -> PluginManager:
    """Discover, filter, and instantiate plugins for the current run."""
    builtin_candidates = _builtin_candidates(builtin)

    discovered: dict[str, object] = {
        _resolve_plugin_name(candidate): candidate for candidate in builtin_candidates
    }

    for candidate in _load_entry_points():
        name = _resolve_plugin_name_strict(candidate)
        if name not in discovered:
            discovered[name] = candidate

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
    "DocstringPayload",
    "PluginConfigurationError",
    "PluginManager",
    "load_plugins",
]
