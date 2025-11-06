"""Optional wrapper around the mkdocs-d2 plugin."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from typing import Any

from mkdocs.plugins import BasePlugin

LOGGER = logging.getLogger("mkdocs.plugins.optional-d2")

try:  # pragma: no cover - dependency may be missing in minimal environments
    from mkdocs_d2_plugin.plugin import D2Plugin
except ModuleNotFoundError:  # pragma: no cover - surface friendly warning later
    D2Plugin = None  # type: ignore[assignment]


def _noop(*_args: object, **_kwargs: object) -> None:
    """Return ``None`` regardless of invocation arguments."""


class OptionalD2Plugin(BasePlugin[dict[str, Any]]):
    """Conditionally delegate to the mkdocs-d2 plugin when available."""

    if D2Plugin is not None:
        config_scheme = D2Plugin.config_scheme  # type: ignore[attr-defined]
    else:  # pragma: no cover - exercised when the dependency is absent
        config_scheme: tuple[tuple[str, object], ...] = ()

    def __init__(self) -> None:
        super().__init__()
        self._delegate: BasePlugin[dict[str, Any]] | None = None
        self._warned = False

    def load_config(self, config: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        """Prepare the wrapped mkdocs-d2 plugin when prerequisites are met.

        Returns
        -------
        dict[str, Any]
            The validated plugin configuration preserved from ``BasePlugin``.
        """
        result = super().load_config(config, **kwargs)
        if D2Plugin is None:
            self._warn_once(
                "mkdocs-d2-plugin is not installed; skipping D2 diagram rendering.",
            )
            self._delegate = None
            return result

        if shutil.which("d2") is None:
            self._warn_once(
                "The 'd2' executable was not found on PATH; existing diagrams will be "
                "served but new renders are skipped. Install the D2 CLI to refresh diagrams.",
            )
            self._delegate = None
            return result

        delegate: BasePlugin[dict[str, Any]] = D2Plugin()
        delegate.load_config(config, **kwargs)
        self._delegate = delegate
        return result

    def _warn_once(self, message: str) -> None:
        """Emit ``message`` at warning level exactly once per build."""
        if not self._warned:
            LOGGER.warning(message)
            self._warned = True

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Delegate mkdocs lifecycle hooks to the wrapped plugin when active.

        Returns
        -------
        collections.abc.Callable[..., Any]
            Lifecycle hook implementation sourced from the wrapped plugin or a
            no-op stub when delegation is disabled.

        Raises
        ------
        AttributeError
            If ``name`` does not resemble an MkDocs plugin lifecycle hook.
        """
        if not name.startswith("on_"):
            raise AttributeError(name)

        if self._delegate is None:
            return _noop

        return getattr(self._delegate, name)


__all__ = ["OptionalD2Plugin"]
