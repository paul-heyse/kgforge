"""Helpers for loading nav metadata stored alongside packages."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any


def _candidate_sidecars(package: str) -> list[Path]:
    """Return ordered sidecar file candidates for ``package``.

    Returns
    -------
    list[Path]
        Candidate paths in priority order where `_nav.json` sidecars may live.
    """
    spec = importlib.util.find_spec(package)
    if spec is None:
        return []

    candidates: list[Path] = []
    origin = Path(spec.origin) if isinstance(spec.origin, str) else None

    if origin is not None:
        if origin.name != "__init__.py":
            candidates.append(origin.with_name(f"{origin.stem}._nav.json"))
        candidates.append(origin.parent / "_nav.json")

    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            location_path = Path(location)
            candidate = location_path / "_nav.json"
            if candidate not in candidates:
                candidates.append(candidate)

    # Remove duplicates while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    return deduped


def _load_sidecar_data(package: str) -> dict[str, Any]:
    """Return metadata loaded from package sidecar files.

    Returns
    -------
    dict[str, Any]
        Parsed JSON payload when a sidecar exists, otherwise an empty dict.
    """
    for candidate in _candidate_sidecars(package):
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _load_runtime_nav(package: str) -> dict[str, Any]:
    """Return runtime ``__navmap__`` data if available.

    Returns
    -------
    dict[str, Any]
        Deep-copied runtime navmap if exposed by the module, else empty dict.
    """
    module = sys.modules.get(package)
    if module is None:
        try:
            module = importlib.import_module(package)
        except ImportError:
            module = None
    if module is None:
        return {}
    runtime_nav = getattr(module, "__navmap__", None)
    if isinstance(runtime_nav, dict):
        return copy.deepcopy(runtime_nav)
    return {}


@cache
def load_nav_metadata(package: str, exports: tuple[str, ...]) -> dict[str, Any]:
    """Return navigation metadata for ``package``.

    Parameters
    ----------
    package : str
        Fully qualified package name whose metadata should be loaded.
    exports : tuple[str, ...]
        Public export names exposed via ``__all__``. These drive the default
        section and symbol lists when the sidecar omits explicit values.

    Returns
    -------
    dict[str, Any]
        Dictionary matching the ``__navmap__`` schema historically used by the
        documentation toolchain. When no sidecar is present a minimal fallback
        derived from ``exports`` is returned.
    """
    data = _load_sidecar_data(package) or _load_runtime_nav(package)

    normalized_exports = list(dict.fromkeys(exports))

    sections = data.get("sections") or [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": normalized_exports,
        }
    ]

    raw_symbols = data.get("symbols") or {}
    symbol_meta = {
        name: raw_symbols[name] if isinstance(raw_symbols.get(name), dict) else {}
        for name in normalized_exports
    }

    navmap: dict[str, Any] = {
        "title": data.get("title", package),
        "synopsis": data.get("synopsis"),
        "exports": normalized_exports,
        "sections": sections,
        "module_meta": data.get("module_meta", {}),
        "symbols": symbol_meta,
    }

    extras = {key: value for key, value in data.items() if key not in navmap}
    navmap.update(extras)
    return navmap
