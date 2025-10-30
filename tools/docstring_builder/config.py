"""Configuration loading utilities for the docstring builder."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without tomllib
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("docstring_builder.toml")
DEFAULT_MARKER = "<!-- auto:docstring-builder v1 -->"
_CACHE_VERSION = "2025-02-06-indent"
_ENV_CONFIG = "KGF_DOCSTRINGS_CONFIG"
_LEGACY_ENV_CONFIG = "DOCSTRING_BUILDER_CONFIG"


@dataclass(slots=True)
class ConfigSelection:
    """Selected configuration path and its provenance."""

    path: Path
    source: str


@dataclass(slots=True)
class PackageSettings:
    """Per-package overrides to steer summaries and opt-outs."""

    summary_verbs: dict[str, str] = field(default_factory=dict)
    opt_out: set[str] = field(default_factory=set)


@dataclass(slots=True)
class BuilderConfig:
    """Runtime configuration resolved from ``docstring_builder.toml``."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    ownership_marker: str = DEFAULT_MARKER
    dynamic_probes: bool = False
    normalize_sections: bool = False
    package_settings: PackageSettings = field(default_factory=PackageSettings)
    navmap_metadata: bool = True
    ignore: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        """Return a stable hash representing the config values."""
        payload = {
            "include": self.include,
            "exclude": self.exclude,
            "ownership_marker": self.ownership_marker,
            "dynamic_probes": self.dynamic_probes,
            "normalize_sections": self.normalize_sections,
            "summary_verbs": self.package_settings.summary_verbs,
            "opt_out": sorted(self.package_settings.opt_out),
            "navmap_metadata": self.navmap_metadata,
            "ignore": self.ignore,
            "builder_cache_version": _CACHE_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        LOGGER.debug("docstring_builder config missing at %s", path)
        return {}
    LOGGER.debug("Loading docstring builder config from %s", path)
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    msg = f"Unsupported list-like value: {value!r}"
    raise TypeError(msg)


def load_config(path: Path | None = None) -> BuilderConfig:
    """Load configuration from the provided path or default location."""
    config_path = path or DEFAULT_CONFIG_PATH
    data = _load_toml(config_path)
    include = _as_list(data.get("include")) or ["src/**/*.py"]
    exclude = _as_list(data.get("exclude")) or ["tests/**", "site/**", "docs/_build/**", "tools/**"]
    ignore = _as_list(data.get("ignore"))
    ownership_marker = data.get("ownership_marker", DEFAULT_MARKER)
    dynamic_probes = bool(data.get("dynamic_probes", False))
    normalize_sections = bool(data.get("normalize_sections", False))
    navmap_metadata = bool(data.get("navmap_metadata", True))

    package_section = data.get("packages", {}) or {}
    summary_verbs = package_section.get("summary_verbs", {}) or {}
    opt_out = set(_as_list(package_section.get("opt_out")))
    package_settings = PackageSettings(
        summary_verbs={str(key): str(value) for key, value in summary_verbs.items()},
        opt_out={str(item) for item in opt_out},
    )

    config = BuilderConfig(
        include=[str(pattern) for pattern in include],
        exclude=[str(pattern) for pattern in exclude],
        ownership_marker=str(ownership_marker),
        dynamic_probes=dynamic_probes,
        normalize_sections=normalize_sections,
        package_settings=package_settings,
        navmap_metadata=navmap_metadata,
        ignore=[str(pattern) for pattern in ignore],
    )
    if path:
        LOGGER.debug("Loaded docstring builder config hash: %s", config.config_hash)
    return config


def resolve_config_path(start: Path | None = None) -> Path:
    """Find configuration file by walking up the directory tree."""
    current = start or Path.cwd()
    for directory in [current, *current.parents]:
        candidate = directory / DEFAULT_CONFIG_PATH
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_PATH


def _coerce_path(value: str) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve(strict=False)
    except FileNotFoundError:  # pragma: no cover - defensive guard for py<3.11
        return path


def select_config_path(override: str | Path | None = None) -> ConfigSelection:
    """Determine the configuration path honouring CLI and environment precedence."""
    if override:
        path = _coerce_path(str(override))
        return ConfigSelection(path=path, source="cli")

    env_override = os.environ.get(_ENV_CONFIG)
    if env_override:
        path = _coerce_path(env_override)
        return ConfigSelection(path=path, source=f"env:{_ENV_CONFIG}")

    legacy_override = os.environ.get(_LEGACY_ENV_CONFIG)
    if legacy_override:
        path = _coerce_path(legacy_override)
        return ConfigSelection(path=path, source=f"env:{_LEGACY_ENV_CONFIG}")

    default_path = resolve_config_path()
    return ConfigSelection(path=default_path, source="default")


def load_config_with_selection(
    override: str | Path | None = None,
) -> tuple[BuilderConfig, ConfigSelection]:
    """Load configuration while also returning metadata about the selection."""
    selection = select_config_path(override)
    config = load_config(selection.path)
    return config, selection


def load_config_from_env() -> BuilderConfig:
    """Load configuration using backwards compatible environment precedence."""
    config, _ = load_config_with_selection(None)
    return config


__all__ = [
    "BuilderConfig",
    "ConfigSelection",
    "PackageSettings",
    "load_config",
    "load_config_from_env",
    "load_config_with_selection",
    "resolve_config_path",
    "select_config_path",
]
