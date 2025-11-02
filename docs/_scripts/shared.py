"""Shared utilities for documentation build scripts."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from tools import StructuredLoggerAdapter, get_logger, with_fields
from tools.detect_pkg import detect_packages, detect_primary
from tools.griffe_utils import resolve_griffe

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from griffe import Object as GriffeRuntimeObject

    GriffeObject = GriffeRuntimeObject
else:
    GriffeObject = object

LinkMode = Literal["editor", "github", "both"]

_LINK_MODE_MAP: dict[str, LinkMode] = {
    "editor": "editor",
    "github": "github",
    "both": "both",
}


@dataclass(frozen=True, slots=True)
class BuildEnvironment:
    """Repository paths used by docs tooling."""

    root: Path
    src: Path
    tools_dir: Path


@dataclass(frozen=True, slots=True)
class DocsSettings:
    """Normalised environment configuration for docs builds."""

    packages: tuple[str, ...]
    link_mode: LinkMode
    github_org: str | None
    github_repo: str | None
    github_sha: str | None
    docs_build_dir: Path
    navmap_candidates: tuple[Path, ...]


class GriffeLoader(Protocol):
    """Runtime loader capable of resolving packages into Griffe nodes."""

    def load(self, package: str) -> GriffeObject:  # pragma: no cover - runtime protocol
        """Return the module graph for ``package``."""
        ...


class WarningLogger(Protocol):
    """Logger interface supporting ``warning`` calls."""

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log a warning message."""


_LOADER_FACTORY = cast(Callable[[Sequence[str]], GriffeLoader], resolve_griffe().loader_type)


@lru_cache(maxsize=1)
def _detect_environment_cached() -> BuildEnvironment:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    tools_dir = root / "tools"
    return BuildEnvironment(root=root, src=src, tools_dir=tools_dir)


def detect_environment() -> BuildEnvironment:
    """Return filesystem locations relevant to docs tooling."""
    return _detect_environment_cached()


def ensure_sys_paths(env: BuildEnvironment) -> None:
    """Add docs-relevant paths to ``sys.path`` when missing."""
    for candidate in (env.src, env.root, env.tools_dir):
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def _load_settings_cached() -> DocsSettings:
    env = detect_environment()

    env_pkgs = os.environ.get("DOCS_PKG")
    if env_pkgs:
        packages = tuple(pkg.strip() for pkg in env_pkgs.split(",") if pkg.strip())
    else:
        discovered = detect_packages()
        packages = tuple(discovered or [detect_primary()])

    link_mode_raw = os.environ.get("DOCS_LINK_MODE", "both").lower()
    link_mode = _LINK_MODE_MAP.get(link_mode_raw)
    if link_mode is None:
        LOGGER.warning("Unsupported DOCS_LINK_MODE '%s'; defaulting to 'both'", link_mode_raw)
        link_mode = "both"

    docs_build_dir = env.root / "docs" / "_build"
    navmap_candidates: tuple[Path, ...] = (
        docs_build_dir / "navmap.json",
        docs_build_dir / "navmap" / "navmap.json",
        env.root / "site" / "_build" / "navmap" / "navmap.json",
    )

    return DocsSettings(
        packages=packages,
        link_mode=link_mode,
        github_org=os.environ.get("DOCS_GITHUB_ORG"),
        github_repo=os.environ.get("DOCS_GITHUB_REPO"),
        github_sha=os.environ.get("DOCS_GITHUB_SHA"),
        docs_build_dir=docs_build_dir,
        navmap_candidates=navmap_candidates,
    )


def load_settings() -> DocsSettings:
    """Return docs settings derived from environment variables."""
    return _load_settings_cached()


def make_loader(env: BuildEnvironment) -> GriffeLoader:
    """Instantiate a Griffe loader configured for the repository layout."""
    search_root = env.src if env.src.exists() else env.root
    return _LOADER_FACTORY([str(search_root)])


def resolve_git_sha(
    env: BuildEnvironment,
    settings: DocsSettings,
    *,
    logger: WarningLogger,
) -> str:
    """Return the Git SHA used when rendering GitHub permalinks."""
    if settings.github_sha:
        return settings.github_sha

    git_dir = env.root / ".git"
    head_path = git_dir / "HEAD"
    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Git metadata missing; using HEAD sentinel",
            extra={"status": "fallback"},
        )
        return "HEAD"

    if head_contents.startswith("ref: "):
        ref_path = git_dir / head_contents.removeprefix("ref: ").strip()
        with suppress(FileNotFoundError):
            ref_contents = ref_path.read_text(encoding="utf-8").strip()
            if ref_contents:
                return ref_contents
        logger.warning(
            "Ref %s missing; using HEAD sentinel",
            ref_path,
            extra={"status": "fallback"},
        )
        return "HEAD"

    return head_contents or "HEAD"


def make_logger(
    operation: str,
    *,
    artifact: str | None = None,
    logger: logging.Logger | StructuredLoggerAdapter | None = None,
) -> StructuredLoggerAdapter:
    """Return a structured logger adapter enriched with docs metadata."""
    base_logger: logging.Logger | StructuredLoggerAdapter = (
        logger if logger is not None else get_logger(f"docs.{operation}")
    )

    fields: dict[str, object] = {"operation": operation}
    if artifact is not None:
        fields["artifact"] = artifact
    return with_fields(base_logger, **fields)
