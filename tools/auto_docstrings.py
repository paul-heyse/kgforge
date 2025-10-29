"""Compatibility shim exposing the legacy auto-docstring helpers."""

from __future__ import annotations

from pathlib import Path

from tools.docstring_builder import legacy as _legacy
from tools.docstring_builder.legacy import (
    _STANDARD_METHOD_EXTENDED_SUMMARIES,
    DEFAULT_MAGIC_METHOD_FALLBACK,
    DEFAULT_PYDANTIC_ARTIFACT_SUMMARY,
    MAGIC_METHOD_EXTENDED_SUMMARIES,
    PYDANTIC_ARTIFACT_SUMMARIES,
    QUALIFIED_NAME_OVERRIDES,
    _is_magic,
    _is_pydantic_artifact,
    _normalize_qualified_name,
    _required_sections,
    annotation_to_text,
    build_docstring,
    build_examples,
    detect_raises,
    extended_summary,
    parameters_for,
    summarize,
)

REPO_ROOT = _legacy.REPO_ROOT
SRC_ROOT = _legacy.SRC_ROOT


def _sync_roots() -> None:
    """Sync the compatibility shim root paths with the legacy implementation."""
    _legacy.REPO_ROOT = REPO_ROOT
    _legacy.SRC_ROOT = SRC_ROOT


def module_name_for(file_path: Path) -> str:
    """Return the module path for ``file_path`` using the configured roots."""
    _sync_roots()
    return _legacy.module_name_for(file_path)


def process_file(file_path: Path) -> bool:
    """Generate docstrings for the provided source file."""
    _sync_roots()
    return _legacy.process_file(file_path)


__all__ = [
    "DEFAULT_MAGIC_METHOD_FALLBACK",
    "DEFAULT_PYDANTIC_ARTIFACT_SUMMARY",
    "MAGIC_METHOD_EXTENDED_SUMMARIES",
    "PYDANTIC_ARTIFACT_SUMMARIES",
    "QUALIFIED_NAME_OVERRIDES",
    "REPO_ROOT",
    "SRC_ROOT",
    "_STANDARD_METHOD_EXTENDED_SUMMARIES",
    "_is_magic",
    "_is_pydantic_artifact",
    "_normalize_qualified_name",
    "_required_sections",
    "annotation_to_text",
    "build_docstring",
    "build_examples",
    "detect_raises",
    "extended_summary",
    "module_name_for",
    "parameters_for",
    "process_file",
    "summarize",
]
