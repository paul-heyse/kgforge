"""Compatibility shim exposing legacy docstring generation utilities."""

# pylint: disable=protected-access

from pathlib import Path

from tools.docstring_builder import legacy as _legacy
from tools.docstring_builder.legacy import (
    DEFAULT_MAGIC_METHOD_FALLBACK,
    DEFAULT_PYDANTIC_ARTIFACT_SUMMARY,
    MAGIC_METHOD_EXTENDED_SUMMARIES,
    PYDANTIC_ARTIFACT_SUMMARIES,
    QUALIFIED_NAME_OVERRIDES,
    annotation_to_text,
    build_docstring,
    build_examples,
    detect_raises,
    extended_summary,
    parameters_for,
    summarize,
)
from tools.docstring_builder.models import DocstringIRParameter

STANDARD_METHOD_EXTENDED_SUMMARIES = _legacy._STANDARD_METHOD_EXTENDED_SUMMARIES


def is_magic(name: str) -> bool:
    """Return True when ``name`` refers to a recognised magic method."""
    return _legacy._is_magic(name)


def is_pydantic_artifact(name: str) -> bool:
    """Return True when ``name`` is associated with Pydantic internals."""
    return _legacy._is_pydantic_artifact(name)


def normalize_qualified_name(name: str) -> str:
    """Return the canonical qualified name for ``name`` using override mappings."""
    return _legacy._normalize_qualified_name(name)


def required_sections(  # noqa: PLR0913
    kind: str,
    parameters: list[DocstringIRParameter],
    returns: str | None,
    raises: list[str],
    *,
    name: str,
    is_public: bool,
) -> list[str]:
    """Return the ordered docstring sections required for the provided symbol metadata."""
    _ = name  # Retained for compatibility with legacy signature
    context = _legacy._RequiredSectionsContext(
        kind=kind,
        parameters=parameters,
        returns_annotation=returns,
        raises=raises,
        is_public=is_public,
    )
    return _legacy._required_sections(context)


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
    "STANDARD_METHOD_EXTENDED_SUMMARIES",
    "annotation_to_text",
    "build_docstring",
    "build_examples",
    "detect_raises",
    "extended_summary",
    "is_magic",
    "is_pydantic_artifact",
    "module_name_for",
    "normalize_qualified_name",
    "parameters_for",
    "process_file",
    "required_sections",
    "summarize",
]
