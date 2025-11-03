"""Compatibility shim exposing legacy docstring generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.docstring_builder.legacy import (
    DEFAULT_MAGIC_METHOD_FALLBACK,
    DEFAULT_PYDANTIC_ARTIFACT_SUMMARY,
    MAGIC_METHOD_EXTENDED_SUMMARIES,
    PYDANTIC_ARTIFACT_SUMMARIES,
    QUALIFIED_NAME_OVERRIDES,
    STANDARD_METHOD_EXTENDED_SUMMARIES,
    annotation_to_text,
    build_docstring,
    build_examples,
    configure_roots,
    detect_raises,
    extended_summary,
    parameters_for,
    summarize,
)
from tools.docstring_builder.legacy import (
    REPO_ROOT as LEGACY_REPO_ROOT,
)
from tools.docstring_builder.legacy import (
    SRC_ROOT as LEGACY_SRC_ROOT,
)
from tools.docstring_builder.legacy import (
    is_magic as legacy_is_magic,
)
from tools.docstring_builder.legacy import (
    is_pydantic_artifact as legacy_is_pydantic_artifact,
)
from tools.docstring_builder.legacy import (
    module_name_for as legacy_module_name_for,
)
from tools.docstring_builder.legacy import (
    normalize_qualified_name as legacy_normalize_qualified_name,
)
from tools.docstring_builder.legacy import (
    process_file as legacy_process_file,
)
from tools.docstring_builder.legacy import (
    required_sections as legacy_required_sections,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tools.docstring_builder.models import DocstringIRParameter


def is_magic(name: str) -> bool:
    """Return True when ``name`` refers to a recognised magic method."""
    return legacy_is_magic(name)


def is_pydantic_artifact(name: str) -> bool:
    """Return True when ``name`` is associated with Pydantic internals."""
    return legacy_is_pydantic_artifact(name)


def normalize_qualified_name(name: str) -> str:
    """Return the canonical qualified name for ``name`` using override mappings."""
    return legacy_normalize_qualified_name(name)


def required_sections(
    kind: str,
    parameters: Sequence[DocstringIRParameter],
    returns: str | None,
    raises: Sequence[str],
    **options: object,
) -> list[str]:
    """Return the ordered docstring sections required for the provided symbol metadata."""
    name = options.pop("name", None)
    is_public_raw = options.pop("is_public", None)
    if options:
        unexpected = ", ".join(sorted(options))
        message = f"Unexpected keyword arguments: {unexpected}"
        raise TypeError(message)
    if is_public_raw is None:
        message = "required_sections() missing required keyword argument: 'is_public'"
        raise TypeError(message)
    is_public = bool(is_public_raw)
    _ = name  # Retained for compatibility with legacy signature
    return legacy_required_sections(
        kind,
        parameters,
        returns,
        raises,
        is_public=is_public,
    )


REPO_ROOT = LEGACY_REPO_ROOT
SRC_ROOT = LEGACY_SRC_ROOT


def _sync_roots() -> None:
    """Sync the compatibility shim root paths with the legacy implementation."""
    configure_roots(REPO_ROOT, SRC_ROOT)


def module_name_for(file_path: Path) -> str:
    """Return the module path for ``file_path`` using the configured roots."""
    _sync_roots()
    return legacy_module_name_for(file_path)


def process_file(file_path: Path) -> bool:
    """Generate docstrings for the provided source file."""
    _sync_roots()
    return legacy_process_file(file_path)


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
