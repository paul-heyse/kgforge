"""Typed configuration objects for docs toolchain operations.

This module provides frozen dataclasses for configuring docs toolchain
symbol index build and delta operations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocsSymbolIndexConfig:
    """Configuration for symbol index building operations.

    This frozen dataclass defines all options for the symbol index builder,
    ensuring immutable configuration throughout the build process.

    Attributes
    ----------
    output_format : str
        Output format for the symbol index (default: "json").
    include_private : bool
        Whether to include private symbols in the index (default: False).
    max_depth : int | None
        Maximum depth for symbol traversal, or None for unlimited (default: None).
    """

    output_format: str = "json"
    include_private: bool = False
    max_depth: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        valid_formats = {"json", "yaml", "xml"}
        if self.output_format not in valid_formats:
            msg = f"output_format must be one of {valid_formats}, got {self.output_format!r}"
            raise ValueError(msg)
        if self.max_depth is not None and self.max_depth < 0:
            msg = f"max_depth must be non-negative or None, got {self.max_depth}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DocsDeltaConfig:
    """Configuration for symbol delta operations.

    This frozen dataclass defines all options for computing deltas between
    symbol indices to track changes and breaking API modifications.

    Attributes
    ----------
    include_removals : bool
        Whether to include removed symbols in delta (default: True).
    include_modifications : bool
        Whether to include modified symbols in delta (default: True).
    severity_threshold : str
        Minimum severity level to report: "info", "warning", "error" (default: "info").
    """

    include_removals: bool = True
    include_modifications: bool = True
    severity_threshold: str = "info"

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        valid_thresholds = {"error", "info", "warning"}
        if self.severity_threshold not in valid_thresholds:
            msg = f"severity_threshold must be one of {valid_thresholds}, got {self.severity_threshold!r}"
            raise ValueError(msg)


__all__ = [
    "DocsDeltaConfig",
    "DocsSymbolIndexConfig",
]
