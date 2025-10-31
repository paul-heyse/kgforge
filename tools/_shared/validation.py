"""Utility helpers for validating and normalising CLI inputs."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ValidationError",
    "require_directory",
    "require_file",
    "resolve_path",
]


class ValidationError(ValueError):
    """Raised when user-provided inputs fail validation."""


def resolve_path(
    value: str | Path,
    *,
    base: Path | None = None,
    strict: bool = False,
) -> Path:
    """Resolve ``value`` relative to ``base`` (if provided) into an absolute path."""
    candidate = Path(value).expanduser()
    if base is not None and not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=strict)


def require_file(
    value: str | Path,
    *,
    base: Path | None = None,
    description: str = "file",
) -> Path:
    """Return ``value`` as a path and ensure it references an existing file."""
    candidate = resolve_path(value, base=base, strict=False)
    if not candidate.exists():
        message = f"{description.capitalize()} '{candidate}' does not exist"
        raise ValidationError(message)
    if not candidate.is_file():
        message = f"{description.capitalize()} '{candidate}' must be a file"
        raise ValidationError(message)
    return candidate


def require_directory(
    value: str | Path,
    *,
    base: Path | None = None,
    description: str = "directory",
) -> Path:
    """Return ``value`` as a path and ensure it references an existing directory."""
    candidate = resolve_path(value, base=base, strict=False)
    if not candidate.exists():
        message = f"{description.capitalize()} '{candidate}' does not exist"
        raise ValidationError(message)
    if not candidate.is_dir():
        message = f"{description.capitalize()} '{candidate}' must be a directory"
        raise ValidationError(message)
    return candidate
