"""Utilities for generating a manifest of installed Tree-sitter grammars."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

from kgfoundry_common.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
LANG_BUILD_DIR = ROOT / "codeintel" / "build"
LANG_MANIFEST = LANG_BUILD_DIR / "languages.json"

LANG_PACKAGES: Final[dict[str, str]] = {
    "python": "tree_sitter_python",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
    "toml": "tree_sitter_toml",
    "markdown": "tree_sitter_markdown",
}


def _resolve_package(package: str) -> tuple[str, str]:
    """Resolve a Tree-sitter language package and return metadata.

    Parameters
    ----------
    package : str
        Importable package name that exposes a ``language`` factory function.

    Returns
    -------
    tuple[str, str]
        The module's fully qualified name and its installed version.

    Raises
    ------
    ConfigurationError
        If the package cannot be imported, its version cannot be determined, or it
        does not expose the expected ``language`` factory function.
    """
    try:
        module = import_module(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        message = (
            f"Tree-sitter package '{package}' is not installed. Run 'scripts/bootstrap.sh' "
            "to sync dependencies."
        )
        raise ConfigurationError(message, cause=exc) from exc
    try:
        pkg_version = version(package)
    except PackageNotFoundError as exc:  # pragma: no cover - should not happen for wheels
        message = f"Unable to determine version for package '{package}'."
        raise ConfigurationError(message, cause=exc) from exc
    if not hasattr(module, "language"):
        message = f"Tree-sitter package '{package}' does not expose a 'language()' factory."
        raise ConfigurationError(message)
    return module.__name__, pkg_version


def main() -> int:
    """Generate a manifest enumerating the bundled Tree-sitter language packages.

    Returns
    -------
    int
        Exit status code suitable for ``sys.exit``.

    Raises
    ------
    ConfigurationError
        If any configured Tree-sitter package cannot be resolved.
    """
    LANG_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    manifest_languages: dict[str, dict[str, str]] = {}
    for name, package in LANG_PACKAGES.items():
        try:
            module_name, pkg_version = _resolve_package(package)
        except ConfigurationError as exc:
            message = f"Failed to resolve Tree-sitter package '{package}' for language '{name}'."
            raise ConfigurationError(message, cause=exc) from exc
        manifest_languages[name] = {
            "package": package,
            "module": module_name,
            "version": pkg_version,
        }
    manifest = {"languages": manifest_languages}
    LANG_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sys.stdout.write(f"Wrote Tree-sitter manifest to {LANG_MANIFEST}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
