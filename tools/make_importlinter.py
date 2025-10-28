"""Make Importlinter utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional


def _build_template(pkg: str) -> str:
    """Return the contents of the ``.importlinter`` file for ``pkg``."""

    return f"""[importlinter]
root_package = {pkg}

[importlinter:contract:layers]
name = Respect layered architecture
type = layers
layers =
    {pkg}.presentation
    {pkg}.domain
    {pkg}.infrastructure
"""


def main(
    *,
    root_package: Optional[str] = None,
    output_path: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    detect: Optional[Callable[[], str]] = None,
) -> Path:
    """Generate the ``.importlinter`` configuration file.

    Parameters are optional so tests can call this function without touching the
    project's real ``.importlinter`` file.
    """

    detected_root = root_dir or Path(__file__).resolve().parents[1]
    if root_package is None:
        detect_primary = detect or _import_detect_primary
        pkg = detect_primary()
    else:
        pkg = root_package
    destination = Path(output_path) if output_path is not None else detected_root / ".importlinter"
    destination.write_text(_build_template(pkg), encoding="utf-8")
    return destination


def _import_detect_primary() -> str:
    from detect_pkg import detect_primary

    return detect_primary()


if __name__ == "__main__":
    out_path = main()
    print(f"Wrote {out_path}")
