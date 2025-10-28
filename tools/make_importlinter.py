"""Overview of make importlinter.

This module bundles make importlinter logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


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
    root_package: str | None = None,
    output_path: Path | None = None,
    root_dir: Path | None = None,
    detect: Callable[[], str] | None = None,
) -> Path:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    root_package : str | None
    root_package : str | None, optional, default=None
        Description for ``root_package``.
    output_path : Path | None
    output_path : Path | None, optional, default=None
        Description for ``output_path``.
    root_dir : Path | None
    root_dir : Path | None, optional, default=None
        Description for ``root_dir``.
    detect : Callable[[], str] | None
    detect : Callable[[], str] | None, optional, default=None
        Description for ``detect``.
    
    Returns
    -------
    Path
        Description of return value.
    
    Examples
    --------
    >>> from tools.make_importlinter import main
    >>> result = main()
    >>> result  # doctest: +ELLIPSIS
    ...
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
    """Import detect primary.

    Returns
    -------
    str
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _import_detect_primary(...)
    """
    from tools.detect_pkg import detect_primary

    return detect_primary()


if __name__ == "__main__":
    out_path = main()
    print(f"Wrote {out_path}")
