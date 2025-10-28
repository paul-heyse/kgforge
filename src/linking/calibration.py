"""Calibration utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["isotonic_calibrate"]

__navmap__: Final[NavMap] = {
    "title": "linking.calibration",
    "synopsis": "Placeholder calibration utilities for the linking package",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@linking",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "isotonic_calibrate": {
            "owner": "@linking",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor isotonic_calibrate]
def isotonic_calibrate(pairs: list[tuple[float, int]]) -> dict[str, object]:
    """Compute isotonic calibrate.

    Carry out the isotonic calibrate operation.

    Parameters
    ----------
    pairs : List[Tuple[float, int]]
        Description for ``pairs``.

    Returns
    -------
    Mapping[str, object]
        Description of return value.
    """
    
    
    
    
    # NOTE: fit isotonic regression parameters when calibrator is implemented
    return {"kind": "isotonic", "params": []}
