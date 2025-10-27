"""Module for linking.calibration.

NavMap:
- NavMap: Structure describing a module navmap.
- isotonic_calibrate: Calibrate score/label pairs using an isotonic regression….
"""

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
            "symbols": ["isotonic_calibrate"],
        },
    ],
}


# [nav:anchor isotonic_calibrate]
def isotonic_calibrate(pairs: list[tuple[float, int]]) -> dict[str, object]:
    """Calibrate score/label pairs using an isotonic regression placeholder.

    Parameters
    ----------
    pairs : list[tuple[float, int]]
        Sequence of ``(score, label)`` pairs where score is in ``[0, 1]`` and
        label is either ``0`` or ``1``.

    Returns
    -------
    dict[str, object]
        Calibration artefact describing the fitted model (placeholder).
    """
    # NOTE: fit isotonic regression parameters when calibrator is implemented
    return {"kind": "isotonic", "params": []}
