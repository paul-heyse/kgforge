"""Module for linking.calibration.

NavMap:
- isotonic_calibrate: Skeleton: calibrate scores (y=float in [0,1]) vs labels….
"""

from __future__ import annotations


def isotonic_calibrate(pairs: list[tuple[float, int]]) -> dict[str, object]:
    """Skeleton: calibrate scores (y=float in [0,1]) vs labels (0/1)."""
    # NOTE: fit isotonic regression parameters when calibrator is implemented
    return {"kind": "isotonic", "params": []}
