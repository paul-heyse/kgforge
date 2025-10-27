"""Module for linking.calibration."""

from __future__ import annotations


def isotonic_calibrate(pairs: list[tuple[float, int]]) -> dict[str, object]:
    """Skeleton: calibrate scores (y=float in [0,1]) vs labels (0/1)."""
    # TODO: fit isotonic regression parameters
    return {"kind": "isotonic", "params": []}
