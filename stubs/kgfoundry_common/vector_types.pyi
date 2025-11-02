from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NewType

import numpy as np
import numpy.typing as npt

VectorMatrix = npt.NDArray[np.float32]
VECTOR_MATRIX_NDIM: Final[int]

VectorId = NewType("VectorId", str)

class VectorValidationError(ValueError):
    errors: tuple[str, ...]

    def __init__(self, message: str, *, errors: Sequence[str] | None = ...) -> None: ...

@dataclass(frozen=True, slots=True)
class VectorBatch:
    ids: tuple[VectorId, ...]
    matrix: VectorMatrix

    def __post_init__(self) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def count(self) -> int: ...

def assert_vector_matrix(arr: object) -> VectorMatrix: ...
def validate_vector_batch(batch: VectorBatch) -> VectorBatch: ...
def coerce_vector_batch(records: Iterable[Mapping[str, object]]) -> VectorBatch: ...
