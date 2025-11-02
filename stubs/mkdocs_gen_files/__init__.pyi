from __future__ import annotations

from contextlib import AbstractContextManager
from os import PathLike
from typing import TextIO

__all__ = ["open"]

def open(  # noqa: A001
    path: str | PathLike[str], mode: str = "w"
) -> AbstractContextManager[TextIO]: ...
