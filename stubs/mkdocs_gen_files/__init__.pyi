from __future__ import annotations

from contextlib import AbstractContextManager
from os import PathLike
from typing import TextIO

__all__ = ["open"]

def open(
    path: str | PathLike[str], mode: str = "w"
) -> AbstractContextManager[TextIO]: ...  # noqa: A001 - public API reuses builtin name
