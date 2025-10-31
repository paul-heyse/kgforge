from __future__ import annotations

from pathlib import Path

class _TemporaryFileWrapper:
    """Temporary file wrapper returned by NamedTemporaryFile."""

    name: str

    def write(self, data: str | bytes) -> int: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> _TemporaryFileWrapper: ...
    def __exit__(self, *args: object) -> None: ...

def NamedTemporaryFile(  # noqa: N802  # Matches stdlib API
    mode: str = "w+b",
    buffering: int = -1,
    encoding: str | None = None,
    newline: str | None = None,
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | Path | None = None,  # noqa: A002  # Matches stdlib parameter name
    delete: bool = True,
    *,
    errors: str | None = None,
    delete_on_close: bool = True,
) -> _TemporaryFileWrapper: ...
