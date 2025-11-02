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

class TemporaryDirectory:
    """Context manager for temporary directories."""

    name: str

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | Path | None = None,  # noqa: A002  # Matches stdlib parameter name
    ) -> None: ...
    def __enter__(self) -> str: ...
    def __exit__(self, *args: object) -> None: ...
    def cleanup(self) -> None: ...

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
def mkdtemp(  # Matches stdlib API
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | Path | None = None,  # noqa: A002  # Matches stdlib parameter name
) -> str: ...
