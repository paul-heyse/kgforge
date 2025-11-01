from collections.abc import Callable
from pathlib import Path

def main(
    *,
    root_package: str | None = ...,
    output_path: Path | None = ...,
    root_dir: Path | None = ...,
    detect: Callable[[], str] | None = ...,
    check: bool = ...,
) -> Path: ...
