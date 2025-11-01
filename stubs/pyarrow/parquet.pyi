from __future__ import annotations

from pathlib import Path

from pyarrow import Table

def read_table(source: str | Path) -> Table: ...
def write_table(
    table: Table,
    where: str | Path,
    *,
    compression: str | dict[str, str] | None = ...,
    compression_level: int | None = ...,
    data_page_size: int | None = ...,
    **kwargs: object,
) -> None: ...
