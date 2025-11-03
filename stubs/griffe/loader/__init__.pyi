from pathlib import Path
from typing import Any

class ModuleLoader:
    def load(
        self,
        objspec: str | Path | None = None,
        /,
        *,
        submodules: bool = True,
        try_relative_path: bool = True,
        find_stubs_package: bool = False,
    ) -> Any: ...  # noqa: ANN401

__all__ = [
    "ModuleLoader",
]
