from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class ModuleInfo:
    module: str
    path: Path
    exports: list[str]
    sections: dict[str, int]
    anchors: dict[str, int]
    nav_sections: list[Any]
    navmap_dict: dict[str, Any]

def _collect_module(py: Path) -> ModuleInfo | None: ...
def build_index(*, json_path: Path | None = ...) -> dict[str, Any]: ...
def main(argv: list[str] | None = ...) -> int: ...
