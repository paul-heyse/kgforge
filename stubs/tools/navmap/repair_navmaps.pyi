from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.navmap.build_navmap import ModuleInfo

@dataclass(frozen=True)
class RepairResult:
    """Aggregate outcome for repairing a single module."""

    module: Path
    messages: list[str]
    changed: bool
    applied: bool

def repair_module(info: ModuleInfo, apply: bool = False) -> RepairResult: ...
