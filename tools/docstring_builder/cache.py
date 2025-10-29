"""File-based cache tracking processed files for docstring builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class CacheEntry:
    """Entry describing a processed file."""

    mtime: float
    config_hash: str


class BuilderCache:
    """Persist and query cache entries keyed by file path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                self._entries[key] = CacheEntry(**value)

    def needs_update(self, file_path: Path, config_hash: str) -> bool:
        """Determine whether a file requires regeneration."""
        key = str(file_path)
        entry = self._entries.get(key)
        mtime = file_path.stat().st_mtime
        if entry is None:
            return True
        if entry.config_hash != config_hash:
            return True
        return entry.mtime < mtime

    def update(self, file_path: Path, config_hash: str) -> None:
        """Record updated metadata for a file."""
        key = str(file_path)
        mtime = file_path.stat().st_mtime
        self._entries[key] = CacheEntry(mtime=mtime, config_hash=config_hash)

    def write(self) -> None:
        """Persist cache entries to disk."""
        payload = {key: asdict(entry) for key, entry in self._entries.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def clear(self) -> None:
        """Reset the cache by removing the backing file."""
        self._entries.clear()
        if self.path.exists():
            self.path.unlink()


__all__ = ["BuilderCache", "CacheEntry"]
