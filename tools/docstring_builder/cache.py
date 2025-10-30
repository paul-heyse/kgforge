"""File-based cache tracking processed files for docstring builder."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
        self._lock = threading.Lock()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("Cache contents must be a mapping.")
                for key, value in data.items():
                    if not isinstance(value, dict):
                        raise TypeError("Cache entry values must be mappings.")
                    self._entries[key] = CacheEntry(**value)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Failed to load builder cache from %s: %s. Resetting cache.",
                    path,
                    exc,
                )
                self.clear()

    def needs_update(self, file_path: Path, config_hash: str) -> bool:
        """Determine whether a file requires regeneration."""
        key = str(file_path)
        mtime = file_path.stat().st_mtime
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return True
            if entry.config_hash != config_hash:
                return True
            return entry.mtime < mtime

    def update(self, file_path: Path, config_hash: str) -> None:
        """Record updated metadata for a file."""
        key = str(file_path)
        mtime = file_path.stat().st_mtime
        with self._lock:
            self._entries[key] = CacheEntry(mtime=mtime, config_hash=config_hash)

    def write(self) -> None:
        """Persist cache entries to disk."""
        with self._lock:
            payload = {key: asdict(entry) for key, entry in self._entries.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def clear(self) -> None:
        """Reset the cache by removing the backing file."""
        with self._lock:
            self._entries.clear()
        if self.path.exists():
            self.path.unlink()


__all__ = ["BuilderCache", "CacheEntry"]
