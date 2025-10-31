# ruff: noqa: N815
"""File-based cache tracking processed files for docstring builder."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from tools import validate_tools_payload

logger = logging.getLogger(__name__)

DOCSTRING_CACHE_SCHEMA = "docstring_cache.json"
DOCSTRING_CACHE_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/docstring-cache.json"
DOCSTRING_CACHE_VERSION = "1.0.0"


class CacheEntry(msgspec.Struct, kw_only=True):
    """Entry describing a processed file."""

    mtime: float
    config_hash: str


class CacheDocument(msgspec.Struct, kw_only=True):
    """Persisted cache document with schema metadata."""

    schemaVersion: str = DOCSTRING_CACHE_VERSION
    schemaId: str = DOCSTRING_CACHE_SCHEMA_ID
    generatedAt: str = msgspec.field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    entries: dict[str, CacheEntry] = msgspec.field(default_factory=dict)


class BuilderCache:
    """Persist and query cache entries keyed by file path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._document = CacheDocument()
        self._lock = threading.Lock()
        if path.exists():
            raw_payload = path.read_bytes()
            try:
                self._document = msgspec.json.decode(raw_payload, type=CacheDocument)
            except msgspec.DecodeError:
                self._load_legacy_payload(raw_payload)
            except TypeError as exc:
                self._handle_load_error("Cache entry schema mismatch", exc)
            else:
                try:
                    validate_tools_payload(
                        msgspec.to_builtins(self._document), DOCSTRING_CACHE_SCHEMA
                    )
                except Exception as exc:  # noqa: BLE001 - propagate structured log but reset cache
                    self._handle_load_error("Cache schema validation failed", exc)

    def needs_update(self, file_path: Path, config_hash: str) -> bool:
        """Determine whether a file requires regeneration.

        Returns
        -------
        bool
            ``True`` when the cache entry is missing or stale.
        """
        key = str(file_path)
        mtime = file_path.stat().st_mtime
        with self._lock:
            entry = self._document.entries.get(key)
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
            self._document.entries[key] = CacheEntry(mtime=mtime, config_hash=config_hash)
            self._document.generatedAt = datetime.now(tz=UTC).isoformat()

    def write(self) -> None:
        """Persist cache entries to disk."""
        with self._lock:
            payload = msgspec.to_builtins(self._document)
        validate_tools_payload(payload, DOCSTRING_CACHE_SCHEMA)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2)
        self.path.write_text(encoded, encoding="utf-8")

    def clear(self) -> None:
        """Reset the cache by removing the backing file."""
        with self._lock:
            self._document = CacheDocument()
        if self.path.exists():
            self.path.unlink()

    def _handle_load_error(self, reason: str, exc: Exception | None = None) -> None:
        """Log and reset the cache when persisted data is invalid."""
        if exc is not None:
            logger.warning(
                "Failed to load builder cache from %s. Resetting cache: %s",
                self.path,
                reason,
                exc_info=exc,
            )
        else:
            logger.warning(
                "Failed to load builder cache from %s. Resetting cache: %s",
                self.path,
                reason,
            )
        self.clear()

    def _load_legacy_payload(self, raw_payload: bytes) -> None:
        """Load and normalise legacy cache payloads without schema metadata."""
        try:
            legacy_entries = msgspec.json.decode(raw_payload, type=dict[str, dict[str, object]])
        except msgspec.DecodeError as exc:
            self._handle_load_error("Invalid cache payload", exc)
            return

        entries: dict[str, CacheEntry] = {}
        for key, value in legacy_entries.items():
            if not isinstance(value, dict):
                self._handle_load_error("Legacy cache entry must be a mapping")
                return
            mtime = value.get("mtime")
            config_hash = value.get("config_hash")
            if not isinstance(mtime, (int, float)) or not isinstance(config_hash, str):
                self._handle_load_error("Legacy cache entry schema mismatch")
                return
            entries[key] = CacheEntry(mtime=float(mtime), config_hash=config_hash)

        self._document.entries = entries
        self._document.generatedAt = datetime.now(tz=UTC).isoformat()
        validate_tools_payload(msgspec.to_builtins(self._document), DOCSTRING_CACHE_SCHEMA)


__all__ = ["BuilderCache", "CacheEntry"]
