# ruff: noqa: N815
"""File-based cache tracking processed files for docstring builder."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from tools import validate_tools_payload

logger = logging.getLogger(__name__)

DOCSTRING_CACHE_SCHEMA = "docstring_cache.json"
DOCSTRING_CACHE_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/docstring-cache.json"
DOCSTRING_CACHE_VERSION = "1.0.0"


@dataclass(slots=True)
class CacheEntry:
    """Entry describing a processed file."""

    mtime: float
    config_hash: str


@dataclass(slots=True)
class CacheDocument:
    """Persisted cache document with schema metadata."""

    schemaVersion: str = DOCSTRING_CACHE_VERSION
    schemaId: str = DOCSTRING_CACHE_SCHEMA_ID
    generatedAt: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    entries: dict[str, CacheEntry] = field(default_factory=dict)


class CacheEntryPayload(TypedDict):
    mtime: float
    config_hash: str


class CacheDocumentPayload(TypedDict):
    schemaVersion: str
    schemaId: str
    generatedAt: str
    entries: dict[str, CacheEntryPayload]


class BuilderCache:
    """Persist and query cache entries keyed by file path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._document = CacheDocument()
        self._lock = threading.Lock()
        if not path.exists():
            return

        raw_payload = path.read_text(encoding="utf-8")
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            self._handle_load_error("Invalid cache payload", exc)
            return

        if not isinstance(decoded, dict) or "schemaVersion" not in decoded:
            self._load_legacy_payload(raw_payload)
            return

        payload = cast(CacheDocumentPayload, decoded)
        try:
            validate_tools_payload(_payload_mapping(payload), DOCSTRING_CACHE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            self._handle_load_error("Cache schema validation failed", exc)
            return

        try:
            self._document = _payload_to_document(payload)
        except (KeyError, TypeError, ValueError) as exc:
            self._handle_load_error("Cache entry schema mismatch", exc)

    def needs_update(self, file_path: Path, config_hash: str) -> bool:
        """Determine whether a file requires regeneration."""
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
            payload = _document_to_payload(self._document)
        validate_tools_payload(_payload_mapping(payload), DOCSTRING_CACHE_SCHEMA)
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

    def _load_legacy_payload(self, raw_payload: str) -> None:
        """Load and normalise legacy cache payloads without schema metadata."""
        try:
            legacy_entries = cast(dict[str, dict[str, object]], json.loads(raw_payload))
        except json.JSONDecodeError as exc:
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

        payload: CacheDocumentPayload = {
            "schemaVersion": DOCSTRING_CACHE_VERSION,
            "schemaId": DOCSTRING_CACHE_SCHEMA_ID,
            "generatedAt": datetime.now(tz=UTC).isoformat(),
            "entries": {
                key: {"mtime": entry.mtime, "config_hash": entry.config_hash}
                for key, entry in entries.items()
            },
        }
        try:
            validate_tools_payload(_payload_mapping(payload), DOCSTRING_CACHE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            self._handle_load_error("Legacy cache schema validation failed", exc)
            return

        self._document = _payload_to_document(payload)


def _payload_to_document(payload: CacheDocumentPayload) -> CacheDocument:
    entries = {
        key: CacheEntry(mtime=value["mtime"], config_hash=value["config_hash"])
        for key, value in payload["entries"].items()
    }
    return CacheDocument(
        schemaVersion=payload["schemaVersion"],
        schemaId=payload["schemaId"],
        generatedAt=payload["generatedAt"],
        entries=entries,
    )


def _document_to_payload(document: CacheDocument) -> CacheDocumentPayload:
    return {
        "schemaVersion": document.schemaVersion,
        "schemaId": document.schemaId,
        "generatedAt": document.generatedAt,
        "entries": {
            key: {"mtime": entry.mtime, "config_hash": entry.config_hash}
            for key, entry in document.entries.items()
        },
    }


def _payload_mapping(payload: CacheDocumentPayload) -> Mapping[str, object]:
    return cast(Mapping[str, object], payload)


__all__ = ["BuilderCache", "CacheDocument", "CacheEntry"]
