"""File-based cache tracking processed files for docstring builder."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field

from kgfoundry_common.errors import DeserializationError, SchemaValidationError
from tools._shared.schema import validate_tools_payload

logger = logging.getLogger(__name__)

DOCSTRING_CACHE_SCHEMA: Final[str] = "docstring_cache.json"
DOCSTRING_CACHE_SCHEMA_ID: Final[str] = "https://kgfoundry.dev/schema/tools/docstring-cache.json"
DOCSTRING_CACHE_VERSION: Final[str] = "1.0.0"


def _default_generated_at() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def _default_entries() -> dict[str, CacheEntry]:
    """Return an empty entries dict for :class:`CacheDocument`."""
    return {}


class CacheEntry(BaseModel):
    """Entry describing a processed file."""

    model_config = ConfigDict(populate_by_name=True)

    mtime: float
    config_hash: str


class CacheDocument(BaseModel):
    """Persisted cache document with schema metadata."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(DOCSTRING_CACHE_VERSION, alias="schemaVersion")
    schema_id: str = Field(DOCSTRING_CACHE_SCHEMA_ID, alias="schemaId")
    generated_at: str = Field(default_factory=_default_generated_at, alias="generatedAt")
    entries: dict[str, CacheEntry] = Field(default_factory=_default_entries)


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
            decoded: object = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            self._handle_load_error("Invalid cache payload", exc)
            return

        if not isinstance(decoded, dict) or "schemaVersion" not in decoded:
            self._load_legacy_payload(raw_payload)
            return

        payload_dict = cast(dict[str, object], decoded)
        try:
            validate_tools_payload(payload_dict, DOCSTRING_CACHE_SCHEMA)
        except (FileNotFoundError, SchemaValidationError, DeserializationError) as exc:
            self._handle_load_error("Cache schema validation failed", exc)
            return

        try:
            self._document = CacheDocument.model_validate(payload_dict)
        except (ValueError, TypeError) as exc:
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
            entries = dict(self._document.entries)
            entries[key] = CacheEntry(mtime=mtime, config_hash=config_hash)
            self._document = CacheDocument(
                schema_version=self._document.schema_version,
                schema_id=self._document.schema_id,
                generated_at=datetime.now(tz=UTC).isoformat(),
                entries=entries,
            )

    def write(self) -> None:
        """Persist cache entries to disk."""
        with self._lock:
            payload = self._document.model_dump(by_alias=True, exclude_none=False)
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

    def _load_legacy_payload(self, raw_payload: str) -> None:
        """Load and normalise legacy cache payloads without schema metadata."""
        try:
            legacy_entries = cast(dict[str, dict[str, object]], json.loads(raw_payload))
        except json.JSONDecodeError as exc:
            self._handle_load_error("Invalid cache payload", exc)
            return

        entries: dict[str, CacheEntry] = {}
        for key, value in legacy_entries.items():
            mtime_val = value.get("mtime")
            config_hash_val = value.get("config_hash")
            if not isinstance(mtime_val, (int, float)) or not isinstance(config_hash_val, str):
                self._handle_load_error("Legacy cache entry schema mismatch")
                return
            entries[key] = CacheEntry(mtime=float(mtime_val), config_hash=config_hash_val)

        payload_dict: dict[str, object] = {
            "schemaVersion": DOCSTRING_CACHE_VERSION,
            "schemaId": DOCSTRING_CACHE_SCHEMA_ID,
            "generatedAt": datetime.now(tz=UTC).isoformat(),
            "entries": {
                key: {"mtime": entry.mtime, "config_hash": entry.config_hash}
                for key, entry in entries.items()
            },
        }
        try:
            validate_tools_payload(payload_dict, DOCSTRING_CACHE_SCHEMA)
        except (FileNotFoundError, SchemaValidationError, DeserializationError) as exc:
            self._handle_load_error("Legacy cache schema validation failed", exc)
            return

        try:
            self._document = CacheDocument.model_validate(payload_dict)
        except (ValueError, TypeError) as exc:
            self._handle_load_error("Legacy cache conversion failed", exc)


def from_payload(payload: dict[str, object]) -> CacheDocument:
    """Convert a payload dictionary to a :class:`CacheDocument`.

    Parameters
    ----------
    payload : dict[str, object]
        Raw payload dictionary (typically from JSON).

    Returns
    -------
    CacheDocument
        Typed document instance.

    Raises
    ------
    ValueError
        If the payload cannot be converted to the expected structure.
    """
    validate_tools_payload(payload, DOCSTRING_CACHE_SCHEMA)
    return CacheDocument.model_validate(payload)


def to_payload(document: CacheDocument) -> dict[str, object]:
    """Convert a :class:`CacheDocument` to a payload dictionary.

    Parameters
    ----------
    document : CacheDocument
        Typed document instance.

    Returns
    -------
    dict[str, object]
        Payload dictionary suitable for JSON serialization.
    """
    return cast(dict[str, object], document.model_dump(by_alias=True))


def validate_document(document: CacheDocument) -> None:
    """Validate a :class:`CacheDocument` against the schema.

    Parameters
    ----------
    document : CacheDocument
        Document to validate.

    Raises
    ------
    Exception
        If validation fails.
    """
    payload = to_payload(document)
    validate_tools_payload(payload, DOCSTRING_CACHE_SCHEMA)


__all__ = [
    "BuilderCache",
    "CacheDocument",
    "CacheEntry",
    "from_payload",
    "to_payload",
    "validate_document",
]
