"""Safe serialization helpers with schema validation and checksums.

This module provides JSON serialization with schema validation and SHA256
checksums to replace unsafe `pickle` usage. All serialized data is validated
against JSON Schema 2020-12 before persistence.

Examples
--------
>>> from pathlib import Path
>>> from kgfoundry_common.serialization import serialize_json, deserialize_json
>>> data = {"k1": 0.9, "b": 0.4, "N": 100}
>>> schema_path = Path("schema/bm25_metadata.v1.json")
>>> output_path = Path("/tmp/index.json")
>>> checksum = serialize_json(data, schema_path, output_path)
>>> loaded = deserialize_json(output_path, schema_path)
>>> assert loaded == data
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema
from jsonschema.exceptions import SchemaError, ValidationError

from kgfoundry_common.errors import (
    DeserializationError,
    SchemaValidationError,
    SerializationError,
)
from kgfoundry_common.fs import atomic_write, read_text, write_text
from kgfoundry_common.logging import get_logger

if TYPE_CHECKING:
    from typing import Any
else:
    # Runtime: use object for JSON-serializable values (matches jsonschema expectations)
    Any = object  # type: ignore[assignment, misc]

# JSON Schema type (from jsonschema stubs: Mapping[str, object])
JsonSchema = Mapping[str, object]

__all__ = [
    "compute_checksum",
    "deserialize_json",
    "serialize_json",
    "validate_payload",
    "verify_checksum",
]

logger = get_logger(__name__)

# Schema cache: module-level dict mapping schema paths (as strings) to parsed schema objects
_schema_cache: dict[str, dict[str, object]] = {}


@lru_cache(maxsize=128)  # type: ignore[misc]  # lru_cache is a descriptor, handled by mypy built-in stubs
def _load_schema_cached(schema_path: Path) -> dict[str, object]:
    """Load and parse a JSON Schema file with caching.

    <!-- auto:docstring-builder v1 -->

    This function uses LRU cache to avoid repeated I/O for the same schema file.
    The cache is keyed by the resolved Path object (converted to string).

    Parameters
    ----------
    schema_path : Path
        Path to JSON Schema 2020-12 file.

    Returns
    -------
    dict[str, object]
        Parsed schema dictionary.

    Raises
    ------
    FileNotFoundError
        If schema file does not exist.
    SchemaValidationError
        If schema is invalid JSON or fails schema validation.
"""
    if not schema_path.exists():
        msg = f"Schema file not found: {schema_path}"
        raise FileNotFoundError(msg)

    # Convert Path to string for caching (Path objects are not hashable)
    schema_key = str(schema_path.resolve())

    # Check module-level cache first
    if schema_key in _schema_cache:
        return _schema_cache[schema_key]

    try:
        schema_text = read_text(schema_path)
        schema_obj: dict[str, object] = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in schema file {schema_path}: {exc}"
        raise SchemaValidationError(msg) from exc

    # Validate against JSON Schema 2020-12 meta-schema
    try:
        jsonschema.Draft202012Validator.check_schema(schema_obj)
    except SchemaError as exc:
        msg = f"Invalid JSON Schema 2020-12 in {schema_path}: {exc.message}"
        raise SchemaValidationError(msg) from exc

    # Store in module-level cache
    _schema_cache[schema_key] = schema_obj
    return schema_obj


def validate_payload(payload: Mapping[str, object], schema_path: Path) -> None:
    """Validate a payload against a JSON Schema 2020-12.

    <!-- auto:docstring-builder v1 -->

    This function loads the schema (with caching), validates the payload,
    and raises SchemaValidationError if validation fails.

    Parameters
    ----------
    payload : str | object
        Payload to validate (must be JSON-serializable).
    schema_path : Path
        Path to JSON Schema 2020-12 file.

    Raises
    ------
    FileNotFoundError
        If schema file does not exist.
    SchemaValidationError
        If payload does not match schema.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     tmp = Path(tmpdir)
    ...     schema = tmp / "schema.json"
    ...     schema.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
    ...     validate_payload({"k1": 0.9}, schema)
    ...     validate_payload({"k1": "invalid"}, schema)  # doctest: +SKIP
    SchemaValidationError: Schema validation failed
"""
    schema_obj = _load_schema_cached(schema_path)
    try:
        jsonschema.validate(instance=payload, schema=schema_obj)
    except ValidationError as exc:
        msg = f"Schema validation failed: {exc.message}"
        raise SchemaValidationError(msg) from exc
    except SchemaError as exc:
        msg = f"Invalid schema: {exc.message}"
        raise SchemaValidationError(msg) from exc


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of binary data.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    data : bytes
        Binary data to checksum.

    Returns
    -------
    str
        Hexadecimal SHA256 checksum (64 characters).

    Examples
    --------
    >>> checksum = compute_checksum(b"test data")
    >>> len(checksum) == 64
    True
    >>> checksum.startswith("9")
    True
"""
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> None:
    """Verify SHA256 checksum matches expected value.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    data : bytes
        Binary data to verify.
    expected : str
        Expected hexadecimal SHA256 checksum.

    Raises
    ------
    SerializationError
        If checksum does not match.

    Examples
    --------
    >>> verify_checksum(b"test", compute_checksum(b"test"))
    >>> verify_checksum(b"wrong", compute_checksum(b"test"))  # doctest: +SKIP
    SerializationError: Checksum mismatch
"""
    actual = compute_checksum(data)
    if actual != expected:
        msg = f"Checksum mismatch: expected {expected[:16]}..., got {actual[:16]}..."
        raise SerializationError(msg)


def serialize_json(
    obj: object,  # JSON-serializable: dict, list, str, int, float, bool, None
    schema_path: Path,
    output_path: Path,
    *,
    include_checksum: bool = True,
    indent: int | None = 2,
) -> str:
    """Serialize object to JSON with schema validation and optional checksum.

    <!-- auto:docstring-builder v1 -->

    The serialized JSON is validated against the provided JSON Schema 2020-12
    before writing. If `include_checksum=True`, a checksum file is written
    alongside the JSON file.

    Parameters
    ----------
    obj : object
        Python object to serialize (must be JSON-serializable).
    schema_path : Path
        Path to JSON Schema 2020-12 file for validation.
    output_path : Path
        Output file path for JSON data.
    include_checksum : bool, optional
        If True, write a `.sha256` checksum file. Defaults to True.
        Defaults to ``True``.
    indent : int | NoneType, optional
        JSON indentation (None for compact). Defaults to 2.
        Defaults to ``2``.

    Returns
    -------
    str
        SHA256 checksum of the serialized JSON (hexadecimal, 64 characters).

    Raises
    ------
    SerializationError
        If serialization fails or file write fails.
    SchemaValidationError
        If schema validation fails.
    FileNotFoundError
        If schema file does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     tmp = Path(tmpdir)
    ...     schema = tmp / "schema.json"
    ...     schema.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
    ...     data = {"k1": 0.9}
    ...     output = tmp / "data.json"
    ...     checksum = serialize_json(data, schema, output)
    ...     assert len(checksum) == 64
    ...     assert output.exists()
"""
    try:
        # Validate payload against schema (uses cached schema loader)
        if isinstance(obj, Mapping):
            validate_payload(obj, schema_path)
        else:
            # For non-Mapping objects, load schema and validate directly
            schema_obj = _load_schema_cached(schema_path)
            try:
                jsonschema.validate(instance=obj, schema=schema_obj)
            except ValidationError as exc:
                msg = f"Schema validation failed: {exc.message}"
                raise SchemaValidationError(msg) from exc

        # Serialize object
        try:
            json_text: str = json.dumps(obj, indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            msg = f"Failed to serialize object to JSON: {exc}"
            raise SerializationError(msg) from exc

        json_bytes = json_text.encode("utf-8")

        # Compute checksum
        checksum = compute_checksum(json_bytes)

        # Write JSON atomically
        atomic_write(output_path, json_text, mode="text")

        # Write checksum file if requested
        if include_checksum:
            checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
            write_text(checksum_path, checksum)

        logger.debug(
            "Serialized JSON",
            extra={
                "output_path": str(output_path),
                "schema_path": str(schema_path),
                "checksum": checksum,
            },
        )
        return checksum  # noqa: TRY300
    except FileNotFoundError:
        # Preserve FileNotFoundError for missing schema files
        raise
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Serialization failed: {exc}"
        raise SerializationError(msg) from exc


def deserialize_json(  # noqa: C901, PLR0912
    data_path: Path,
    schema_path: Path,
    *,
    verify_checksum_file: bool = True,
) -> object:  # JSON types: dict, list, str, int, float, bool, None
    """Deserialize JSON with schema validation and checksum verification.

    <!-- auto:docstring-builder v1 -->

    The JSON is validated against the provided schema and (optionally) verified
    against a `.sha256` checksum file before parsing.

    Parameters
    ----------
    data_path : Path
        Path to JSON file to deserialize.
    schema_path : Path
        Path to JSON Schema 2020-12 file for validation.
    verify_checksum_file : bool, optional
        If True, verify against `.sha256` checksum file. Defaults to True.
        Defaults to ``True``.

    Returns
    -------
    object
        Deserialized Python object (JSON-serializable types).

    Raises
    ------
    DeserializationError
        If deserialization fails or checksum mismatch.
    SchemaValidationError
        If schema validation fails.
    FileNotFoundError
        If data or schema file does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     tmp = Path(tmpdir)
    ...     schema = tmp / "schema.json"
    ...     schema.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
    ...     data_path = tmp / "data.json"
    ...     data_path.write_text('{"k1": 0.9}')
    ...     loaded = deserialize_json(data_path, schema)
    ...     assert loaded == {"k1": 0.9}
"""
    try:
        # Verify checksum if requested
        if verify_checksum_file:
            checksum_path = data_path.with_suffix(data_path.suffix + ".sha256")
            if checksum_path.exists():
                expected_checksum = read_text(checksum_path).strip()
                data_bytes = data_path.read_bytes()
                try:
                    verify_checksum(data_bytes, expected_checksum)
                except SerializationError as exc:
                    # verify_checksum raises SerializationError, wrap in DeserializationError
                    msg = f"Checksum verification failed for {data_path}"
                    raise DeserializationError(msg) from exc
            else:
                logger.warning(
                    "Checksum file not found, skipping verification",
                    extra={"data_path": str(data_path), "checksum_path": str(checksum_path)},
                )

        # Read and parse JSON
        if not data_path.exists():
            msg = f"Data file not found: {data_path}"
            raise FileNotFoundError(msg)  # noqa: TRY301
        json_text = read_text(data_path)
        try:
            obj: object = json.loads(json_text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {data_path}: {exc}"
            raise DeserializationError(msg) from exc

        # Validate against schema (uses cached schema loader)
        if isinstance(obj, Mapping):
            validate_payload(obj, schema_path)
        else:
            # For non-Mapping objects, load schema and validate directly
            schema_obj = _load_schema_cached(schema_path)
            try:
                jsonschema.validate(instance=obj, schema=schema_obj)
            except ValidationError as exc:
                msg = f"Schema validation failed: {exc.message}"
                raise SchemaValidationError(msg) from exc
            except SchemaError as exc:
                msg = f"Invalid schema: {exc.message}"
                raise SchemaValidationError(msg) from exc

        logger.debug(
            "Deserialized JSON",
            extra={
                "data_path": str(data_path),
                "schema_path": str(schema_path),
            },
        )
    except FileNotFoundError:
        # Preserve FileNotFoundError for missing files
        raise
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Deserialization failed: {exc}"
        raise DeserializationError(msg) from exc
    else:
        return obj
