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
import logging
from pathlib import Path
from typing import Any

import jsonschema

from kgfoundry_common.errors import DeserializationError, SerializationError
from kgfoundry_common.fs import atomic_write, read_text, write_text

__all__ = [
    "compute_checksum",
    "deserialize_json",
    "serialize_json",
    "verify_checksum",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of binary data.

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
    obj: Any,  # noqa: ANN401  # JSON-serializable: dict, list, str, int, float, bool, None (Any required for JSON)
    schema_path: Path,
    output_path: Path,
    *,
    include_checksum: bool = True,
    indent: int | None = 2,
) -> str:
    """Serialize object to JSON with schema validation and optional checksum.

    The serialized JSON is validated against the provided JSON Schema 2020-12
    before writing. If `include_checksum=True`, a checksum file is written
    alongside the JSON file.

    Parameters
    ----------
    obj : Any
        Python object to serialize (must be JSON-serializable).
    schema_path : Path
        Path to JSON Schema 2020-12 file for validation.
    output_path : Path
        Output file path for JSON data.
    include_checksum : bool, optional
        If True, write a `.sha256` checksum file. Defaults to True.
    indent : int | None, optional
        JSON indentation (None for compact). Defaults to 2.

    Returns
    -------
    str
        SHA256 checksum of the serialized JSON (hexadecimal, 64 characters).

    Raises
    ------
    SerializationError
        If serialization fails, schema validation fails, or file write fails.
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
        # Load and validate schema
        if not schema_path.exists():
            msg = f"Schema file not found: {schema_path}"
            raise FileNotFoundError(msg)  # noqa: TRY301
        schema_text = read_text(schema_path)
        schema_obj = json.loads(schema_text)

        # Serialize object
        try:
            json_text = json.dumps(obj, indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            msg = f"Failed to serialize object to JSON: {exc}"
            raise SerializationError(msg) from exc

        json_bytes = json_text.encode("utf-8")

        # Validate against schema
        try:
            jsonschema.validate(instance=obj, schema=schema_obj)
        except jsonschema.ValidationError as exc:
            msg = f"Schema validation failed: {exc.message}"
            raise SerializationError(msg) from exc
        except jsonschema.SchemaError as exc:
            msg = f"Invalid schema: {exc.message}"
            raise SerializationError(msg) from exc

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


def deserialize_json(  # noqa: C901
    data_path: Path,
    schema_path: Path,
    *,
    verify_checksum_file: bool = True,
) -> Any:  # noqa: ANN401  # JSON types: dict, list, str, int, float, bool, None (Any required for JSON)
    """Deserialize JSON with schema validation and checksum verification.

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

    Returns
    -------
    Any
        Deserialized Python object.

    Raises
    ------
    SerializationError
        If deserialization fails, schema validation fails, or checksum mismatch.
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

        # Load schema
        if not schema_path.exists():
            msg = f"Schema file not found: {schema_path}"
            raise FileNotFoundError(msg)  # noqa: TRY301
        schema_text = read_text(schema_path)
        try:
            schema_obj = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON schema in {schema_path}: {exc}"
            raise DeserializationError(msg) from exc

        # Read and parse JSON
        if not data_path.exists():
            msg = f"Data file not found: {data_path}"
            raise FileNotFoundError(msg)  # noqa: TRY301
        json_text = read_text(data_path)
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {data_path}: {exc}"
            raise DeserializationError(msg) from exc

        # Validate against schema
        try:
            jsonschema.validate(instance=obj, schema=schema_obj)
        except jsonschema.ValidationError as exc:
            msg = f"Schema validation failed: {exc.message}"
            raise DeserializationError(msg) from exc
        except jsonschema.SchemaError as exc:
            msg = f"Invalid schema: {exc.message}"
            raise DeserializationError(msg) from exc

        logger.debug(
            "Deserialized JSON",
            extra={
                "data_path": str(data_path),
                "schema_path": str(schema_path),
            },
        )
        return obj  # noqa: TRY300

    except FileNotFoundError:
        # Preserve FileNotFoundError for missing files
        raise
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Deserialization failed: {exc}"
        raise DeserializationError(msg) from exc
