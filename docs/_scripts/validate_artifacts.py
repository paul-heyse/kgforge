"""Command-line interface for validating documentation artifacts against schemas.

This module provides a CLI tool to validate all documentation artifacts (symbol
index, delta, reverse lookups) against their canonical JSON Schema definitions.
It ensures that all artifacts conform to the spec before they are written to disk.

The validator orchestrates calls to specialized functions for each artifact type
(see `validate_symbol_index`, `validate_symbol_delta`) and integrates with the
observability pipeline via:

- **Structured Logging**: each artifact validation produces logs with operation,
  artifact, and status fields via `shared.make_logger()` for correlation and tracing
- **Metrics**: duration and status captured via `observe_tool_run()` context manager
- **Error Handling**: all validation failures raise `ArtifactValidationError` with
  RFC 9457 Problem Details attached (`.problem` attr)

Correlation IDs
---------------
When invoked as part of a larger build orchestration, context propagates via
`contextvars` (see `tools._shared.contextvars` for details). Logs automatically
inherit correlation metadata from the calling context.

Examples
--------
Typical CLI usage::

    $ python -m docs._scripts.validate_artifacts --artifacts symbols.json
    ✓ symbols.json validated successfully

Programmatic usage::

    >>> from docs._scripts.validate_artifacts import validate_symbol_index
    >>> from pathlib import Path
    >>> artifacts = validate_symbol_index(Path("docs/_build/symbols.json"))
    >>> print(f"Validated {len(artifacts.rows)} symbols")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tools import (
    get_logger,
    observe_tool_run,
)
from tools._shared.proc import ToolExecutionError

from docs._scripts import shared
from docs._scripts.validation import validate_against_schema
from docs.types.artifacts import (
    symbol_delta_from_json,
    symbol_delta_to_payload,
    symbol_index_from_json,
    symbol_index_to_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tools._shared.problem_details import ProblemDetailsDict

    from docs.types.artifacts import (
        JsonPayload,
        SymbolDeltaPayload,
        SymbolIndexArtifacts,
    )

type ReverseLookup = dict[str, tuple[str, ...]]
type ReverseLookupPayload = dict[str, list[str]]

ENV = shared.detect_environment()
shared.ensure_sys_paths(ENV)
SETTINGS = shared.load_settings()
SCHEMA_ROOT = ENV.root / "schema" / "docs"

BASE_LOGGER = get_logger(__name__)
VALIDATION_LOG = shared.make_logger(
    "docs_artifact_validation",
    artifact="artifacts",
    logger=BASE_LOGGER,
)


class ArtifactValidationError(RuntimeError):
    """Exception raised when an artifact fails validation.

    This exception wraps validation failures with RFC 9457 Problem Details,
    providing structured error information suitable for CLI output and logging.

    Attributes
    ----------
    artifact_name : str
        Logical identifier for the artifact (e.g., "symbols.json").
    problem : ProblemDetailsDict
        RFC 9457 Problem Details dict with validation error details.
    """

    def __init__(
        self,
        message: str,
        artifact_name: str,
        problem: ProblemDetailsDict | None = None,
    ) -> None:
        """Initialize ArtifactValidationError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        artifact_name : str
            Logical identifier for the artifact.
        problem : ProblemDetailsDict | None, optional
            RFC 9457 Problem Details dict. Defaults to None.
        """
        super().__init__(message)
        self.artifact_name = artifact_name
        self.problem = problem or {}


@dataclass(slots=True, frozen=True)
class ArtifactCheck:
    """Configuration for validating a single artifact type.

    Attributes
    ----------
    name : str
        Human-readable name (e.g., "Symbol Index").
    artifact_id : str
        Logical identifier (e.g., "symbols.json").
    path : Path
        Path to the artifact file.
    schema : Path
        Path to the JSON Schema file.
    loader : Callable[[Path], object]
        Function to load and parse the artifact file.
    codec : Callable[[object], object]
        Function to convert loaded data to typed model.
    """

    name: str
    artifact_id: str
    path: Path
    schema: Path
    loader: Callable[[Path], object]
    codec: Callable[[object], object]


def _resolve_schema(name: str) -> Path:
    """Return the absolute path to a documentation schema file."""
    return SCHEMA_ROOT / name
def _schema_path(filename: str) -> Path:
    """Return the absolute path to a docs schema file."""
    return ENV.root / "schema" / "docs" / filename


def validate_symbol_index(path: Path) -> SymbolIndexArtifacts:
    """Validate a symbol index JSON file against its schema.

    Parameters
    ----------
    path : Path
        Path to the symbols.json file.

    Returns
    -------
    SymbolIndexArtifacts
        The validated artifact model.

    Raises
    ------
    ArtifactValidationError
        If the file doesn't exist, is invalid JSON, or fails schema validation.
    """
    if not path.exists():
        message = f"Symbol index not found: {path}"
        raise ArtifactValidationError(message, artifact_name="symbols.json")

    try:
        raw_data: JsonPayload = cast("JsonPayload", json.loads(path.read_text(encoding="utf-8")))
        artifacts = symbol_index_from_json(raw_data)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        message = f"Failed to parse symbol index: {e}"
        raise ArtifactValidationError(message, artifact_name="symbols.json") from e

    schema = _resolve_schema("symbol-index.schema.json")
    schema = _schema_path("symbol-index.schema.json")
    try:
        validate_against_schema(symbol_index_to_payload(artifacts), schema, artifact="symbols.json")
    except ToolExecutionError as e:
        raise ArtifactValidationError(
            str(e),
            artifact_name="symbols.json",
            problem=e.problem,
        ) from e

    return artifacts


def validate_symbol_delta(path: Path) -> SymbolDeltaPayload:
    """Validate a symbol delta JSON file against its schema.

    Parameters
    ----------
    path : Path
        Path to the symbols.delta.json file.

    Returns
    -------
    SymbolDeltaPayload
        The validated artifact model.

    Raises
    ------
    ArtifactValidationError
        If the file doesn't exist, is invalid JSON, or fails schema validation.
    """
    if not path.exists():
        message = f"Symbol delta not found: {path}"
        raise ArtifactValidationError(message, artifact_name="symbols.delta.json")

    try:
        raw_data: JsonPayload = cast("JsonPayload", json.loads(path.read_text(encoding="utf-8")))
        payload = symbol_delta_from_json(raw_data)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        message = f"Failed to parse symbol delta: {e}"
        raise ArtifactValidationError(message, artifact_name="symbols.delta.json") from e

    schema = _resolve_schema("symbol-delta.schema.json")
    schema = _schema_path("symbol-delta.schema.json")
    try:
        validate_against_schema(
            symbol_delta_to_payload(payload), schema, artifact="symbols.delta.json"
        )
    except ToolExecutionError as e:
        raise ArtifactValidationError(
            str(e),
            artifact_name="symbols.delta.json",
            problem=e.problem,
        ) from e

    return payload


def _load_reverse_lookup(
    path: Path,
    *,
    artifact_name: str,
    schema_filename: str,
) -> ReverseLookup:
    """Load and validate a reverse lookup artifact."""
    if not path.exists():
        message = f"Reverse lookup not found: {path}"
        raise ArtifactValidationError(message, artifact_name=artifact_name)

    try:
        raw_data: object = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        message = f"Failed to parse {artifact_name}: {e}"
        raise ArtifactValidationError(message, artifact_name=artifact_name) from e

    if not isinstance(raw_data, dict):
        message = f"{artifact_name} must be a JSON object mapping strings to arrays"
        raise ArtifactValidationError(message, artifact_name=artifact_name)

    payload: ReverseLookupPayload = {}
    typed_lookup: ReverseLookup = {}
    for key, value in raw_data.items():
        if not isinstance(key, str) or not key:
            message = f"Invalid key in {artifact_name}: {key!r}"
            raise ArtifactValidationError(message, artifact_name=artifact_name)
        if not isinstance(value, list):
            message = f"{artifact_name} values must be arrays of strings"
            raise ArtifactValidationError(message, artifact_name=artifact_name)

        values: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item:
                message = f"{artifact_name} entries must be non-empty strings (key={key!r}, index={index})"
                raise ArtifactValidationError(message, artifact_name=artifact_name)
            values.append(item)

        payload[key] = values
        typed_lookup[key] = tuple(values)

    schema = _schema_path(schema_filename)
    try:
        validate_against_schema(payload, schema, artifact=artifact_name)
    except ToolExecutionError as e:
        raise ArtifactValidationError(
            str(e),
            artifact_name=artifact_name,
            problem=e.problem,
        ) from e

    return typed_lookup


def validate_by_file_lookup(path: Path) -> ReverseLookup:
    """Validate a by-file reverse lookup JSON file against its schema."""
    return _load_reverse_lookup(
        path,
        artifact_name="by_file.json",
        schema_filename="symbol-reverse-lookup.schema.json",
    )


def validate_by_module_lookup(path: Path) -> ReverseLookup:
    """Validate a by-module reverse lookup JSON file against its schema."""
    return _load_reverse_lookup(
        path,
        artifact_name="by_module.json",
        schema_filename="symbol-reverse-lookup.schema.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate all documentation artifacts against their canonical schemas.

    Validates `symbols.json`, `symbols.delta.json`, and reverse lookup artifacts
    (`by_file.json`, `by_module.json`) against their JSON Schema definitions.
    Emits RFC 9457 Problem Details on validation failure and logs structured
    metadata including operation, artifact, and status for observability.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments. Defaults to None (uses sys.argv).

    Returns
    -------
    int
        Exit code: 0 for success, non-zero for failure.

    Examples
    --------
    >>> main(["--artifacts", "symbols.json"])
    0
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        type=str,
        default=None,
        help="Artifact names to validate (default: all)",
    )
    args = parser.parse_args(argv)
    artifact_names_from_args: list[str] | None = cast("list[str] | None", args.artifacts)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    artifacts_to_check: dict[
        str,
        Callable[[Path], SymbolIndexArtifacts | SymbolDeltaPayload | ReverseLookup],
    ] = {
        "symbols.json": validate_symbol_index,
        "symbols.delta.json": validate_symbol_delta,
        "by_file.json": validate_by_file_lookup,
        "by_module.json": validate_by_module_lookup,
    }

    with observe_tool_run(
        ["docs-validate-artifacts"],
        cwd=SETTINGS.docs_build_dir,
        timeout=None,
    ) as observation:
        failed_count = 0
        artifact_names: list[str] = (
            artifact_names_from_args
            if artifact_names_from_args is not None
            else list(artifacts_to_check.keys())
        )

        for artifact_name in artifact_names:
            if artifact_name not in artifacts_to_check:
                VALIDATION_LOG.warning(
                    "Unknown artifact: %s",
                    artifact_name,
                    extra={"artifact": artifact_name, "status": "skipped"},
                )
                continue

            artifact_log = shared.make_logger(
                "docs_artifact_validation",
                artifact=artifact_name,
                logger=BASE_LOGGER,
            )

            try:
                validator = artifacts_to_check[artifact_name]
                validator(Path(artifact_name))
                artifact_log.info(
                    "Artifact validated successfully",
                    extra={"status": "success"},
                )
            except ArtifactValidationError as e:
                artifact_log.exception(
                    "Artifact validation failed",
                    extra={
                        "status": "failure",
                        "artifact": artifact_name,
                        "error_type": type(e).__name__,
                    },
                )
                failed_count += 1

        if failed_count > 0:
            VALIDATION_LOG.error(
                "Artifact validation failed for %s artifact(s)",
                failed_count,
                extra={
                    "failed_count": failed_count,
                    "status": "failure",
                },
            )
            observation.failure("validation_failed", returncode=1)
            return 1

        VALIDATION_LOG.info(
            "All artifacts validated successfully",
            extra={
                "artifact_count": len(artifact_names),
                "status": "success",
            },
        )
        observation.success(0)
        return 0


if __name__ == "__main__":
    sys.exit(main())
