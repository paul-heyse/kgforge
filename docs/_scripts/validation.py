"""Schema validation helpers for documentation tooling artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import msgspec
from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from tools import ToolExecutionError
from tools._shared.problem_details import build_schema_problem_details
from tools._shared.schema import validate_struct_payload

JsonPayload = Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None

_VALIDATOR_CACHE: dict[Path, Draft202012Validator] = {}


def _get_validator(schema_path: Path) -> Draft202012Validator:
    resolved = schema_path.resolve()
    cached = _VALIDATOR_CACHE.get(resolved)
    if cached is not None:
        return cached

    schema_text = resolved.read_text(encoding="utf-8")
    schema_data = json.loads(schema_text)
    Draft202012Validator.check_schema(schema_data)
    validator = Draft202012Validator(schema_data)
    _VALIDATOR_CACHE[resolved] = validator
    return validator


def _problem_for_validation(
    *,
    artifact: str,
    schema_path: Path,
    error: jsonschema_exceptions.ValidationError,
) -> dict[str, Any]:
    return build_schema_problem_details(
        error=error,
        type="https://kgfoundry.dev/problems/docs-schema-validation",
        title="Docs artifact failed schema validation",
        status=422,
        instance=f"urn:docs:{artifact}:schema-validation",
        extensions={
            "artifact": artifact,
            "schema": str(schema_path),
        },
    )


def _coerce_sequence(value: JsonPayload) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def validate_against_schema(
    payload: JsonPayload,
    schema_path: Path,
    *,
    artifact: str,
    struct_type: type[msgspec.Struct] | None = None,
) -> None:
    """Validate a JSON-compatible ``payload`` against ``schema_path``.

    Parameters
    ----------
    payload : JsonPayload
        JSON-compatible payload to validate.
    schema_path : Path
        Absolute path to the JSON Schema file.
    artifact : str
        Artifact identifier used in problem details when validation fails.
    struct_type : type[Any] | None
        Optional msgspec ``Struct`` used for additional structural validation.
    """
    validator = _get_validator(schema_path)

    try:
        validator.validate(payload)
    except jsonschema_exceptions.ValidationError as exc:  # pragma: no cover - exercised in CLI
        problem = _problem_for_validation(artifact=artifact, schema_path=schema_path, error=exc)
        raise ToolExecutionError(
            f"{artifact} failed schema validation",
            command=["docs-schema-validate", artifact],
            problem=problem,
        ) from exc

    if struct_type is not None:
        if isinstance(payload, Mapping):
            validate_struct_payload(payload, struct_type)
        else:
            sequence = _coerce_sequence(payload)
            if sequence is not None:
                for entry in sequence:
                    if isinstance(entry, Mapping):
                        validate_struct_payload(entry, struct_type)
