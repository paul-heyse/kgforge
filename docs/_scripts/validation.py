"""Schema validation helpers for documentation tooling artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from tools._shared.problem_details import (  # noqa: PLC2701
    ProblemDetailsDict,
    build_schema_problem_details,
)
from tools._shared.proc import ToolExecutionError

JsonPayload = Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None

_VALIDATOR_CACHE: dict[Path, Draft202012Validator] = {}


def _get_validator(schema_path: Path) -> Draft202012Validator:
    resolved = schema_path.resolve()
    cached = _VALIDATOR_CACHE.get(resolved)
    if cached is not None:
        return cached

    schema_text = resolved.read_text(encoding="utf-8")
    schema_data = cast(dict[str, object], json.loads(schema_text))
    Draft202012Validator.check_schema(schema_data)
    validator = Draft202012Validator(schema_data)
    _VALIDATOR_CACHE[resolved] = validator
    return validator


def _problem_for_validation(
    *,
    artifact: str,
    schema_path: Path,
    error: jsonschema_exceptions.ValidationError,
) -> ProblemDetailsDict:
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


def validate_against_schema(
    payload: JsonPayload,
    schema_path: Path,
    *,
    artifact: str,
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
    """
    validator = _get_validator(schema_path)

    try:
        validator.validate(payload)
    except jsonschema_exceptions.ValidationError as exc:  # pragma: no cover - exercised in CLI
        problem = _problem_for_validation(artifact=artifact, schema_path=schema_path, error=exc)
        message = f"{artifact} failed schema validation"
        raise ToolExecutionError(
            message,
            command=("docs-schema-validate", artifact),
            problem=problem,
        ) from exc
