"""Validate documentation build artifacts against JSON Schemas."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from docs._scripts import shared  # noqa: PLC2701
from docs._scripts.validation import validate_against_schema  # noqa: PLC2701
from tools import (
    build_problem_details,
    get_logger,
    observe_tool_run,
    render_problem,
)
from tools._shared.proc import ToolExecutionError

ENV = shared.detect_environment()
shared.ensure_sys_paths(ENV)
SETTINGS = shared.load_settings()

DOCS_BUILD = SETTINGS.docs_build_dir
SCHEMA_DIR = ENV.root / "schema" / "docs"
SYMBOLS_PATH = DOCS_BUILD / "symbols.json"
DELTA_PATH = DOCS_BUILD / "symbols.delta.json"
BY_FILE_PATH = DOCS_BUILD / "by_file.json"
BY_MODULE_PATH = DOCS_BUILD / "by_module.json"
SYMBOL_SCHEMA = SCHEMA_DIR / "symbol-index.schema.json"
DELTA_SCHEMA = SCHEMA_DIR / "symbol-delta.schema.json"

BASE_LOGGER = get_logger(__name__)
VALIDATION_LOG = shared.make_logger(
    "docs_artifact_validation", artifact=str(DOCS_BUILD), logger=BASE_LOGGER
)


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
ProblemDetailsDict = dict[str, JsonValue]
JsonPayload = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue


def _emit_problem(problem: ProblemDetailsDict | None, *, default_message: str) -> None:
    payload = problem or build_problem_details(
        type="https://kgfoundry.dev/problems/docs-artifact-validation",
        title="Documentation artifact validation failed",
        status=500,
        detail=default_message,
        instance="urn:docs:artifact-validation:unexpected-error",
    )
    sys.stderr.write(render_problem(payload) + "\n")


def _load_json(path: Path) -> JsonPayload:
    try:
        return cast(JsonPayload, json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/docs-artifact-validation",
            title="Documentation artifact missing",
            status=404,
            detail=f"Required artifact '{path}' is missing",
            instance=f"urn:docs:artifact-validation:missing:{path.name}",
        )
        message = f"Artifact '{path}' is missing"
        raise ToolExecutionError(
            message,
            command=("docs-validate-artifacts", str(path)),
            problem=problem,
        ) from exc


def _validate_reverse_lookup(payload: object, artifact: str) -> None:
    if not isinstance(payload, Mapping):
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/docs-artifact-validation",
            title="Reverse lookup has invalid structure",
            status=422,
            detail=f"Artifact '{artifact}' must be an object mapping file/module names to symbol paths",
            instance=f"urn:docs:artifact-validation:invalid:{artifact}",
        )
        message = f"Artifact '{artifact}' must be a mapping"
        raise ToolExecutionError(
            message,
            command=("docs-validate-artifacts", artifact),
            problem=problem,
        )
    for key, value in payload.items():
        if not isinstance(key, str):
            problem = build_problem_details(
                type="https://kgfoundry.dev/problems/docs-artifact-validation",
                title="Reverse lookup key must be a string",
                status=422,
                detail=f"Artifact '{artifact}' has a non-string key",
                instance=f"urn:docs:artifact-validation:invalid-key:{artifact}",
            )
            message = f"Artifact '{artifact}' has an invalid key"
            raise ToolExecutionError(
                message,
                command=("docs-validate-artifacts", artifact),
                problem=problem,
            )
        if not isinstance(value, Sequence) or not all(isinstance(item, str) for item in value):
            problem = build_problem_details(
                type="https://kgfoundry.dev/problems/docs-artifact-validation",
                title="Reverse lookup values must be arrays of strings",
                status=422,
                detail=f"Artifact '{artifact}' has an invalid value for '{key}'",
                instance=f"urn:docs:artifact-validation:invalid-value:{artifact}",
            )
            message = f"Artifact '{artifact}' has an invalid value"
            raise ToolExecutionError(
                message,
                command=("docs-validate-artifacts", artifact),
                problem=problem,
            )


def _validate_symbol_index(payload: object) -> None:
    validate_against_schema(cast(JsonPayload, payload), SYMBOL_SCHEMA, artifact=SYMBOLS_PATH.name)


def _validate_symbol_delta(payload: object) -> None:
    if isinstance(payload, Mapping):
        validate_against_schema(cast(JsonPayload, payload), DELTA_SCHEMA, artifact=DELTA_PATH.name)
    else:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/docs-artifact-validation",
            title="Symbol delta payload must be an object",
            status=422,
            detail="Expected an object payload for symbols.delta.json",
            instance="urn:docs:artifact-validation:invalid:symbols-delta",
        )
        message = "symbols.delta.json payload is invalid"
        raise ToolExecutionError(
            message,
            command=("docs-validate-artifacts", DELTA_PATH.name),
            problem=problem,
        )


def main() -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    with observe_tool_run(["docs-validate-artifacts"], cwd=DOCS_BUILD, timeout=None) as observation:
        try:
            symbols_payload = _load_json(SYMBOLS_PATH)
            _validate_symbol_index(symbols_payload)

            delta_payload = _load_json(DELTA_PATH)
            _validate_symbol_delta(delta_payload)

            by_file_payload = _load_json(BY_FILE_PATH)
            _validate_reverse_lookup(by_file_payload, BY_FILE_PATH.name)

            by_module_payload = _load_json(BY_MODULE_PATH)
            _validate_reverse_lookup(by_module_payload, BY_MODULE_PATH.name)
        except ToolExecutionError as exc:
            observation.failure("failure", returncode=1)
            _emit_problem(exc.problem, default_message=str(exc))
            return 1
        except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
            observation.failure("exception", returncode=1)
            problem = build_problem_details(
                type="https://kgfoundry.dev/problems/docs-artifact-validation",
                title="Documentation artifact validation failed",
                status=500,
                detail=str(exc),
                instance="urn:docs:artifact-validation:unexpected-error",
            )
            _emit_problem(problem, default_message=str(exc))
            return 1
        else:
            observation.success(0)

    VALIDATION_LOG.info(
        "Documentation artifacts validated",
        extra={
            "status": "success",
            "symbols": str(SYMBOLS_PATH),
            "delta": str(DELTA_PATH),
            "by_file": str(BY_FILE_PATH),
            "by_module": str(BY_MODULE_PATH),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
