"""Tests for documentation artifact schema validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from docs._scripts.validation import validate_against_schema  # noqa: PLC2701
from tools import ToolExecutionError

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonPayload = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue

REPO_ROOT = Path(__file__).resolve().parents[2]
SYMBOL_SCHEMA = REPO_ROOT / "schema/docs/symbol-index.schema.json"
DELTA_SCHEMA = REPO_ROOT / "schema/docs/symbol-delta.schema.json"
SYMBOL_EXAMPLE = REPO_ROOT / "schema/examples/docs/symbol-index.sample.json"
DELTA_EXAMPLE = REPO_ROOT / "schema/examples/docs/symbol-delta.sample.json"


def _load(path: Path) -> JsonPayload:
    return cast(JsonPayload, json.loads(path.read_text(encoding="utf-8")))


def test_symbol_index_sample_validates() -> None:
    payload = _load(SYMBOL_EXAMPLE)
    validate_against_schema(payload, SYMBOL_SCHEMA, artifact="symbols.json")


def test_symbol_index_invalid_path_raises_problem() -> None:
    payload = _load(SYMBOL_EXAMPLE)
    assert isinstance(payload, list)
    payload_list = cast(list[Mapping[str, JsonValue]], payload)
    broken = dict(payload_list[0])
    broken.pop("path", None)
    with pytest.raises(ToolExecutionError):
        validate_against_schema([broken], SYMBOL_SCHEMA, artifact="symbols.json")


def test_symbol_delta_sample_validates() -> None:
    payload = _load(DELTA_EXAMPLE)
    validate_against_schema(payload, DELTA_SCHEMA, artifact="symbols.delta.json")


def test_symbol_delta_rejects_non_object_payload() -> None:
    with pytest.raises(ToolExecutionError):
        validate_against_schema(
            ["not", "an", "object"], DELTA_SCHEMA, artifact="symbols.delta.json"
        )
