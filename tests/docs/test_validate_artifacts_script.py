"""Regression tests for :mod:`docs._scripts.validate_artifacts`."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

validate_artifacts = import_module("docs._scripts.validate_artifacts")

if TYPE_CHECKING:
    from collections.abc import Callable

    from docs._scripts.validate_artifacts import (
        ArtifactValidationError as ArtifactValidationErrorType,
    )
    from docs.types.artifacts import SymbolDeltaPayload, SymbolIndexArtifacts

    ValidateSymbolIndexFn = Callable[[Path], SymbolIndexArtifacts]
    ValidateSymbolDeltaFn = Callable[[Path], SymbolDeltaPayload]

validate_symbol_index = cast(
    "ValidateSymbolIndexFn",
    validate_artifacts.validate_symbol_index,
)
validate_symbol_delta = cast(
    "ValidateSymbolDeltaFn",
    validate_artifacts.validate_symbol_delta,
)
ArtifactValidationError = cast(
    "type[ArtifactValidationErrorType]",
    validate_artifacts.ArtifactValidationError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYMBOL_EXAMPLE = REPO_ROOT / "schema" / "examples" / "docs" / "symbol-index.sample.json"
DELTA_EXAMPLE = REPO_ROOT / "schema" / "examples" / "docs" / "symbol-delta.sample.json"


def test_validate_symbol_index_resolves_schema_from_repo(tmp_path: Path) -> None:
    """The validator should locate schemas relative to the repository root."""
    destination = tmp_path / "symbols.json"
    destination.write_text(SYMBOL_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts: SymbolIndexArtifacts = validate_symbol_index(destination)

    assert len(artifacts.rows) > 0

    invalid_payload = cast(
        "list[dict[str, object]]",
        json.loads(SYMBOL_EXAMPLE.read_text(encoding="utf-8")),
    )
    assert isinstance(invalid_payload, list)
    if invalid_payload:
        invalid_payload[0].pop("path", None)

    broken_destination = tmp_path / "symbols-invalid.json"
    broken_destination.write_text(
        json.dumps(invalid_payload),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError) as excinfo:
        validate_symbol_index(broken_destination)

    assert isinstance(excinfo.value, ArtifactValidationError)
    assert excinfo.value.artifact_name == "symbols.json"


def test_validate_symbol_delta_resolves_schema_from_repo(tmp_path: Path) -> None:
    """The delta validator should locate schemas relative to the repository root."""
    destination = tmp_path / "symbols.delta.json"
    destination.write_text(DELTA_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    payload: SymbolDeltaPayload = validate_symbol_delta(destination)

    assert payload.added

    invalid_payload = cast(
        "dict[str, object]",
        json.loads(DELTA_EXAMPLE.read_text(encoding="utf-8")),
    )
    invalid_payload["removed"] = "not-a-list"

    broken_destination = tmp_path / "symbols.delta.invalid.json"
    broken_destination.write_text(
        json.dumps(invalid_payload),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError) as excinfo:
        validate_symbol_delta(broken_destination)

    assert isinstance(excinfo.value, ArtifactValidationError)
    assert excinfo.value.artifact_name == "symbols.delta.json"
