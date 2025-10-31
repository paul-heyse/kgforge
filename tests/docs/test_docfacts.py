"""Sanity checks for the generated docfacts artifact."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import ValidationError
from pytest import MonkeyPatch
from tools.docstring_builder import docfacts as docfacts_module
from tools.docstring_builder.cli import DOCFACTS_PATH
from tools.docstring_builder.docfacts import DOCFACTS_VERSION, validate_docfacts_payload

REQUIRED_FIELDS: tuple[str, ...] = (
    "module",
    "kind",
    "filepath",
    "lineno",
    "end_lineno",
    "decorators",
    "is_async",
    "is_generator",
    "owned",
    "parameters",
    "returns",
    "raises",
    "notes",
)


def _missing_docfacts_message(path: Path) -> str:
    return f"Docfacts missing at {path}"


def _invalid_payload_message(expected: str) -> str:
    return f"Docfacts payload must be a {expected}"


def _module_import_error_message(module_name: str, exc: Exception) -> str:
    return f"Module '{module_name}' from docfacts is not importable: {exc}"


def _load_docfacts_document() -> dict[str, Any]:
    if not DOCFACTS_PATH.exists():
        message = _missing_docfacts_message(DOCFACTS_PATH)
        raise FileNotFoundError(message)
    payload = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - defensive guard
        message = _invalid_payload_message("mapping")
        raise TypeError(message)
    return payload


def _load_docfacts_entries() -> list[dict[str, Any]]:
    document = _load_docfacts_document()
    entries = document.get("entries")
    if not isinstance(entries, list):  # pragma: no cover - defensive guard
        message = _invalid_payload_message("list of entries")
        raise TypeError(message)
    validated: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            validated.append(entry)
    return validated


def test_docfacts_matches_schema() -> None:
    document = _load_docfacts_document()
    try:
        validate_docfacts_payload(document)
    except ValidationError as exc:  # pragma: no cover - schema should be authoritative
        message = f"Docfacts schema validation failed: {exc}"
        raise AssertionError(message) from exc


def test_docfacts_has_version_and_provenance() -> None:
    document = _load_docfacts_document()
    assert document["docfactsVersion"] == DOCFACTS_VERSION
    provenance = document.get("provenance")
    assert isinstance(provenance, dict)
    for field in ("builderVersion", "configHash", "commitHash", "generatedAt"):
        assert provenance.get(field), f"Missing provenance field {field}"


def test_docfacts_have_unique_sorted_qnames() -> None:
    entries = _load_docfacts_entries()
    qnames = [entry["qname"] for entry in entries]
    assert qnames == sorted(qnames), "Docfacts must be sorted by qualified name"
    assert len(qnames) == len(set(qnames)), "Docfacts must not contain duplicate qnames"


def test_docfacts_required_fields_present() -> None:
    entries = _load_docfacts_entries()
    for field in REQUIRED_FIELDS:
        for entry in entries:
            assert field in entry, f"Missing {field} field in docfact for {entry.get('qname')}"


def test_docfacts_parameter_metadata_complete() -> None:
    for entry in _load_docfacts_entries():
        parameters = entry.get("parameters", [])
        assert isinstance(parameters, list)
        for parameter in parameters:
            assert parameter.get("display_name"), "parameter display name missing"
            assert parameter.get("kind"), "parameter kind missing"


def test_docfacts_return_and_raise_metadata_complete() -> None:
    for entry in _load_docfacts_entries():
        for return_entry in entry.get("returns", []):
            assert "description" in return_entry, "return description missing"
        for raise_entry in entry.get("raises", []):
            assert "description" in raise_entry, "raise description missing"


def test_docfacts_kind_values_are_valid() -> None:
    allowed = {"class", "function", "method"}
    entries = _load_docfacts_entries()
    assert entries, "Docfacts payload should not be empty"
    for entry in entries:
        assert entry["kind"] in allowed, f"Unexpected kind {entry['kind']}"


def test_docfacts_modules_are_importable(monkeypatch: MonkeyPatch) -> None:
    # Ensure the project sources are importable regardless of PYTHONPATH settings.
    del monkeypatch  # MonkeyPatch fixture reserved for future environment tweaks
    repo_root = Path(__file__).resolve().parents[2]
    original_path = list(sys.path)
    sys.path.insert(0, str(repo_root / "src"))
    try:
        for entry in _load_docfacts_entries():
            module_value = entry.get("module")
            if not isinstance(module_value, str) or not module_value:
                continue
            module_name: str = module_value
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - fail fast for visibility
                message = _module_import_error_message(module_name, exc)
                raise AssertionError(message) from exc
    finally:
        sys.path[:] = original_path


def test_mapping_items_accepts_single_mapping() -> None:
    payload = {"name": "example"}
    assert docfacts_module._mapping_items(payload) == [payload]


def test_mapping_items_filters_non_mappings_from_iterable() -> None:
    mapping = {"name": "example"}
    payload = [mapping, "not-a-mapping", 42]
    assert docfacts_module._mapping_items(payload) == [mapping]


def test_mapping_items_rejects_scalar_values() -> None:
    assert docfacts_module._mapping_items("value") == []
    assert docfacts_module._mapping_items(123) == []
