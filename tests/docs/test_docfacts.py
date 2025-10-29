"""Sanity checks for the generated docfacts artifact."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from tools.docstring_builder.cli import DOCFACTS_PATH


def _missing_docfacts_message(path: Path) -> str:
    return f"Docfacts missing at {path}"


def _invalid_payload_message() -> str:
    return "Docfacts payload must be a list"


def _module_import_error_message(module_name: str, exc: Exception) -> str:
    return f"Module '{module_name}' from docfacts is not importable: {exc}"


def _load_docfacts() -> list[dict[str, object]]:
    if not DOCFACTS_PATH.exists():
        message = _missing_docfacts_message(DOCFACTS_PATH)
        raise FileNotFoundError(message)
    payload = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):  # pragma: no cover - defensive guard
        message = _invalid_payload_message()
        raise TypeError(message)
    return payload


def test_docfacts_have_unique_sorted_qnames() -> None:
    docfacts = _load_docfacts()
    qnames = [entry["qname"] for entry in docfacts]
    assert qnames == sorted(qnames), "Docfacts must be sorted by qualified name"
    assert len(qnames) == len(set(qnames)), "Docfacts must not contain duplicate qnames"


@pytest.mark.parametrize("field", ["module", "kind", "parameters", "returns", "raises", "notes"])
def test_docfacts_required_fields_present(field: str) -> None:
    for entry in _load_docfacts():
        assert field in entry, f"Missing {field} field in docfact for {entry.get('qname')}"


def test_docfacts_parameter_metadata_complete() -> None:
    for entry in _load_docfacts():
        parameters = entry.get("parameters", [])
        assert isinstance(parameters, list)
        for parameter in parameters:
            assert "display_name" in parameter, "parameter display name missing"
            assert "kind" in parameter, "parameter kind missing"
            assert parameter["display_name"], "parameter display name must be truthy"
            assert parameter["kind"], "parameter kind must be truthy"


def test_docfacts_kind_values_are_valid() -> None:
    allowed = {"class", "function", "method"}
    docfacts = _load_docfacts()
    assert docfacts, "Docfacts payload should not be empty"
    for entry in docfacts:
        assert entry["kind"] in allowed, f"Unexpected kind {entry['kind']}"


def test_docfacts_modules_are_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the project sources are importable regardless of PYTHONPATH settings.
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    for entry in _load_docfacts():
        module_value = entry.get("module")
        if not isinstance(module_value, str) or not module_value:
            continue
        module_name: str = module_value
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - fail fast for visibility
            message = _module_import_error_message(module_name, exc)
            pytest.fail(message)
