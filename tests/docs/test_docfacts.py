"""Sanity checks for the generated docfacts artifact."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tools.docstring_builder.cli import DOCFACTS_PATH


def _load_docfacts() -> list[dict[str, object]]:
    if not DOCFACTS_PATH.exists():
        raise FileNotFoundError(f"Docfacts missing at {DOCFACTS_PATH}")
    payload = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):  # pragma: no cover - defensive guard
        raise TypeError("Docfacts payload must be a list")
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
        module_name = entry["module"]
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - fail fast for visibility
            pytest.fail(f"Module '{module_name}' from docfacts is not importable: {exc}")
