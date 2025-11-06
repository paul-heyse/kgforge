from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from codeintel.indexer import cli as indexer_cli
from typer.testing import CliRunner


@pytest.fixture(name="runner")
def _runner() -> CliRunner:
    return CliRunner()


def _read_envelope(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    return cast("dict[str, Any]", json.loads(payload))


def test_query_emits_success_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    monkeypatch.setattr(indexer_cli, "CLI_ENVELOPE_DIR", tmp_path)

    source_file = tmp_path / "example.py"
    source_file.write_text("print('hello world')\n", encoding="utf-8")
    query_file = tmp_path / "capture.scm"
    query_file.write_text("(identifier) @ident\n", encoding="utf-8")

    def fake_load_langs() -> object:
        return object()

    def fake_get_language(_langs: object, _language: str) -> object:
        return object()

    def fake_parse_bytes(_lang: object, _data: bytes) -> object:
        return object()

    def fake_run_query(
        _lang: object, _query_text: str, _tree: object, _data: bytes
    ) -> list[dict[str, object]]:
        return [{"kind": "identifier", "capture": "ident", "match_id": 0}]

    monkeypatch.setattr(indexer_cli, "load_langs", fake_load_langs)
    monkeypatch.setattr(indexer_cli, "get_language", fake_get_language)
    monkeypatch.setattr(indexer_cli, "parse_bytes", fake_parse_bytes)
    monkeypatch.setattr(indexer_cli, "run_query", fake_run_query)

    result = runner.invoke(
        indexer_cli.app,
        [
            "query",
            str(source_file),
            "--language",
            "python",
            "--query",
            str(query_file),
        ],
    )

    assert result.exit_code == 0
    envelope_path = tmp_path / "kgf-codeintel-codeintel-query.json"
    envelope = _read_envelope(envelope_path)
    assert envelope["status"] == "success"
    files = cast("list[dict[str, Any]]", envelope["files"])
    assert any(entry.get("path") == str(source_file) for entry in files)
    captures = json.loads(result.stdout)
    assert isinstance(captures, list)


def test_query_invalid_language_records_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    monkeypatch.setattr(indexer_cli, "CLI_ENVELOPE_DIR", tmp_path)

    source_file = tmp_path / "example.py"
    source_file.write_text("print('hello world')\n", encoding="utf-8")
    query_file = tmp_path / "capture.scm"
    query_file.write_text("(identifier) @ident\n", encoding="utf-8")

    result = runner.invoke(
        indexer_cli.app,
        [
            "query",
            str(source_file),
            "--language",
            "invalid",
            "--query",
            str(query_file),
        ],
    )

    assert result.exit_code == 2
    envelope_path = tmp_path / "kgf-codeintel-codeintel-query.json"
    envelope = _read_envelope(envelope_path)
    assert envelope["status"] == "violation"
    problem = cast("dict[str, Any]", envelope["problem"])
    assert problem["status"] == 422
    assert "unsupported language" in str(problem["detail"]).lower()
