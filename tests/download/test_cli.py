"""Tests for the download CLI migrating to the shared tooling standard."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from download import cli


def test_harvest_emits_envelope(tmp_path: Path, monkeypatch) -> None:
    """The harvest command should write a structured CLI envelope on success."""
    envelope_path = tmp_path / "download.json"
    monkeypatch.setattr(cli, "CLI_ENVELOPE_PATH", envelope_path)
    monkeypatch.setattr(cli, "CLI_ENVELOPE_DIR", envelope_path.parent)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "download",
            "harvest",
            "foundation models",
            "--years",
            ">=2021",
            "--max-works",
            "100",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run" in result.stdout

    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["command"] == cli.CLI_COMMAND
    assert envelope["subcommand"] == "harvest"
    assert envelope["status"] == "success"
    assert envelope["files"][0]["status"] == "success"
