"""Tests for CLI entrypoints."""

from __future__ import annotations

from codeintel.cli import app
from typer.testing import CliRunner


def test_cli_serve_help() -> None:
    """Test that CLI serve command shows help."""
    runner = CliRunner()
    res = runner.invoke(app, ["mcp", "serve", "--help"])
    assert res.exit_code == 0
    assert "Start the CodeIntel MCP server" in res.stdout


def test_cli_index_build_help() -> None:
    """Test that CLI index build command shows help."""
    runner = CliRunner()
    res = runner.invoke(app, ["index", "build", "--help"])
    assert res.exit_code == 0
    assert "Build or update" in res.stdout
