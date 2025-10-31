"""Tests for :mod:`tools.generate_pr_summary`."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.generate_pr_summary import (
    ArtifactSnapshot,
    CheckStatus,
    collect_artifact_snapshot,
    generate_summary,
)


@pytest.mark.parametrize(
    ("artifact_paths", "expected_line"),
    [
        pytest.param(
            {
                "coverage.xml": "",
                "htmlcov/index.html": "",
                "junit.xml": "",
                "docs/_build/placeholder": "",
                "site/_build/agent/index.html": "",
                "dist/package-0.1.0-py3-none-any.whl": "",
                "dist/package-0.1.0.tar.gz": "",
                "codemod.log": "diagnostics",
            },
            "- ✅ Coverage XML: `coverage.xml`",
            id="all-artifacts-present",
        ),
        pytest.param(
            {},
            "- ⚪ No coverage or JUnit artifacts found",
            id="no-artifacts",
        ),
    ],
)
def test_collect_artifact_snapshot(
    tmp_path: Path, artifact_paths: dict[str, str], expected_line: str
) -> None:
    """`collect_artifact_snapshot` surfaces generated artifacts in the summary."""
    for relative_path, contents in artifact_paths.items():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")

    snapshot = collect_artifact_snapshot(tmp_path)
    summary = generate_summary(snapshot=snapshot)
    assert expected_line in summary


@pytest.mark.parametrize(
    ("status", "icon"),
    [
        pytest.param("pass", "✅", id="pass"),
        pytest.param("fail", "❌", id="fail"),
        pytest.param("skip", "⚪", id="skip"),
    ],
)
def test_generate_summary_quality_gate_icons(status: str, icon: str) -> None:
    """`generate_summary` renders icons that match the provided status."""
    snapshot = ArtifactSnapshot(
        coverage_xml=False,
        coverage_html=False,
        junit_xml=False,
        docs_build=False,
        site_build=False,
        agent_portal=False,
        schema_dir=False,
        dist_wheels=0,
        dist_sdists=0,
        codemod_log=False,
    )
    check = CheckStatus("example", status)

    summary = generate_summary(snapshot=snapshot, checks=[check])

    assert f"| example | {icon}" in summary
