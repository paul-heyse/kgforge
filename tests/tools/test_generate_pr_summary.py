"""Tests for :mod:`tools.generate_pr_summary`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import pytest
from tools.generate_pr_summary import (
    ArtifactSnapshot,
    CheckStatus,
    StatusLiteral,
    collect_artifact_snapshot,
    generate_summary,
)

ArtifactCase = tuple[dict[str, str], str]
StatusIconCase = tuple[StatusLiteral, str]

ARTIFACT_CASES: Sequence[ArtifactCase] = (
    (
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
    ),
    (
        {},
        "- ⚪ No coverage or JUnit artifacts found",
    ),
)

ARTIFACT_CASE_IDS: Sequence[str] = (
    "all-artifacts-present",
    "no-artifacts",
)

STATUS_ICON_CASES: Sequence[StatusIconCase] = (
    ("pass", "✅"),
    ("fail", "❌"),
    ("skip", "⚪"),
)

STATUS_ICON_IDS: Sequence[str] = (
    "pass",
    "fail",
    "skip",
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


if TYPE_CHECKING:

    def typed_parametrize(
        *args: object, **kwargs: object
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Type-aware shim for :func:`pytest.mark.parametrize`."""
        ...

else:

    def typed_parametrize(
        *args: object, **kwargs: object
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Wrap :func:`pytest.mark.parametrize` with precise typing."""
        return cast(
            Callable[[Callable[_P, _R]], Callable[_P, _R]],
            pytest.mark.parametrize(*args, **kwargs),
        )


@typed_parametrize(
    ("artifact_paths", "expected_line"),
    ARTIFACT_CASES,
    ids=ARTIFACT_CASE_IDS,
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


@typed_parametrize(
    ("status", "icon"),
    STATUS_ICON_CASES,
    ids=STATUS_ICON_IDS,
)
def test_generate_summary_quality_gate_icons(status: StatusLiteral, icon: str) -> None:
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
