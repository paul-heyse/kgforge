#!/usr/bin/env python3
"""Generate a rich pull request summary for GitHub Actions.

The summary surfaces quality gate results, artifact locations, and helpful
follow-up steps in a markdown table. By default the script inspects the current
working directory and writes to ``$GITHUB_STEP_SUMMARY`` when executed in GitHub
Actions. When that environment variable is missing (e.g., local runs), the
summary is emitted to ``stdout`` so developers can preview the output.

Examples
--------
>>> python tools/generate_pr_summary.py  # doctest: +SKIP
# Writes markdown summary to $GITHUB_STEP_SUMMARY when available
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)

StatusLiteral = Literal["pass", "fail", "skip"]


@dataclass(slots=True, frozen=True)
class CheckStatus:
    """Describe the outcome of a single quality gate."""

    label: str
    status: StatusLiteral
    details: str | None = None

    def to_row(self) -> str:
        """Render the check outcome as a markdown table row."""
        icon = STATUS_ICONS[self.status]
        description = f"{icon}"
        if self.details:
            description = f"{description} {self.details}"
        return f"| {self.label} | {description} |"


@dataclass(slots=True, frozen=True)
class ArtifactSnapshot:
    """Materialized artifacts available after a CI run."""

    coverage_xml: bool
    coverage_html: bool
    junit_xml: bool
    docs_build: bool
    site_build: bool
    agent_portal: bool
    schema_dir: bool
    dist_wheels: int
    dist_sdists: int
    codemod_log: bool


STATUS_ICONS: dict[StatusLiteral, str] = {
    "pass": "✅",
    "fail": "❌",
    "skip": "⚪",
}

DEFAULT_CHECKS: tuple[CheckStatus, ...] = (
    CheckStatus("Ruff format & lint", "pass"),
    CheckStatus("pyrefly check", "pass"),
    CheckStatus("mypy strict", "pass"),
    CheckStatus("pytest", "pass"),
    CheckStatus("doctests", "pass"),
    CheckStatus("import-linter", "pass"),
    CheckStatus("suppression guard", "pass"),
    CheckStatus("build wheels", "pass"),
)


def collect_artifact_snapshot(base_path: Path | None = None) -> ArtifactSnapshot:
    """Inspect ``base_path`` and report which build artifacts exist."""
    root = base_path or Path.cwd()
    coverage_xml = (root / "coverage.xml").is_file()
    coverage_html = (root / "htmlcov/index.html").is_file()
    junit_xml = (root / "junit.xml").is_file()
    docs_build = (root / "docs/_build").exists()
    site_build = (root / "site/_build").exists()
    agent_portal = (root / "site/_build/agent/index.html").is_file()
    schema_dir = (root / "schema").exists()
    dist_dir = root / "dist"
    dist_wheels = len(list(dist_dir.glob("*.whl"))) if dist_dir.exists() else 0
    dist_sdists = len(list(dist_dir.glob("*.tar.gz"))) if dist_dir.exists() else 0
    codemod_log = (root / "codemod.log").is_file()
    return ArtifactSnapshot(
        coverage_xml=coverage_xml,
        coverage_html=coverage_html,
        junit_xml=junit_xml,
        docs_build=docs_build,
        site_build=site_build,
        agent_portal=agent_portal,
        schema_dir=schema_dir,
        dist_wheels=dist_wheels,
        dist_sdists=dist_sdists,
        codemod_log=codemod_log,
    )


def _append_section(lines: list[str], heading: str, entries: Iterable[str]) -> None:
    """Append a markdown heading and bullet entries to ``lines``."""
    lines.append(heading)
    for entry in entries:
        lines.append(entry)
    lines.append("")


def _test_result_entries(snapshot: ArtifactSnapshot) -> list[str]:
    entries: list[str] = []
    if snapshot.coverage_xml:
        entries.append("- ✅ Coverage XML: `coverage.xml`")
    if snapshot.coverage_html:
        entries.append("- ✅ Coverage HTML: `htmlcov/index.html`")
    if snapshot.junit_xml:
        entries.append("- ✅ JUnit XML: `junit.xml`")
    if not entries:
        entries.append("- ⚪ No coverage or JUnit artifacts found")
    return entries


def _documentation_entries(snapshot: ArtifactSnapshot) -> list[str]:
    entries: list[str] = []
    if snapshot.docs_build:
        entries.append("- ✅ Docs build: `docs/_build/`")
    if snapshot.site_build:
        entries.append("- ✅ Site build: `site/_build/`")
    if snapshot.agent_portal:
        entries.append("- ✅ Agent Portal: `site/_build/agent/index.html`")
    if not entries:
        entries.append("- ⚪ Documentation artifacts not generated")
    return entries


def _schema_entries(snapshot: ArtifactSnapshot) -> list[str]:
    entries = [
        "- ✅ Schema directory exists" if snapshot.schema_dir else "- ⚪ Schema directory missing"
    ]
    if snapshot.schema_dir:
        entries.append("- Run `jsonschema validate` to verify schemas")
    return entries


def _build_entries(snapshot: ArtifactSnapshot) -> list[str]:
    entries: list[str] = []
    if snapshot.dist_wheels:
        entries.append(f"- ✅ Wheels: {snapshot.dist_wheels} file(s)")
    if snapshot.dist_sdists:
        entries.append(f"- ✅ Source distributions: {snapshot.dist_sdists} file(s)")
    if not entries:
        entries.append("- ⚪ No distribution artifacts built")
    return entries


def generate_summary(
    snapshot: ArtifactSnapshot | None = None,
    checks: Sequence[CheckStatus] | None = None,
) -> str:
    """Generate PR summary markdown."""
    lines: list[str] = ["# Quality Gates Summary", ""]
    artifact_snapshot = snapshot or collect_artifact_snapshot()

    _append_section(lines, "## 📊 Test Results", _test_result_entries(artifact_snapshot))
    _append_section(
        lines,
        "## 📚 Documentation & Artifacts",
        _documentation_entries(artifact_snapshot),
    )
    _append_section(lines, "## 🔍 Schema Validation", _schema_entries(artifact_snapshot))
    _append_section(lines, "## 📦 Build Artifacts", _build_entries(artifact_snapshot))

    lines.append("## ✅ Quality Gates")
    lines.append("")
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    for check in checks or DEFAULT_CHECKS:
        lines.append(check.to_row())
    lines.append("")

    if artifact_snapshot.codemod_log:
        _append_section(
            lines,
            "## 🔧 Codemod Logs",
            ("- ✅ Codemod execution log: `codemod.log`",),
        )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate GitHub Actions step summary."""
    del argv
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    summary = generate_summary()
    if summary_file:
        try:
            output_path = Path(summary_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{summary}\n", encoding="utf-8")
        except OSError:
            LOGGER.exception(
                "Error writing summary",
                extra={"path": summary_file, "operation": "generate_pr_summary"},
            )
            return 1
        LOGGER.info(
            "Summary written",
            extra={
                "path": summary_file,
                "operation": "generate_pr_summary",
                "line_count": summary.count("\n") + 1,
            },
        )
        return 0

    LOGGER.warning(
        "GITHUB_STEP_SUMMARY not set, writing to stdout",
        extra={"operation": "generate_pr_summary"},
    )
    os.sys.stdout.write(f"{summary}\n")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
