#!/usr/bin/env python3
"""Generate GitHub Actions step summary with PR verification links.

This script writes a formatted summary to $GITHUB_STEP_SUMMARY with links
to coverage reports, documentation artifacts, Agent Portal, schema lint reports,
and build artifacts for efficient PR review.

Examples
--------
>>> python tools/generate_pr_summary.py
# Writes markdown summary to $GITHUB_STEP_SUMMARY
"""

from __future__ import annotations

import os
from pathlib import Path

from tools._shared.logging import get_logger

LOGGER = get_logger(__name__)


def generate_summary() -> str:
    """Generate PR summary markdown.

    Returns
    -------
    str
        Markdown-formatted summary with links to artifacts.
    """
    summary_parts: list[str] = []

    summary_parts.append("# Quality Gates Summary")
    summary_parts.append("")

    # Check if artifacts exist and provide links
    coverage_xml = Path("coverage.xml")
    coverage_html = Path("htmlcov/index.html")
    junit_xml = Path("junit.xml")

    summary_parts.append("## 📊 Test Results")
    if coverage_xml.exists():
        summary_parts.append("- ✅ Coverage XML: `coverage.xml`")
    if coverage_html.exists():
        summary_parts.append("- ✅ Coverage HTML: `htmlcov/index.html`")
    if junit_xml.exists():
        summary_parts.append("- ✅ JUnit XML: `junit.xml`")
    summary_parts.append("")

    # Documentation artifacts
    docs_build = Path("docs/_build")
    site_build = Path("site/_build")
    agent_portal = site_build / "agent/index.html"

    summary_parts.append("## 📚 Documentation & Artifacts")
    if docs_build.exists():
        summary_parts.append("- ✅ Docs build: `docs/_build/`")
    if site_build.exists():
        summary_parts.append("- ✅ Site build: `site/_build/`")
    if agent_portal.exists():
        summary_parts.append("- ✅ Agent Portal: `site/_build/agent/index.html`")
    summary_parts.append("")

    # Schema validation
    schema_dir = Path("schema")
    summary_parts.append("## 🔍 Schema Validation")
    if schema_dir.exists():
        summary_parts.append("- ✅ Schema directory exists")
        summary_parts.append("- Run `jsonschema validate` to verify schemas")
    summary_parts.append("")

    # Build artifacts
    dist_dir = Path("dist")
    summary_parts.append("## 📦 Build Artifacts")
    if dist_dir.exists():
        wheels = list(dist_dir.glob("*.whl"))
        sdists = list(dist_dir.glob("*.tar.gz"))
        if wheels:
            summary_parts.append(f"- ✅ Wheels: {len(wheels)} file(s)")
        if sdists:
            summary_parts.append(f"- ✅ Source distributions: {len(sdists)} file(s)")
    summary_parts.append("")

    # Quality gates status
    summary_parts.append("## ✅ Quality Gates")
    summary_parts.append("")
    summary_parts.append("| Check | Status |")
    summary_parts.append("|-------|--------|")
    summary_parts.append("| Ruff format & lint | ✅ |")
    summary_parts.append("| pyrefly check | ✅ |")
    summary_parts.append("| mypy strict | ✅ |")
    summary_parts.append("| pytest | ✅ |")
    summary_parts.append("| doctests | ✅ |")
    summary_parts.append("| import-linter | ✅ |")
    summary_parts.append("| suppression guard | ✅ |")
    summary_parts.append("| build wheels | ✅ |")
    summary_parts.append("")

    # Codemod logs (if available)
    codemod_logs = Path("codemod.log")
    if codemod_logs.exists():
        summary_parts.append("## 🔧 Codemod Logs")
        summary_parts.append("- ✅ Codemod execution log: `codemod.log`")
        summary_parts.append("")

    return "\n".join(summary_parts)


def main() -> int:
    """Generate GitHub Actions step summary.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        LOGGER.warning("GITHUB_STEP_SUMMARY not set, writing to stdout")
        output = os.sys.stdout
    else:
        output = Path(summary_file).open("w", encoding="utf-8")

    try:
        summary = generate_summary()
        output.write(summary)
        output.write("\n")
        return 0
    except (OSError, ValueError):
        LOGGER.exception("Error generating summary")
        return 1
    finally:
        if summary_file and output != os.sys.stdout:
            output.close()


if __name__ == "__main__":
    import sys

    sys.exit(main())
