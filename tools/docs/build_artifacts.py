"""Coordinator for regenerating documentation artefacts in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STEPS: list[tuple[str, list[str], str]] = [
    (
        "docstrings",
        [
            sys.executable,
            "-m",
            "tools.docstring_builder.cli",
            "--all",
            "--ignore-missing",
            "generate",
        ],
        "[docstrings] synchronized managed docstrings and DocFacts",
    ),
    (
        "navmap",
        [
            sys.executable,
            "-m",
            "tools.navmap.build_navmap",
            "--write",
            str(REPO_ROOT / "site" / "_build" / "navmap" / "navmap.json"),
        ],
        "[navmap] regenerated site/_build/navmap/navmap.json",
    ),
    (
        "test-map",
        [sys.executable, "tools/docs/build_test_map.py"],
        "[testmap] refreshed docs/_build/test_map artefacts",
    ),
    (
        "observability",
        [sys.executable, "tools/docs/scan_observability.py"],
        "[observability] updated docs/_build/configuration snapshots",
    ),
    (
        "schemas",
        [sys.executable, "tools/docs/export_schemas.py"],
        "[schemas] exported API schemas",
    ),
    (
        "agent-catalog",
        [sys.executable, "tools/docs/build_agent_catalog.py"],
        "[agent] generated docs/_build/agent_catalog.json",
    ),
    (
        "agent-api",
        [sys.executable, "tools/docs/build_agent_api.py"],
        "[agent] generated docs/_build/agent_api_openapi.json",
    ),
    (
        "agent-portal",
        [sys.executable, "tools/docs/render_agent_portal.py"],
        "[agent] rendered site/_build/agent/index.html",
    ),
    (
        "agent-analytics",
        [sys.executable, "tools/docs/build_agent_analytics.py"],
        "[agent] wrote docs/_build/analytics.json",
    ),
]


def _run_step(name: str, command: list[str], message: str) -> int:
    """Execute a single artefact regeneration step."""
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603 - commands are static python invocations
    if result.returncode != 0:
        sys.stderr.write(f"[artifacts] {name} failed (exit {result.returncode})\n")
        return result.returncode
    sys.stdout.write(f"{message}\n")
    return 0


def main() -> int:
    """Run all artefact regeneration steps and stop on the first failure."""
    for name, command, message in STEPS:
        status = _run_step(name, command, message)
        if status != 0:
            return status
    sys.stdout.write("[artifacts] all steps completed successfully\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
