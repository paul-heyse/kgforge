"""Coordinator for regenerating documentation artefacts in sequence."""

from __future__ import annotations

import sys
from pathlib import Path

from tools._shared.logging import get_logger, with_fields
from tools._shared.proc import ToolExecutionError, run_tool

LOGGER = get_logger(__name__)

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
    log_adapter = with_fields(LOGGER, operation=name, command=command)
    try:
        result = run_tool(command, timeout=20.0, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            log_adapter.error(
                "[artifacts] %s failed (exit %d)",
                name,
                result.returncode,
                extra={"returncode": result.returncode},
            )
            return result.returncode
    except ToolExecutionError as exc:
        log_adapter.exception("[artifacts] %s failed", name)
        return exc.returncode if exc.returncode is not None else 1
    else:
        log_adapter.info(message)
        return 0


def main() -> int:
    """Run all artefact regeneration steps and stop on the first failure."""
    for name, command, message in STEPS:
        status = _run_step(name, command, message)
        if status != 0:
            return status
    LOGGER.info("[artifacts] all steps completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
