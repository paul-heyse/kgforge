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
]


def _run_step(name: str, command: list[str], message: str) -> int:
    """Execute a single artefact regeneration step."""
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"[artifacts] {name} failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode
    print(message)
    return 0


def main() -> int:
    """Run all artefact regeneration steps and stop on the first failure."""
    for name, command, message in STEPS:
        status = _run_step(name, command, message)
        if status != 0:
            return status
    print("[artifacts] all steps completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
