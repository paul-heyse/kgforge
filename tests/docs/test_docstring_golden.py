from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "docs" / "fixtures" / "docstring_examples.py"
GOLDEN_PATH = REPO_ROOT / "tests" / "docs" / "goldens" / "docstring_examples.diff"


def _run_builder() -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "tools.docstring_builder.cli",
        "--diff",
        "--force",
        "lint",
        "--no-docfacts",
        str(FIXTURE_PATH),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_docstring_builder_golden() -> None:
    result = _run_builder()
    actual = result.stdout.strip()
    if os.getenv("UPDATE_GOLDENS") == "1":
        GOLDEN_PATH.write_text(actual + "\n", encoding="utf-8")
        return
    assert result.returncode == 1, result.stderr
    expected = GOLDEN_PATH.read_text(encoding="utf-8").strip()
    assert actual == expected
