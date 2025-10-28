import subprocess
import sys

import pytest

pytest.importorskip("prefect", reason="prefect extra required for pipeline CLI test")


def test_cli_e2e() -> None:
    # Just ensure CLI entrypoint works
    proc = subprocess.run(
        [sys.executable, "-m", "orchestration.cli", "e2e"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "harvest" in proc.stdout
