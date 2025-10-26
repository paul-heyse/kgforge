
import subprocess, sys

def test_cli_e2e():
    # Just ensure CLI entrypoint works
    proc = subprocess.run([sys.executable, "-m", "orchestration.cli", "e2e"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "harvest" in proc.stdout
