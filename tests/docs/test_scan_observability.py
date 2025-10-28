"""Regression coverage for the observability scanner CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch
from tools.docs import scan_observability


def test_config_summary_invoked_once(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Ensure the summary writer executes exactly once per run."""
    root = tmp_path
    src = root / "src"
    src.mkdir()

    out = root / "docs" / "_build"
    out.mkdir(parents=True)

    # Wire up the module to operate on the temporary layout created above.
    monkeypatch.setattr(scan_observability, "ROOT", root)
    monkeypatch.setattr(scan_observability, "SRC", src)
    monkeypatch.setattr(scan_observability, "OUT", out)
    monkeypatch.setattr(scan_observability, "CONFIG_MD", out / "config.md")
    monkeypatch.setattr(
        scan_observability,
        "POLICY_PATH",
        root / "docs" / "policies" / "observability.yml",
    )

    # Capture sys.exit calls so the CLI doesn't terminate the test process.
    exit_codes: list[int | None] = []

    def fake_exit(code: int | None = None) -> None:
        exit_codes.append(code)

    monkeypatch.setattr(scan_observability, "sys", SimpleNamespace(exit=fake_exit))

    # Track how many times the summary helper runs while still delegating to it.
    call_count = 0
    original_write = scan_observability._write_config_summary

    def tracked_write(
        metrics: list[scan_observability.MetricRow],
        logs: list[scan_observability.LogRow],
        traces: list[scan_observability.TraceRow],
    ) -> None:
        nonlocal call_count
        call_count += 1
        return original_write(metrics, logs, traces)

    monkeypatch.setattr(scan_observability, "_write_config_summary", tracked_write)

    scan_observability.main()

    assert call_count == 1
    assert exit_codes == [0]
    assert (out / "config.md").exists()
