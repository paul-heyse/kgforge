"""Tests for the docs graph builder utility."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.docs import build_graphs


class _DummyExecutor:
    """Synchronous stand-in for :class:`ProcessPoolExecutor` used in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_DummyExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:  # type: ignore[override]
        fut: Future = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)
        return fut

    def shutdown(self, wait: bool = True) -> None:  # pragma: no cover - interface compat
        return None


def test_main_exits_on_failure_and_cleans_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should abort and delete partial files when a per-package build fails."""

    out_dir = tmp_path / "graphs"
    out_dir.mkdir()
    monkeypatch.setattr(build_graphs, "OUT", out_dir)

    cache_dir = tmp_path / "cache"
    layers = tmp_path / "layers.yml"
    allow = tmp_path / "allow.json"
    layers.write_text("{}", encoding="utf-8")
    allow.write_text("{}", encoding="utf-8")

    args = SimpleNamespace(
        packages="demo_pkg",
        format="svg",
        max_bacon=4,
        exclude=[],
        layers=str(layers),
        allowlist=str(allow),
        fail_on_cycles=True,
        fail_on_layer_violations=True,
        max_workers=1,
        cache_dir=str(cache_dir),
        no_cache=False,
        verbose=False,
    )
    monkeypatch.setattr(build_graphs, "parse_args", lambda: args)

    def failing_pydeps(pkg: str, out_svg, excludes, max_bacon, fmt) -> None:
        out_svg.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    def fake_pyreverse(pkg: str, out_dir_arg, fmt: str) -> None:
        out_dir_arg.mkdir(parents=True, exist_ok=True)
        (out_dir_arg / f"{pkg}-uml.{fmt}").write_text("uml", encoding="utf-8")

    monkeypatch.setattr(build_graphs, "build_pydeps_for_package", failing_pydeps)
    monkeypatch.setattr(build_graphs, "build_pyreverse_for_package", fake_pyreverse)
    monkeypatch.setattr(build_graphs, "ProcessPoolExecutor", _DummyExecutor)
    monkeypatch.setattr(build_graphs, "pydot", object())
    monkeypatch.setattr(build_graphs, "nx", object())
    monkeypatch.setattr(build_graphs, "yaml", object())
    monkeypatch.setattr(build_graphs.shutil, "which", lambda _name: True)

    with pytest.raises(SystemExit) as excinfo:
        build_graphs.main()

    assert excinfo.value.code == 3

    result = (build_graphs.OUT / "demo_pkg-imports.svg", build_graphs.OUT / "demo_pkg-uml.svg")
    assert not result[0].exists()
    assert not result[1].exists()

    captured = capsys.readouterr()
    assert "build failures detected" in captured.err
    assert "demo_pkg" in captured.err
