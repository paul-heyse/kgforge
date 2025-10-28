"""Tests for the docs graph builder utility."""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.docs import build_graphs


class _DummyExecutor:
    """Synchronous stand-in for :class:`ProcessPoolExecutor` used in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> _DummyExecutor:
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


def test_build_one_package_supports_png_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Non-default formats should render and populate the cache without rebuilding."""

    out_dir = tmp_path / "graphs"
    out_dir.mkdir()
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(build_graphs, "OUT", out_dir)
    monkeypatch.setattr(build_graphs, "last_tree_commit", lambda _pkg: "deadbeefcafebabe")

    fmt = "png"
    pydeps_calls = 0
    pyrev_calls = 0

    def fake_pydeps(pkg: str, out_path: Path, excludes, max_bacon: int, requested_fmt: str) -> None:
        nonlocal pydeps_calls
        pydeps_calls += 1
        assert requested_fmt == fmt
        out_path.write_text("deps", encoding="utf-8")

    def fake_pyreverse(pkg: str, out_dir_arg: Path, requested_fmt: str) -> None:
        nonlocal pyrev_calls
        pyrev_calls += 1
        assert requested_fmt == fmt
        (out_dir_arg / f"{pkg}-uml.{fmt}").write_text("uml", encoding="utf-8")

    monkeypatch.setattr(build_graphs, "build_pydeps_for_package", fake_pydeps)
    monkeypatch.setattr(build_graphs, "build_pyreverse_for_package", fake_pyreverse)

    result = build_graphs.build_one_package(
        pkg="demo_pkg",
        fmt=fmt,
        excludes=[],
        max_bacon=4,
        cache_dir=cache_dir,
        use_cache=True,
        verbose=False,
    )

    imports_path = out_dir / "demo_pkg-imports.png"
    uml_path = out_dir / "demo_pkg-uml.png"
    assert imports_path.exists()
    assert uml_path.exists()
    assert result == ("demo_pkg", False, True, True)
    assert pydeps_calls == 1
    assert pyrev_calls == 1

    # Delete the freshly rendered files and ensure a cache hit restores them without rerunning builders.
    imports_path.unlink()
    uml_path.unlink()

    def fail_pydeps(*_args, **_kwargs) -> None:  # pragma: no cover - defensive guard
        raise AssertionError("pydeps should not run on cache hit")

    def fail_pyreverse(*_args, **_kwargs) -> None:  # pragma: no cover - defensive guard
        raise AssertionError("pyreverse should not run on cache hit")

    monkeypatch.setattr(build_graphs, "build_pydeps_for_package", fail_pydeps)
    monkeypatch.setattr(build_graphs, "build_pyreverse_for_package", fail_pyreverse)

    cached_result = build_graphs.build_one_package(
        pkg="demo_pkg",
        fmt=fmt,
        excludes=[],
        max_bacon=4,
        cache_dir=cache_dir,
        use_cache=True,
        verbose=False,
    )

    assert cached_result == ("demo_pkg", True, True, True)
    assert imports_path.exists()
    assert uml_path.exists()
