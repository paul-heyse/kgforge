"""Tests for tools.docs.build_graphs failure handling."""

from __future__ import annotations

import argparse
import importlib.util
import types
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "docs" / "build_graphs.py"
SPEC = importlib.util.spec_from_file_location("tools.docs.build_graphs", MODULE_PATH)
assert SPEC and SPEC.loader, "Failed to load tools.docs.build_graphs"
build_graphs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_graphs)


class FakeFuture:
    def __init__(self, result: tuple[str, bool, bool, bool]) -> None:
        self._result = result

    def result(self) -> tuple[str, bool, bool, bool]:
        return self._result


class FakeExecutor:
    def __init__(self, results: dict[str, tuple[str, bool, bool, bool]]) -> None:
        self._results = results

    def __enter__(self) -> FakeExecutor:
        """Return the executor for context manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> Literal[False]:
        """Propagate any exception raised inside the context."""
        return False

    def submit(
        self,
        fn: Callable[..., tuple[str, bool, bool, bool]],
        pkg: str,
        *args: object,
        **kwargs: object,
    ) -> FakeFuture:
        return FakeFuture(self._results[pkg])


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        packages="",
        format="svg",
        max_bacon=4,
        exclude=None,
        layers=str(tmp_path / "layers.yml"),
        allowlist=str(tmp_path / "allow.json"),
        fail_on_cycles=False,
        fail_on_layer_violations=False,
        max_workers=1,
        cache_dir=str(tmp_path / "cache"),
        no_cache=True,
        verbose=False,
    )


@pytest.fixture(name="graph_env")
def fixture_graph_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "layers.yml").write_text("{}", encoding="utf-8")
    (tmp_path / "allow.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(build_graphs, "parse_args", lambda: _make_args(tmp_path))
    monkeypatch.setattr(build_graphs, "find_top_packages", lambda: ["pkg_ok", "pkg_bad"])
    monkeypatch.setattr(
        build_graphs,
        "ProcessPoolExecutor",
        lambda max_workers: FakeExecutor(
            {
                "pkg_ok": ("pkg_ok", True, True, True),
                "pkg_bad": ("pkg_bad", False, False, True),
            }
        ),
    )
    monkeypatch.setattr(build_graphs, "as_completed", lambda futures: futures)
    monkeypatch.setattr(build_graphs.shutil, "which", lambda name: f"/usr/bin/{name}")

    fake_yaml = types.SimpleNamespace(safe_load=lambda _: {})
    monkeypatch.setattr(build_graphs, "yaml", fake_yaml)
    monkeypatch.setattr(build_graphs, "pydot", object())
    monkeypatch.setattr(build_graphs, "nx", object())
    monkeypatch.setattr(
        build_graphs,
        "build_global_pydeps",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("global graph should not build when packages fail")
        ),
    )


def test_main_exits_when_package_build_fails(
    graph_env: None, capfd: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_graphs.main()

    assert excinfo.value.code == 3
    stderr = capfd.readouterr().err
    assert "pkg_bad" in stderr
    assert "pydeps" in stderr
