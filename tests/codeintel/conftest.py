"""Test fixtures and configuration for CodeIntel tests."""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import pytest

P = ParamSpec("P")
T = TypeVar("T")


@pytest.fixture(scope="session")
def repo_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a synthetic repository for testing.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest temporary path factory.

    Returns
    -------
    Path
        Path to synthetic repository root.
    """
    r = tmp_path_factory.mktemp("repo")
    (r / "pkg").mkdir()
    (r / "pkg" / "mod.py").write_text("class A:\n    def f(self, x):\n        return x\n\n")
    (r / "README.md").write_text("# sample\n\n```python\nprint('hi')\n```\n")
    (r / "pyproject.toml").write_text('[tool]\nname="demo"\n')
    # Create queries directory structure
    queries_dir = r / "codeintel" / "queries"
    queries_dir.mkdir(parents=True)
    # Create minimal Python query file for tests
    (queries_dir / "python.scm").write_text(
        "(function_definition name: (identifier) @def.name) @def.node\n"
    )
    return r


@pytest.fixture(autouse=True)
def set_env(repo_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables and module state for test isolation.

    Parameters
    ----------
    repo_fixture : Path
        Synthetic repository root.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("KGF_REPO_ROOT", str(repo_fixture))
    monkeypatch.setenv("CODEINTEL_MAX_AST_BYTES", "65536")
    monkeypatch.setenv("CODEINTEL_LIMIT_MAX", "1000")
    monkeypatch.setenv("CODEINTEL_ENABLE_TS_QUERY", "1")  # enable advanced query in tests
    # Update REPO_ROOT in tools module since it's set at import time
    from codeintel.mcp_server import tools

    monkeypatch.setattr(tools, "REPO_ROOT", repo_fixture.resolve())
    monkeypatch.setattr(tools, "QUERIES_DIR", repo_fixture.resolve() / "codeintel" / "queries")


try:
    _BENCHMARK_AVAILABLE = importlib.util.find_spec("pytest_benchmark.plugin") is not None
except ModuleNotFoundError:  # pragma: no cover - optional dep
    _BENCHMARK_AVAILABLE = False


if not _BENCHMARK_AVAILABLE:  # pragma: no cover - optional dep

    class _BenchmarkStub:
        """Lightweight fallback for pytest-benchmark when the plugin is unavailable."""

        __slots__ = ("_iterations", "stats")

        def __init__(self, iterations: int = 5) -> None:
            self._iterations = max(iterations, 1)
            self.stats: dict[str, float] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "iterations": 0.0,
            }

        def __call__(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
            durations: list[float] = []
            result: T | None = None
            for _ in range(self._iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                durations.append(time.perf_counter() - start)
            stats = {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "iterations": float(len(durations)),
            }
            self.stats = stats
            return cast("T", result)

    @pytest.fixture
    def benchmark() -> _BenchmarkStub:
        """Provide a minimal benchmark fixture when pytest-benchmark is absent.

        Returns
        -------
        _BenchmarkStub
            Simple callable capturing timing statistics for the provided function.
        """
        return _BenchmarkStub()
