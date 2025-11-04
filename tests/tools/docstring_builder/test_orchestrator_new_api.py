"""Tests for the new public API of the docstring builder orchestrator.

This module tests the typed configuration-based API introduced in Phase 1.2 of
the public API hardening, verifying that run_build() and run_legacy() functions
work correctly with the new DocstringBuildConfig and cache interfaces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config_models import DocstringBuildConfig
from tools.docstring_builder.orchestrator import run_build, run_docstring_builder, run_legacy


class _AnyCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


if TYPE_CHECKING:
    from tools.docstring_builder.cache import DocstringBuilderCache


def _call_untyped_run_build(*args: object, **kwargs: object) -> object:
    untyped = cast("_AnyCallable", run_build)
    return untyped(*args, **kwargs)


class TestRunBuild:
    """Tests for the new run_build() function."""

    def test_run_build_signature_requires_keyword_only(self) -> None:
        """Verify run_build() enforces keyword-only parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DocstringBuildConfig()
            cache = BuilderCache(Path(tmpdir) / "cache.json")

            # Should fail with positional args
            with pytest.raises(TypeError, match="positional argument"):
                _call_untyped_run_build(config, cache)

    def test_run_build_accepts_typed_config(self) -> None:
        """Verify run_build() accepts DocstringBuildConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DocstringBuildConfig(enable_plugins=True, emit_diff=False, timeout_seconds=600)
            cache = BuilderCache(Path(tmpdir) / "cache.json")

            # Should not raise TypeError for missing args
            # (will raise NotImplementedError for unimplemented feature)
            with pytest.raises(NotImplementedError, match="not yet fully implemented"):
                run_build(config=config, cache=cache)

    def test_run_build_requires_cache_interface(self) -> None:
        """Verify run_build() accepts DocstringBuilderCache protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DocstringBuildConfig()
            cache: DocstringBuilderCache = BuilderCache(Path(tmpdir) / "cache.json")

            # Should accept any object satisfying the protocol
            with pytest.raises(NotImplementedError):
                run_build(config=config, cache=cache)

    def test_run_build_not_yet_implemented(self) -> None:
        """Verify run_build() raises NotImplementedError as placeholder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DocstringBuildConfig()
            cache = BuilderCache(Path(tmpdir) / "cache.json")

            with pytest.raises(NotImplementedError, match="not yet fully implemented"):
                run_build(config=config, cache=cache)


class TestRunLegacy:
    """Tests for the deprecation wrapper run_legacy()."""

    def test_run_legacy_emits_deprecation_warning(self) -> None:
        """Verify run_legacy() emits DeprecationWarning."""
        with (
            pytest.warns(DeprecationWarning, match="run_legacy.*deprecated"),
            pytest.raises(NotImplementedError),
        ):
            run_legacy()

    def test_run_legacy_warning_message_guides_migration(self) -> None:
        """Verify deprecation message guides users to new API."""
        with (
            pytest.warns(DeprecationWarning, match="Use run_build\\(config=.*\\) instead"),
            pytest.raises(NotImplementedError),
        ):
            run_legacy()

    def test_run_legacy_accepts_any_args(self) -> None:
        """Verify run_legacy() accepts arbitrary positional and keyword args."""
        # Should emit warning but not raise TypeError
        with (
            pytest.warns(DeprecationWarning, match="deprecated"),
            pytest.raises(NotImplementedError),
        ):
            run_legacy("arg1", "arg2", kwarg1="value1")

    def test_run_legacy_not_yet_implemented(self) -> None:
        """Verify run_legacy() raises NotImplementedError as placeholder."""
        with (
            pytest.warns(DeprecationWarning, match="deprecated"),
            pytest.raises(NotImplementedError, match="not yet fully implemented"),
        ):
            run_legacy()

    def test_run_legacy_warning_appears_once(self) -> None:
        """Verify deprecation warning appears for each call (not suppressed)."""
        # Each call should emit a warning
        for _ in range(2):
            with (
                pytest.warns(DeprecationWarning, match="deprecated"),
                pytest.raises(NotImplementedError),
            ):
                run_legacy()


class TestPublicAPIMigration:
    """Integration tests for migration from legacy to new API."""

    def test_both_apis_available(self) -> None:
        """Verify both run_docstring_builder and run_build are available."""
        assert callable(run_docstring_builder)
        assert callable(run_build)
        assert callable(run_legacy)

    def test_run_build_and_run_legacy_are_distinct(self) -> None:
        """Verify new and legacy APIs are separate functions."""
        assert run_build != run_legacy
        assert run_build.__doc__ != run_legacy.__doc__
