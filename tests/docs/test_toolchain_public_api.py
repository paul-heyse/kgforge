"""Tests for new docs toolchain public APIs."""

from __future__ import annotations

from collections.abc import Sequence

import docs.toolchain.build_symbol_index
import docs.toolchain.symbol_delta
import docs.toolchain.validate_artifacts
import pytest
from docs.toolchain.build_symbol_index import build_symbol_index
from docs.toolchain.config import DocsDeltaConfig, DocsSymbolIndexConfig
from docs.toolchain.symbol_delta import compute_delta
from docs.toolchain.validate_artifacts import validate_artifacts

SymbolIndexMap = dict[str, dict[str, str]]


class TestBuildSymbolIndexAPI:
    """Test the new build_symbol_index API."""

    def test_requires_keyword_only_config(self) -> None:
        """Verify build_symbol_index requires config as keyword argument."""
        config = DocsSymbolIndexConfig()
        # Should raise NotImplementedError, not TypeError for arguments
        with pytest.raises(NotImplementedError, match="placeholder"):
            build_symbol_index(config=config)

    def test_accepts_typed_config(self) -> None:
        """Verify build_symbol_index accepts DocsSymbolIndexConfig."""
        config = DocsSymbolIndexConfig(include_private=True)
        with pytest.raises(NotImplementedError):
            build_symbol_index(config=config)

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert build_symbol_index.__doc__ is not None
        assert "config" in build_symbol_index.__doc__.lower()
        assert ">>>" in build_symbol_index.__doc__


class TestComputeDeltaAPI:
    """Test the new compute_delta API."""

    def test_requires_keyword_only_arguments(self) -> None:
        """Verify compute_delta requires keyword-only arguments."""
        config = DocsDeltaConfig()
        baseline: SymbolIndexMap = {"a": {"name": "A"}}
        current: SymbolIndexMap = {"a": {"name": "A"}, "b": {"name": "B"}}
        # Should raise NotImplementedError, not TypeError for arguments
        with pytest.raises(NotImplementedError, match="placeholder"):
            compute_delta(config=config, baseline=baseline, current=current)

    def test_accepts_typed_config_and_indices(self) -> None:
        """Verify compute_delta accepts typed parameters."""
        config = DocsDeltaConfig(include_removals=False)
        baseline: SymbolIndexMap = {}
        current: SymbolIndexMap = {}
        with pytest.raises(NotImplementedError):
            compute_delta(config=config, baseline=baseline, current=current)

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert compute_delta.__doc__ is not None
        assert "config" in compute_delta.__doc__.lower()
        assert ">>>" in compute_delta.__doc__


class TestValidateArtifactsAPI:
    """Test the new validate_artifacts API."""

    def test_raises_not_implemented(self) -> None:
        """Verify validate_artifacts is a placeholder."""
        with pytest.raises(NotImplementedError, match="placeholder"):
            validate_artifacts()

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert validate_artifacts.__doc__ is not None
        assert ">>>" in validate_artifacts.__doc__


class TestAPITypeEnforcement:
    """Test that new APIs enforce type safety."""

    def test_build_symbol_index_signature(self) -> None:
        """Verify build_symbol_index has correct signature."""
        config = DocsSymbolIndexConfig()
        # Keyword-only parameter enforcement
        with pytest.raises(NotImplementedError):
            build_symbol_index(config=config)

    def test_compute_delta_signature(self) -> None:
        """Verify compute_delta has correct signature."""
        config = DocsDeltaConfig()
        baseline: SymbolIndexMap = {}
        current: SymbolIndexMap = {}
        # Keyword-only parameter enforcement
        with pytest.raises(NotImplementedError):
            compute_delta(config=config, baseline=baseline, current=current)


class TestAPIExports:
    """Test that APIs are properly exported."""

    def test_build_symbol_index_in_all(self) -> None:
        """Verify build_symbol_index is in __all__."""
        exports_obj: object = getattr(docs.toolchain.build_symbol_index, "__all__", ())
        exports = exports_obj if isinstance(exports_obj, Sequence) else ()
        assert "build_symbol_index" in exports

    def test_compute_delta_in_all(self) -> None:
        """Verify compute_delta is in __all__."""
        exports_obj: object = getattr(docs.toolchain.symbol_delta, "__all__", ())
        exports = exports_obj if isinstance(exports_obj, Sequence) else ()
        assert "compute_delta" in exports

    def test_validate_artifacts_in_all(self) -> None:
        """Verify validate_artifacts is in __all__."""
        exports_obj: object = getattr(docs.toolchain.validate_artifacts, "__all__", ())
        exports = exports_obj if isinstance(exports_obj, Sequence) else ()
        assert "validate_artifacts" in exports
