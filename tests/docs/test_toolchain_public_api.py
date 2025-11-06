"""Tests for docs toolchain public APIs."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

from docs.toolchain.build_symbol_index import build_symbol_index
from docs.toolchain.symbol_delta import symbol_delta
from docs.toolchain.validate_artifacts import validate_artifacts


class TestBuildSymbolIndexAPI:
    """Basic contract checks for build_symbol_index."""

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert build_symbol_index.__doc__ is not None
        assert "operation" in build_symbol_index.__doc__.lower()


class TestSymbolDeltaAPI:
    """Basic contract checks for symbol_delta."""

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert symbol_delta.__doc__ is not None
        assert "delta" in symbol_delta.__doc__.lower()


class TestValidateArtifactsAPI:
    """Basic contract checks for validate_artifacts."""

    def test_function_has_docstring(self) -> None:
        """Verify function is properly documented."""
        assert validate_artifacts.__doc__ is not None
        assert "validate" in validate_artifacts.__doc__.lower()


class TestAPIExports:
    """Test that APIs are properly exported."""

    @staticmethod
    def _module_exports(module_name: str) -> Sequence[str]:
        module = import_module(module_name)
        exports_obj: object = getattr(module, "__all__", ())
        return exports_obj if isinstance(exports_obj, Sequence) else ()

    def test_build_symbol_index_in_all(self) -> None:
        """Verify build_symbol_index is in __all__."""
        exports = self._module_exports("docs.toolchain.build_symbol_index")
        assert "build_symbol_index" in exports

    def test_symbol_delta_in_all(self) -> None:
        """Verify symbol_delta is in __all__."""
        exports = self._module_exports("docs.toolchain.symbol_delta")
        assert "symbol_delta" in exports

    def test_validate_artifacts_in_all(self) -> None:
        """Verify validate_artifacts is in __all__."""
        exports = self._module_exports("docs.toolchain.validate_artifacts")
        assert "validate_artifacts" in exports
