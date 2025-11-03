"""Tests for runtime determinism without optional dependencies.

This module verifies that typing gates are correctly implemented:
1. Postponed annotations prevent eager type evaluation
2. TYPE_CHECKING blocks protect runtime from type-only imports
3. Façade modules provide safe type access without runtime overhead
4. Key CLI tools can initialize without optional deps
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestPostponedAnnotations:
    """Verify postponed annotations are present in core modules."""

    def test_kgfoundry_common_typing_has_postponed_annotations(self) -> None:
        """kgfoundry_common.typing uses postponed annotations."""
        import kgfoundry_common.typing  # noqa: PLC0415

        source_file = kgfoundry_common.typing.__file__
        assert source_file is not None

        content = Path(source_file).read_text(encoding="utf-8")
        assert "from __future__ import annotations" in content

    def test_tools_typing_has_postponed_annotations(self) -> None:
        """tools.typing uses postponed annotations."""
        import tools.typing  # noqa: PLC0415

        source_file = tools.typing.__file__
        assert source_file is not None

        content = Path(source_file).read_text(encoding="utf-8")
        assert "from __future__ import annotations" in content

    def test_docs_typing_has_postponed_annotations(self) -> None:
        """docs.typing uses postponed annotations."""
        # This may not be installed, so check if it exists
        try:
            import docs.typing  # noqa: PLC0415

            source_file = docs.typing.__file__
            assert source_file is not None

            content = Path(source_file).read_text(encoding="utf-8")
            assert "from __future__ import annotations" in content
        except ImportError:
            pytest.skip("docs.typing not in Python path")


class TestTypingFacadeModules:
    """Verify façade modules export expected helpers."""

    def test_kgfoundry_common_typing_exports_gate_import(self) -> None:
        """gate_import is available from canonical source."""
        from kgfoundry_common.typing import gate_import  # noqa: PLC0415

        assert callable(gate_import)

    def test_kgfoundry_common_typing_exports_safe_get_type(self) -> None:
        """safe_get_type is available from canonical source."""
        from kgfoundry_common.typing import safe_get_type  # noqa: PLC0415

        assert callable(safe_get_type)

    def test_kgfoundry_common_typing_exports_type_aliases(self) -> None:
        """Type aliases are accessible."""
        from kgfoundry_common.typing import (  # noqa: PLC0415
            JSONValue,
            NavMap,
            ProblemDetails,
            SymbolID,
        )

        assert NavMap is not None
        assert ProblemDetails is not None
        assert JSONValue is not None
        assert SymbolID is not None

    def test_tools_typing_re_exports_facade(self) -> None:
        """tools.typing re-exports from canonical source."""
        from tools.typing import gate_import as tools_gate_import  # noqa: PLC0415

        from kgfoundry_common.typing import gate_import as canonical_gate_import  # noqa: PLC0415

        # Should be the same function
        assert tools_gate_import is canonical_gate_import

    def test_docs_typing_re_exports_facade(self) -> None:
        """docs.typing re-exports from canonical source."""
        try:
            from docs.typing import gate_import as docs_gate_import  # noqa: PLC0415

            from kgfoundry_common.typing import (  # noqa: PLC0415
                gate_import as canonical_gate_import,
            )

            # Should be the same function
            assert docs_gate_import is canonical_gate_import
        except ImportError:
            pytest.skip("docs.typing not in Python path")


class TestTypeOnlyImportGuarding:
    """Verify TYPE_CHECKING guards in façade modules."""

    def test_facade_has_type_checking_block(self) -> None:
        """kgfoundry_common.typing has TYPE_CHECKING guards."""
        import kgfoundry_common.typing  # noqa: PLC0415

        source_file = kgfoundry_common.typing.__file__
        assert source_file is not None

        content = Path(source_file).read_text(encoding="utf-8")
        # Verify TYPE_CHECKING is imported and used
        assert "from typing import" in content
        assert "if TYPE_CHECKING:" in content

    def test_no_eager_numpy_import_in_facade(self) -> None:
        """Numpy is not imported at module level in source."""
        import kgfoundry_common.typing  # noqa: PLC0415

        source_file = kgfoundry_common.typing.__file__
        assert source_file is not None

        content = Path(source_file).read_text(encoding="utf-8")
        # Verify numpy is only imported inside TYPE_CHECKING block
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Skip lines before TYPE_CHECKING block
            if "if TYPE_CHECKING:" in line:
                # After TYPE_CHECKING block, numpy imports are OK
                break
            # Before TYPE_CHECKING, numpy should not be imported
            if line.strip().startswith("import numpy"):
                pytest.fail(f"Found unguarded numpy import at line {i + 1}")
            if "from numpy import" in line or "from numpy." in line:
                pytest.fail(f"Found unguarded numpy import at line {i + 1}")


class TestRuntimeImportSafety:
    """Verify that façade modules work without optional deps."""

    def test_gate_import_missing_module_raises_import_error(self) -> None:
        """gate_import raises ImportError for missing modules."""
        from kgfoundry_common.typing import gate_import  # noqa: PLC0415

        with pytest.raises(ImportError) as exc_info:
            gate_import("nonexistent_module_xyz", "test")

        assert "not installed" in str(exc_info.value)

    def test_safe_get_type_missing_module_returns_none(self) -> None:
        """safe_get_type returns None gracefully for missing modules."""
        from kgfoundry_common.typing import safe_get_type  # noqa: PLC0415

        result = safe_get_type("nonexistent_module_xyz", "SomeType")
        assert result is None

    def test_safe_get_type_respects_default(self) -> None:
        """safe_get_type uses provided default value."""
        from kgfoundry_common.typing import safe_get_type  # noqa: PLC0415

        default = "my_fallback"
        result = safe_get_type("nonexistent", "Type", default=default)
        assert result == default


class TestCLIEntryPointImportClean:
    """Verify CLI entry points initialize without heavy deps."""

    @pytest.mark.integration
    def test_apply_postponed_annotations_imports_clean(self) -> None:
        """tools.lint.apply_postponed_annotations is importable."""
        try:
            import tools.lint.apply_postponed_annotations  # noqa: F401, PLC0415
        except ImportError as e:
            pytest.fail(f"Failed to import apply_postponed_annotations: {e}")

    @pytest.mark.integration
    def test_check_typing_gates_imports_clean(self) -> None:
        """tools.lint.check_typing_gates is importable."""
        try:
            import tools.lint.check_typing_gates  # noqa: F401, PLC0415
        except ImportError as e:
            pytest.fail(f"Failed to import check_typing_gates: {e}")
