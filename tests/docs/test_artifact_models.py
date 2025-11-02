"""Unit tests for typed artifact models.

Tests cover round-trip serialization, schema compliance, and validation of
symbol index and delta artifacts.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
from docs.types.artifacts import (
    ArtifactValidationError,
    JsonPayload,
    LineSpan,
    SymbolDeltaChange,
    SymbolDeltaPayload,
    SymbolIndexArtifacts,
    SymbolIndexRow,
    symbol_delta_from_json,
    symbol_delta_to_payload,
    symbol_index_from_json,
    symbol_index_to_payload,
)


def _empty_source_link() -> dict[str, str]:
    """Return a fresh, typed source link mapping."""
    return {}


def _empty_reverse_map() -> dict[str, tuple[str, ...]]:
    """Return a fresh, typed reverse lookup mapping."""
    return {}


class TestLineSpan:
    """Tests for LineSpan data class."""

    def test_line_span_creation(self) -> None:
        """Test LineSpan creation with start and end lines."""
        span = LineSpan(start=10, end=20)
        assert span.start == 10
        assert span.end == 20

    def test_line_span_partial(self) -> None:
        """Test LineSpan with None values."""
        span = LineSpan(start=None, end=None)
        assert span.start is None
        assert span.end is None


class TestSymbolIndexRow:
    """Tests for SymbolIndexRow model."""

    def test_minimal_row_creation(self) -> None:
        """Test creating a SymbolIndexRow with minimal fields."""
        row = SymbolIndexRow(
            path="my.module.func",
            kind="function",
            doc="A function",
            tested_by=(),
            source_link=_empty_source_link(),
        )
        assert row.path == "my.module.func"
        assert row.kind == "function"
        assert row.doc == "A function"
        assert row.canonical_path is None

    def test_full_row_creation(self) -> None:
        """Test creating a SymbolIndexRow with all fields."""
        span = LineSpan(start=10, end=25)
        row = SymbolIndexRow(
            path="pkg.mod.Class.method",
            kind="method",
            doc="A method documentation.",
            tested_by=("test_class.py::test_method",),
            source_link={"github": "https://github.com/..."},
            canonical_path="pkg.mod.alias",
            module="pkg.mod",
            package="pkg",
            file="pkg/mod.py",
            span=span,
            signature="(self, x: int) -> str",
            owner="pkg.mod.Class",
            stability="stable",
            since="0.1.0",
            deprecated_in=None,
            section="methods",
            is_async=False,
            is_property=False,
        )
        assert row.path == "pkg.mod.Class.method"
        assert row.canonical_path == "pkg.mod.alias"
        assert row.kind == "method"
        assert row.span == span
        assert row.tested_by == ("test_class.py::test_method",)


class TestSymbolIndexRoundTrip:
    """Tests for SymbolIndexArtifacts round-trip serialization."""

    def test_single_row_round_trip(self) -> None:
        """Test round-trip serialization of single symbol row."""
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="Function documentation.",
            tested_by=("test_pkg.py::test_func",),
            source_link=_empty_source_link(),
            module="pkg",
            package="pkg",
            file="pkg/__init__.py",
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file={"pkg/__init__.py": ("pkg.func",)},
            by_module={"pkg": ("pkg.func",)},
        )

        # Serialize
        payload = symbol_index_to_payload(artifacts)
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["path"] == "pkg.func"

        # Deserialize
        restored = symbol_index_from_json(cast(JsonPayload, payload))
        assert len(restored.rows) == 1
        assert restored.rows[0].path == "pkg.func"
        assert restored.rows[0].kind == "function"

    def test_multiple_rows_round_trip(self) -> None:
        """Test round-trip with multiple rows."""
        rows = [
            SymbolIndexRow(
                path="pkg.mod1.func1",
                kind="function",
                doc="func1 doc",
                tested_by=(),
                source_link=_empty_source_link(),
            ),
            SymbolIndexRow(
                path="pkg.mod2.func2",
                kind="function",
                doc="func2 doc",
                tested_by=(),
                source_link=_empty_source_link(),
            ),
            SymbolIndexRow(
                path="pkg.Class",
                kind="class",
                doc="class doc",
                tested_by=(),
                source_link=_empty_source_link(),
            ),
        ]
        artifacts = SymbolIndexArtifacts(
            rows=tuple(rows),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )

        payload = symbol_index_to_payload(artifacts)
        restored = symbol_index_from_json(cast(JsonPayload, payload))

        assert len(restored.rows) == 3
        paths = [r.path for r in restored.rows]
        assert paths == ["pkg.mod1.func1", "pkg.mod2.func2", "pkg.Class"]

    def test_row_with_span_round_trip(self) -> None:
        """Test round-trip preserves line span information."""
        span = LineSpan(start=42, end=100)
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="doc",
            tested_by=(),
            source_link=_empty_source_link(),
            span=span,
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )

        payload = symbol_index_to_payload(artifacts)
        assert payload[0]["lineno"] == 42
        assert payload[0]["endlineno"] == 100

        restored = symbol_index_from_json(cast(JsonPayload, payload))
        assert restored.rows[0].span is not None
        assert restored.rows[0].span.start == 42
        assert restored.rows[0].span.end == 100

    def test_row_with_tested_by_round_trip(self) -> None:
        """Test round-trip preserves tested_by list."""
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="doc",
            tested_by=("test_mod.py::test_func", "integration_tests.py::test_all"),
            source_link=_empty_source_link(),
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )

        payload = symbol_index_to_payload(artifacts)
        assert set(cast(list[str], payload[0]["tested_by"])) == {
            "test_mod.py::test_func",
            "integration_tests.py::test_all",
        }

        restored = symbol_index_from_json(cast(JsonPayload, payload))
        assert restored.rows[0].tested_by == (
            "test_mod.py::test_func",
            "integration_tests.py::test_all",
        )


class TestSymbolDeltaPayload:
    """Tests for SymbolDeltaPayload model."""

    def test_empty_delta(self) -> None:
        """Test creating empty delta with no changes."""
        delta = SymbolDeltaPayload(
            base_sha="abc123",
            head_sha="def456",
            added=(),
            removed=(),
            changed=(),
        )
        assert delta.base_sha == "abc123"
        assert delta.added == ()
        assert delta.removed == ()
        assert delta.changed == ()

    def test_delta_with_changes(self) -> None:
        """Test delta with added, removed, and changed symbols."""
        change = SymbolDeltaChange(
            path="pkg.mod.func",
            before={"signature": "(x: int)"},
            after={"signature": "(x: int, y: str)"},
            reasons=("signature_changed",),
        )
        delta = SymbolDeltaPayload(
            base_sha="abc123",
            head_sha="def456",
            added=("pkg.mod.new_func",),
            removed=("pkg.old_func",),
            changed=(change,),
        )

        assert delta.added == ("pkg.mod.new_func",)
        assert delta.removed == ("pkg.old_func",)
        assert len(delta.changed) == 1


class TestSymbolDeltaRoundTrip:
    """Tests for SymbolDeltaPayload round-trip serialization."""

    def test_delta_round_trip(self) -> None:
        """Test round-trip serialization of delta."""
        change = SymbolDeltaChange(
            path="pkg.func",
            before={"kind": "function", "signature": "(x)"},
            after={"kind": "function", "signature": "(x, y)"},
            reasons=("signature_changed", "doc_updated"),
        )
        delta = SymbolDeltaPayload(
            base_sha="sha1",
            head_sha="sha2",
            added=("pkg.new_func",),
            removed=("pkg.old_func",),
            changed=(change,),
        )

        # Serialize
        payload = symbol_delta_to_payload(delta)
        assert isinstance(payload, dict)
        assert payload["base_sha"] == "sha1"
        assert payload["head_sha"] == "sha2"
        assert payload["added"] == ["pkg.new_func"]
        assert payload["removed"] == ["pkg.old_func"]
        assert len(cast(list[str], payload["changed"])) == 1

        # Deserialize
        restored = symbol_delta_from_json(cast(JsonPayload, payload))
        assert restored.base_sha == "sha1"
        assert restored.head_sha == "sha2"
        assert restored.added == ("pkg.new_func",)
        assert restored.removed == ("pkg.old_func",)
        assert len(restored.changed) == 1
        assert restored.changed[0].path == "pkg.func"


class TestArtifactValidation:
    """Tests for artifact validation error handling."""

    def test_invalid_symbol_index_not_list(self) -> None:
        """Test validation error when index is not a list."""
        with pytest.raises(ArtifactValidationError) as exc_info:
            symbol_index_from_json({"not": "a list"})
        assert "Expected list" in str(exc_info.value)

    def test_invalid_symbol_index_missing_path(self) -> None:
        """Test validation error when row missing required path field."""
        with pytest.raises(ArtifactValidationError) as exc_info:
            symbol_index_from_json([{"kind": "function"}])
        assert "failed to construct SymbolIndexRow" in str(exc_info.value)

    def test_invalid_delta_not_dict(self) -> None:
        """Test validation error when delta is not a dict."""
        with pytest.raises(ArtifactValidationError) as exc_info:
            symbol_delta_from_json([1, 2, 3])
        assert "Expected dict" in str(exc_info.value)

    def test_valid_empty_index(self) -> None:
        """Test that empty index is valid."""
        artifacts = symbol_index_from_json([])
        assert len(artifacts.rows) == 0

    def test_valid_empty_delta(self) -> None:
        """Test that delta with minimal fields is valid."""
        delta = symbol_delta_from_json({})
        assert delta.base_sha is None
        assert delta.head_sha is None
        assert delta.added == ()
        assert delta.removed == ()
        assert delta.changed == ()


class TestArtifactCoercion:
    """Tests for field coercion during deserialization."""

    def test_tested_by_coercion_from_list(self) -> None:
        """Test that tested_by list is coerced to tuple."""
        payload = [
            {
                "path": "pkg.func",
                "kind": "function",
                "doc": "doc",
                "tested_by": ["test1.py", "test2.py"],
                "source_link": {},
            }
        ]
        artifacts = symbol_index_from_json(cast(JsonPayload, payload))
        assert artifacts.rows[0].tested_by == ("test1.py", "test2.py")

    def test_tested_by_coercion_from_tuple(self) -> None:
        """Test that tested_by tuple stays as tuple."""
        payload = [
            {
                "path": "pkg.func",
                "kind": "function",
                "doc": "doc",
                "tested_by": ("test1.py", "test2.py"),
                "source_link": {},
            }
        ]
        artifacts = symbol_index_from_json(cast(JsonPayload, payload))
        assert artifacts.rows[0].tested_by == ("test1.py", "test2.py")

    def test_tested_by_empty_when_missing(self) -> None:
        """Test that tested_by defaults to empty tuple."""
        payload = [{"path": "pkg.func", "kind": "function", "doc": "doc", "source_link": {}}]
        artifacts = symbol_index_from_json(cast(JsonPayload, payload))
        assert artifacts.rows[0].tested_by == ()

    def test_is_async_coercion(self) -> None:
        """Test that is_async is coerced to bool."""
        payload = [
            {
                "path": "pkg.func",
                "kind": "function",
                "doc": "doc",
                "tested_by": [],
                "source_link": {},
                "is_async": 1,
            }
        ]
        artifacts = symbol_index_from_json(cast(JsonPayload, payload))
        assert artifacts.rows[0].is_async is True

        payload = [
            {
                "path": "pkg.func",
                "kind": "function",
                "doc": "doc",
                "tested_by": [],
                "source_link": {},
                "is_async": 0,
            }
        ]
        artifacts = symbol_index_from_json(cast(JsonPayload, payload))
        assert artifacts.rows[0].is_async is False


class TestArtifactJSONFormatting:
    """Tests for JSON serialization formatting."""

    def test_json_formatting_readable(self) -> None:
        """Test that serialized JSON is formatted for readability."""
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="doc",
            tested_by=(),
            source_link=_empty_source_link(),
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )
        payload = symbol_index_to_payload(artifacts)

        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        lines = json_str.split("\n")
        # Should have indentation
        assert any(line.startswith("  ") for line in lines)

    def test_unicode_handling(self) -> None:
        """Test that unicode characters are preserved."""
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="doc",
            tested_by=(),
            source_link=_empty_source_link(),
            section="Ñoño API",
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )
        payload = symbol_index_to_payload(artifacts)

        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        assert "Ñoño API" in json_str  # Not escaped as \u...
