"""Unit tests for typed artifact models.

Tests cover round-trip serialization, schema compliance, and validation of
symbol index and delta artifacts.
"""
# pyright: ignore[reportCallIssue]  # Test fixtures intentionally use keyword args for optional Pydantic fields
# This file uses SymbolIndexRow, SymbolDeltaChange, etc. with minimal required arguments,
# relying on Pydantic field defaults for optional parameters. Pyright's strict mode flags this
# as missing arguments, but Pydantic's Field(...) with defaults handles it correctly.
#
# To verify: all fields are properly typed with defaults in docs/_types/artifacts.py:
# - tested_by: tuple[str, ...] = ()
# - source_link: dict[str, str] = Field(default_factory=dict)
# - canonical_path, module, package, file, etc.: str | None = Field(None, ...)
# - is_async, is_property: bool = Field(False, ...)
#
# The ignore applies to the entire module since these patterns are intentional throughout.

from __future__ import annotations

import json
from typing import cast

import pytest
from docs.types.artifacts import (
    SYMBOL_INDEX_ROW_FIELDS,
    ArtifactValidationError,
    JsonPayload,
    JsonValue,
    LineSpan,
    SymbolDeltaChange,
    SymbolDeltaPayload,
    SymbolIndexArtifacts,
    SymbolIndexRow,
    align_schema_fields,
    symbol_delta_from_json,
    symbol_delta_to_payload,
    symbol_index_from_json,
    symbol_index_to_payload,
)

from kgfoundry_common.errors import (
    ArtifactDeserializationError,
)

type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]
type JsonObjectArray = list[JsonObject]


def _empty_source_link() -> dict[str, str]:
    """Return a fresh, typed source link mapping."""
    return {}


def _source_link(**pairs: str) -> dict[str, str]:
    """Return a typed source link mapping populated with provided pairs."""
    return dict(pairs)


def _empty_reverse_map() -> dict[str, tuple[str, ...]]:
    """Return a fresh, typed reverse lookup mapping."""
    return {}


def _symbol_row(
    path: str = "pkg.func",
    kind: str = "function",
    doc: str = "doc",
    *,
    tested_by: tuple[str, ...] | list[str] = (),
    source_link: dict[str, str] | None = None,
    canonical_path: str | None = None,
    module: str | None = None,
    package: str | None = None,
    file: str | None = None,
    span: LineSpan | None = None,
    signature: str | None = None,
    owner: str | None = None,
    stability: str | None = None,
    since: str | None = None,
    deprecated_in: str | None = None,
    section: str | None = None,
    is_async: bool = False,
    is_property: bool = False,
) -> SymbolIndexRow:
    """Create a SymbolIndexRow with sensible defaults for testing.

    This helper ensures all required fields are provided and Pyright
    type checking passes with complete type safety.

    Parameters
    ----------
    path : str, optional
        Symbol path. Defaults to "pkg.func".
    kind : str, optional
        Symbol kind. Defaults to "function".
    doc : str, optional
        Documentation. Defaults to "doc".
    tested_by : tuple[str, ...] | list[str], optional
        Test paths. Defaults to ().
    source_link : dict[str, str] | None, optional
        Source links. Defaults to None (converted to {}).
    canonical_path : str | None, optional
        Canonical path if alias. Defaults to None.
    module : str | None, optional
        Module name. Defaults to None.
    package : str | None, optional
        Package name. Defaults to None.
    file : str | None, optional
        File path. Defaults to None.
    span : LineSpan | None, optional
        Line span. Defaults to None.
    signature : str | None, optional
        Function signature. Defaults to None.
    owner : str | None, optional
        Owner class. Defaults to None.
    stability : str | None, optional
        Stability tag. Defaults to None.
    since : str | None, optional
        Version introduced. Defaults to None.
    deprecated_in : str | None, optional
        Deprecation version. Defaults to None.
    section : str | None, optional
        Documentation section. Defaults to None.
    is_async : bool, optional
        Is async function. Defaults to False.
    is_property : bool, optional
        Is property. Defaults to False.

    Returns
    -------
    SymbolIndexRow
        A properly constructed symbol index row.
    """
    return SymbolIndexRow(
        path=path,
        kind=kind,
        doc=doc,
        tested_by=tuple(tested_by) if isinstance(tested_by, list) else tested_by,
        source_link=source_link or _empty_source_link(),
        canonical_path=canonical_path,
        module=module,
        package=package,
        file=file,
        span=span,
        signature=signature,
        owner=owner,
        stability=stability,
        since=since,
        deprecated_in=deprecated_in,
        section=section,
        is_async=is_async,
        is_property=is_property,
    )


def _symbol_payload(**overrides: JsonValue) -> dict[str, JsonValue]:
    """Construct a canonical symbol payload with optional overrides."""
    payload: dict[str, JsonValue] = {
        "path": "pkg.func",
        "kind": "function",
        "doc": "doc",
        "tested_by": [],
        "source_link": cast(JsonValue, _empty_source_link()),
    }
    payload.update(overrides)
    return payload


def expect_json_string(value: JsonValue) -> str:
    if not isinstance(value, str):
        message = f"Expected str JSON value, got {type(value)!r}"
        raise TypeError(message)
    return value


def expect_json_int(value: JsonValue) -> int:
    if not isinstance(value, int):
        message = f"Expected int JSON value, got {type(value)!r}"
        raise TypeError(message)
    return value


def expect_json_mapping(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        message = f"Expected mapping JSON value, got {type(value)!r}"
        raise TypeError(message)
    return value


def expect_json_sequence(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        message = f"Expected sequence JSON value, got {type(value)!r}"
        raise TypeError(message)
    return value


def to_json_payload(rows: JsonObjectArray) -> JsonPayload:
    return [cast(JsonValue, row) for row in rows]


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
        row = _symbol_row(
            path="my.module.func",
            kind="function",
            doc="A function",
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
            source_link=_source_link(github="https://github.com/..."),
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
        row = _symbol_row(
            path="pkg.func",
            kind="function",
            doc="Function documentation.",
            tested_by=("test_pkg.py::test_func",),
            module="pkg",
            package="pkg",
            file="pkg/__init__.py",
        )
        by_file: dict[str, tuple[str, ...]] = {"pkg/__init__.py": ("pkg.func",)}
        by_module: dict[str, tuple[str, ...]] = {"pkg": ("pkg.func",)}
        artifacts = SymbolIndexArtifacts(rows=(row,), by_file=by_file, by_module=by_module)

        # Serialize
        payload_rows: JsonObjectArray = symbol_index_to_payload(artifacts)
        assert len(payload_rows) == 1
        first_row = expect_json_mapping(payload_rows[0])
        path_value = expect_json_string(first_row["path"])
        assert path_value == "pkg.func"

        # Deserialize
        restored = symbol_index_from_json(to_json_payload(payload_rows))
        assert len(restored.rows) == 1
        assert restored.rows[0].path == "pkg.func"
        assert restored.rows[0].kind == "function"

    def test_multiple_rows_round_trip(self) -> None:
        """Test round-trip with multiple rows."""
        rows: list[SymbolIndexRow] = [
            _symbol_row(
                path="pkg.mod1.func1",
                kind="function",
                doc="func1 doc",
            ),
            _symbol_row(
                path="pkg.mod2.func2",
                kind="function",
                doc="func2 doc",
            ),
            _symbol_row(
                path="pkg.Class",
                kind="class",
                doc="class doc",
            ),
        ]
        rows_tuple: tuple[SymbolIndexRow, ...] = tuple(rows)
        artifacts = SymbolIndexArtifacts(
            rows=rows_tuple,
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )

        payload_rows: JsonObjectArray = symbol_index_to_payload(artifacts)
        restored = symbol_index_from_json(to_json_payload(payload_rows))

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

        payload_rows: JsonObjectArray = symbol_index_to_payload(artifacts)
        first_row = expect_json_mapping(payload_rows[0])
        assert expect_json_int(first_row["lineno"]) == 42
        assert expect_json_int(first_row["endlineno"]) == 100

        restored = symbol_index_from_json(to_json_payload(payload_rows))
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

        payload_rows: JsonObjectArray = symbol_index_to_payload(artifacts)
        tested_by_value = expect_json_sequence(payload_rows[0]["tested_by"])
        assert {expect_json_string(entry) for entry in tested_by_value} == {
            "test_mod.py::test_func",
            "integration_tests.py::test_all",
        }

        restored = symbol_index_from_json(to_json_payload(payload_rows))
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
        before_payload: dict[str, JsonValue] = {"signature": "(x: int)"}
        after_payload: dict[str, JsonValue] = {"signature": "(x: int, y: str)"}
        change = SymbolDeltaChange(
            path="pkg.mod.func",
            before=before_payload,
            after=after_payload,
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
        before_payload: dict[str, JsonValue] = {"kind": "function", "signature": "(x)"}
        after_payload: dict[str, JsonValue] = {
            "kind": "function",
            "signature": "(x, y)",
        }
        change = SymbolDeltaChange(
            path="pkg.func",
            before=before_payload,
            after=after_payload,
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
        payload_raw = symbol_delta_to_payload(delta)
        assert isinstance(payload_raw, dict)
        assert payload_raw["base_sha"] == "sha1"
        assert payload_raw["head_sha"] == "sha2"
        assert payload_raw["added"] == ["pkg.new_func"]
        assert payload_raw["removed"] == ["pkg.old_func"]
        changed_entries_value = expect_json_sequence(payload_raw["changed"])
        changed_entries = [expect_json_mapping(entry) for entry in changed_entries_value]
        assert len(changed_entries) == len(changed_entries_value)
        assert len(changed_entries) == 1

        # Deserialize
        restored = symbol_delta_from_json(payload_raw)
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
        with pytest.raises(ArtifactDeserializationError) as exc_info:
            symbol_index_from_json({"not": "a list"})
        assert "Expected list" in str(exc_info.value)

    def test_invalid_symbol_index_missing_path(self) -> None:
        """Test validation error when row missing required path field."""
        with pytest.raises(ArtifactValidationError) as exc_info:
            symbol_index_from_json([{"kind": "function"}])
        assert "path" in str(exc_info.value).lower()

    def test_invalid_delta_not_dict(self) -> None:
        """Test validation error when delta is not a dict."""
        with pytest.raises(ArtifactDeserializationError) as exc_info:
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
        artifacts = symbol_index_from_json(
            to_json_payload(
                [
                    _symbol_payload(tested_by=cast(JsonValue, ["test1.py", "test2.py"])),
                ]
            ),
        )
        assert artifacts.rows[0].tested_by == ("test1.py", "test2.py")

    def test_tested_by_coercion_from_tuple(self) -> None:
        """Test that tested_by tuple stays as tuple."""
        tuple_value = cast(JsonValue, ("test1.py", "test2.py"))
        artifacts = symbol_index_from_json(
            to_json_payload([_symbol_payload(tested_by=tuple_value)]),
        )
        assert artifacts.rows[0].tested_by == ("test1.py", "test2.py")

    def test_tested_by_empty_when_missing(self) -> None:
        """Test that tested_by defaults to empty tuple."""
        artifacts = symbol_index_from_json(to_json_payload([_symbol_payload()]))
        assert artifacts.rows[0].tested_by == ()

    def test_is_async_coercion(self) -> None:
        """Test that is_async is coerced to bool."""
        artifacts = symbol_index_from_json(
            to_json_payload([_symbol_payload(is_async=cast(JsonValue, 1))]),
        )
        assert artifacts.rows[0].is_async is True

        artifacts = symbol_index_from_json(
            to_json_payload([_symbol_payload(is_async=cast(JsonValue, 0))]),
        )
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
        payload_raw = symbol_index_to_payload(artifacts)
        json_str = json.dumps(payload_raw, indent=2, ensure_ascii=False)
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
        payload_raw = symbol_index_to_payload(artifacts)
        json_str = json.dumps(payload_raw, indent=2, ensure_ascii=False)
        assert "Ñoño API" in json_str  # Not escaped as \u...


class TestCanonicalConstructors:
    """Tests for canonical keyword-based constructor usage with alignment.

    These tests verify that constructors are invoked with canonical field names
    matching the JSON Schema, and that alignment helpers properly validate
    payloads before model construction.
    """

    def test_canonical_constructor_with_all_fields(self) -> None:
        """Test SymbolIndexRow construction with canonical field names."""
        span = LineSpan(start=10, end=25)
        row = SymbolIndexRow(
            path="pkg.mod.Class.method",
            kind="method",
            doc="A method documentation.",
            tested_by=("test_class.py::test_method",),
            source_link=_source_link(github="https://github.com/..."),
            canonical_path="pkg.mod.alias",
            module="pkg.mod",
            package="pkg",
            file="pkg/mod.py",
            span=span,
            signature="(self, x: int) -> str",
            owner="pkg.mod.Class",
            stability="stable",
            since="0.1.0",
            deprecated_in="0.2.0",  # canonical field name
            section="methods",
            is_async=False,
            is_property=False,
        )

        # Verify all fields are correctly set
        assert row.path == "pkg.mod.Class.method"
        assert row.deprecated_in == "0.2.0"
        assert row.stability == "stable"
        assert row.span == span

    def test_canonical_constructor_serialization_roundtrip(self) -> None:
        """Test that canonical constructor output round-trips through JSON."""
        row = SymbolIndexRow(
            path="pkg.func",
            kind="function",
            doc="A function.",
            tested_by=(),
            source_link=_empty_source_link(),
            deprecated_in="0.2.0",
        )
        artifacts = SymbolIndexArtifacts(
            rows=(row,),
            by_file=_empty_reverse_map(),
            by_module=_empty_reverse_map(),
        )

        # Serialize to payload
        payload = symbol_index_to_payload(artifacts)
        assert len(payload) == 1
        first_row = expect_json_mapping(payload[0])
        assert first_row["path"] == "pkg.func"
        assert first_row["deprecated_in"] == "0.2.0"

        # Deserialize and verify fidelity
        restored = symbol_index_from_json(cast(JsonPayload, payload))
        assert restored.rows[0].path == "pkg.func"
        assert restored.rows[0].deprecated_in == "0.2.0"

    def test_alignment_helper_with_canonical_payload(self) -> None:
        """Test align_schema_fields with valid canonical payload."""
        payload = {
            "path": "pkg.func",
            "kind": "function",
            "doc": "A function",
            "deprecated_in": "0.2.0",
            "tested_by": [],
            "source_link": {},
        }

        # Validate with alignment helper
        aligned = align_schema_fields(
            payload,
            expected_fields=SYMBOL_INDEX_ROW_FIELDS,
            artifact_id="symbol-index-row",
        )

        # Construct model from aligned payload using model_validate for type safety
        row = SymbolIndexRow.model_validate(aligned)
        assert row.path == "pkg.func"
        assert row.deprecated_in == "0.2.0"

    def test_constructor_with_minimal_fields(self) -> None:
        """Test constructor with only required fields."""
        row = _symbol_row(
            path="pkg.func",
            kind="function",
            doc="A function",
        )

        assert row.path == "pkg.func"
        assert row.canonical_path is None
        assert row.stability is None
        assert row.deprecated_in is None

    def test_constructor_field_order_irrelevant(self) -> None:
        """Test that keyword argument order does not matter."""
        row1 = _symbol_row(
            path="pkg.func",
            kind="function",
            doc="doc",
        )

        # Different order
        row2 = _symbol_row(
            doc="doc",
            kind="function",
            path="pkg.func",
        )

        assert row1.path == row2.path
        assert row1.kind == row2.kind
        assert row1.doc == row2.doc

    def test_schema_roundtrip_preserves_canonical_casing(self) -> None:
        """Verify schema round-trip preserves canonical field casing."""
        original_payload = [
            {
                "path": "pkg.func",
                "kind": "function",
                "doc": "A function",
                "deprecated_in": "0.2.0",
                "tested_by": [],
                "source_link": {},
            }
        ]

        # Deserialize
        artifacts = symbol_index_from_json(cast(JsonPayload, original_payload))

        # Serialize back
        output_payload = symbol_index_to_payload(artifacts)

        # Verify canonical field names are preserved
        assert isinstance(output_payload[0], dict)
        first_row = output_payload[0]
        assert "deprecated_in" in first_row
        assert first_row["deprecated_in"] == "0.2.0"

    def test_alignment_blocks_unknown_fields(self) -> None:
        """Verify alignment helper rejects payloads with unknown fields."""
        payload = {
            "path": "pkg.func",
            "kind": "function",
            "doc": "A function",
            "unknown_field": "invalid",
            "tested_by": [],
            "source_link": {},
        }

        with pytest.raises(ArtifactValidationError) as exc_info:
            align_schema_fields(
                payload,
                expected_fields=SYMBOL_INDEX_ROW_FIELDS,
                artifact_id="symbol-index-row",
            )

        error = exc_info.value
        assert "unknown_field" in str(error)
        assert error.context is not None
        assert "unknown_fields" in error.context
