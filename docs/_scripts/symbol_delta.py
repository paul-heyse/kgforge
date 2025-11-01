#!/usr/bin/env python3
"""Overview of symbol delta.

This module bundles symbol delta logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from tools import ToolExecutionError, ToolRunResult, get_logger, run_tool, with_fields

ROOT = Path(__file__).resolve().parents[2]
DOCS_BUILD = ROOT / "docs" / "_build"
SYMBOLS_PATH = DOCS_BUILD / "symbols.json"
TRACKED_KEYS = {
    "canonical_path",
    "signature",
    "kind",
    "file",
    "lineno",
    "endlineno",
    "doc",
    "owner",
    "stability",
    "since",
    "deprecated_in",
    "section",
    "package",
    "module",
    "tested_by",
    "is_async",
    "is_property",
}

DEFAULT_DELTA_PATH = DOCS_BUILD / "symbols.delta.json"

LOGGER = get_logger(__name__)
LOG = with_fields(LOGGER, operation="symbol_delta")

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


class SymbolRow(TypedDict, total=False):
    """Typed snapshot of a symbol entry."""

    path: str
    canonical_path: str | None
    signature: str | None
    kind: str | None
    file: str | None
    lineno: int | None
    endlineno: int | None
    doc: str | None
    owner: str | None
    stability: str | None
    since: str | None
    deprecated_in: str | None
    section: str | None
    package: str | None
    module: str | None
    tested_by: list[str] | None
    is_async: bool | None
    is_property: bool | None


class ChangeEntry(TypedDict):
    """Typed representation of a change for a symbol."""

    path: str
    before: dict[str, JSONValue]
    after: dict[str, JSONValue]
    reasons: list[str]


class SymbolDeltaPayload(TypedDict):
    """Top-level delta payload written to disk."""

    base_sha: str | None
    head_sha: str | None
    added: list[str]
    removed: list[str]
    changed: list[ChangeEntry]


_REF_PATTERN = re.compile(r"^[A-Za-z0-9._\-/]+$")
_STR_FIELDS: tuple[str, ...] = (
    "canonical_path",
    "signature",
    "kind",
    "file",
    "doc",
    "owner",
    "stability",
    "since",
    "deprecated_in",
    "section",
    "package",
    "module",
)
_INT_FIELDS: tuple[str, ...] = ("lineno", "endlineno")
_BOOL_FIELDS: tuple[str, ...] = ("is_async", "is_property")


def _coerce_json_value(value: object) -> JSONValue:
    """Return ``value`` as a JSON-compatible representation."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    return str(value)


def _assign_str_fields(
    row: dict[str, JSONValue], payload: Mapping[str, JSONValue], keys: Sequence[str]
) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            row[key] = value


def _assign_int_fields(
    row: dict[str, JSONValue], payload: Mapping[str, JSONValue], keys: Sequence[str]
) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            row[key] = value
        elif isinstance(value, float):
            row[key] = int(value)


def _assign_bool_fields(
    row: dict[str, JSONValue], payload: Mapping[str, JSONValue], keys: Sequence[str]
) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            row[key] = value


def _validate_git_ref(ref: str) -> str:
    """Validate ``ref`` to avoid shell injection in git commands."""
    candidate = ref.strip()
    if not candidate:
        message = "Git reference must not be empty"
        raise ValueError(message)
    if candidate != ref:
        message = "Git reference may not contain leading or trailing whitespace"
        raise ValueError(message)
    if candidate.startswith("-"):
        message = "Git reference must not start with '-'"
        raise ValueError(message)
    if not _REF_PATTERN.fullmatch(candidate):
        message = f"Git reference '{ref}' contains invalid characters"
        raise ValueError(message)
    return candidate


def _make_symbol_row(payload: Mapping[str, JSONValue]) -> SymbolRow | None:
    """Convert a generic mapping into a :class:`SymbolRow` when possible."""
    path_value = payload.get("path")
    if not isinstance(path_value, str):
        return None

    row_dict: dict[str, JSONValue] = {"path": path_value}

    _assign_str_fields(row_dict, payload, _STR_FIELDS)
    _assign_int_fields(row_dict, payload, _INT_FIELDS)
    _assign_bool_fields(row_dict, payload, _BOOL_FIELDS)

    tested_by_value = payload.get("tested_by")
    if isinstance(tested_by_value, list):
        row_dict["tested_by"] = [str(item) for item in tested_by_value]
    elif isinstance(tested_by_value, str):
        row_dict["tested_by"] = [tested_by_value]

    return cast(SymbolRow, row_dict)


def _coerce_symbol_rows(data: object, *, source: str) -> list[SymbolRow]:
    """Validate raw JSON payloads into a list of :class:`SymbolRow` objects."""
    if not isinstance(data, list):
        message = f"{source} is not a JSON array"
        raise TypeError(message)

    rows: list[SymbolRow] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        candidate = _make_symbol_row(entry)
        if candidate is not None:
            rows.append(candidate)
    return rows


def _load_symbol_rows(path: Path) -> list[SymbolRow]:
    """Read ``path`` as UTF-8 JSON and return rows."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return _coerce_symbol_rows(raw, source=str(path))


def _symbols_from_git_blob(blob: str, *, source: str) -> list[SymbolRow]:
    """Parse git blob content into :class:`SymbolRow` entries."""
    try:
        data: object = json.loads(blob)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        message = f"{source} does not contain valid JSON"
        raise ValueError(message) from exc
    return _coerce_symbol_rows(data, source=source)


def _git_rev_parse(ref: str) -> str | None:
    """Return ``git rev-parse`` for ``ref`` if possible."""
    try:
        result: ToolRunResult = run_tool(["git", "rev-parse", ref], cwd=ROOT, check=True)
    except ToolExecutionError:
        return None
    sha = result.stdout.strip()
    return sha or None


def _load_base_snapshot(arg: str) -> tuple[list[SymbolRow], str | None]:
    """Return the base snapshot rows and resolved SHA from ``arg``."""
    candidate = Path(arg)
    if candidate.exists():
        return _load_symbol_rows(candidate), _git_rev_parse("HEAD")

    try:
        ref = _validate_git_ref(arg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        result: ToolRunResult = run_tool(
            ["git", "show", f"{ref}:docs/_build/symbols.json"],
            cwd=ROOT,
            check=True,
        )
    except ToolExecutionError:
        LOG.warning(
            "No baseline symbols.json at git ref",
            extra={
                "status": "missing_baseline",
                "ref": ref,
            },
        )
        return [], _git_rev_parse(ref)

    rows = _symbols_from_git_blob(result.stdout, source=f"git:{ref}")
    return rows, _git_rev_parse(ref)


def _index_rows(rows: list[SymbolRow]) -> dict[str, SymbolRow]:
    """Index rows by their ``path`` field."""
    indexed: dict[str, SymbolRow] = {}
    for row in rows:
        path = row.get("path")
        if isinstance(path, str):
            indexed[path] = row
    return indexed


def _diff_rows(
    base: dict[str, SymbolRow], head: dict[str, SymbolRow]
) -> tuple[list[str], list[str], list[ChangeEntry]]:
    """Return (added, removed, changed) deltas between ``base`` and ``head`` maps."""
    base_paths = set(base)
    head_paths = set(head)

    added = sorted(head_paths - base_paths)
    removed = sorted(base_paths - head_paths)

    changed: list[ChangeEntry] = []
    for path in sorted(base_paths & head_paths):
        before = base[path]
        after = head[path]
        reasons: list[str] = []
        before_subset: dict[str, JSONValue] = {}
        after_subset: dict[str, JSONValue] = {}
        for key in sorted(TRACKED_KEYS):
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val != after_val:
                reasons.append(key)
                before_subset[key] = _coerce_json_value(before_val)
                after_subset[key] = _coerce_json_value(after_val)
        if reasons:
            changed.append(
                {
                    "path": path,
                    "before": before_subset,
                    "after": after_subset,
                    "reasons": reasons,
                }
            )

    return added, removed, changed


def _write_delta(delta_path: Path, payload: SymbolDeltaPayload) -> None:
    """Write the delta file if it changed."""
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if delta_path.exists():
        existing = delta_path.read_text(encoding="utf-8")
        if existing == serialized:
            LOG.info(
                "Delta unchanged",
                extra={
                    "status": "unchanged",
                    "destination": str(delta_path),
                },
            )
            return
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(serialized, encoding="utf-8")
    LOG.info(
        "Delta written",
        extra={
            "status": "updated",
            "destination": str(delta_path),
            "added": len(payload["added"]),
            "removed": len(payload["removed"]),
            "changed": len(payload["changed"]),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Render a symbol delta report comparing baseline and current snapshots."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Git ref or path to the baseline symbols.json snapshot",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DELTA_PATH),
        help="Override the destination delta file",
    )
    args = parser.parse_args(argv)

    delta_path = Path(args.output)

    if not SYMBOLS_PATH.exists():
        message = f"Missing current snapshot: {SYMBOLS_PATH}"
        raise SystemExit(message)

    try:
        head_rows = _load_symbol_rows(SYMBOLS_PATH)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_rows, base_sha = _load_base_snapshot(args.base)
    head_sha = _git_rev_parse("HEAD")

    added, removed, changed = _diff_rows(_index_rows(base_rows), _index_rows(head_rows))

    delta: SymbolDeltaPayload = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "added": added,
        "removed": removed,
        "changed": changed,
    }

    _write_delta(delta_path, delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
