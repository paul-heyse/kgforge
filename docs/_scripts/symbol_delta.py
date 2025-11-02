"""Compute schema-validated deltas between symbol index snapshots."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import msgspec
from docs._scripts import shared
from docs._scripts.models import (
    JsonValue,
    SymbolDeltaChangeModel,
    SymbolDeltaPayloadModel,
)
from docs._scripts.validation import validate_against_schema
from tools import ToolExecutionError, ToolRunResult, get_logger, run_tool
from tools._shared.problem_details import build_problem_details, render_problem

ENV = shared.detect_environment()
shared.ensure_sys_paths(ENV)
SETTINGS = shared.load_settings()

DOCS_BUILD = SETTINGS.docs_build_dir
SYMBOLS_PATH = DOCS_BUILD / "symbols.json"
DEFAULT_DELTA_PATH = DOCS_BUILD / "symbols.delta.json"
SCHEMA_DIR = ENV.root / "schema" / "docs"
SYMBOL_DELTA_SCHEMA = SCHEMA_DIR / "symbol-delta.schema.json"

BASE_LOGGER = get_logger(__name__)
DELTA_LOG = shared.make_logger("symbol_delta", artifact="symbols.delta.json", logger=BASE_LOGGER)

GIT_TIMEOUT_SECONDS = 30.0
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


@dataclass(frozen=True, slots=True)
class SymbolRow:
    """Normalized symbol row extracted from ``symbols.json``."""

    path: str
    canonical_path: str | None = None
    signature: JsonValue = None
    kind: JsonValue = None
    file: JsonValue = None
    lineno: JsonValue = None
    endlineno: JsonValue = None
    doc: JsonValue = None
    owner: JsonValue = None
    stability: JsonValue = None
    since: JsonValue = None
    deprecated_in: JsonValue = None
    section: JsonValue = None
    package: JsonValue = None
    module: JsonValue = None
    tested_by: JsonValue = None
    is_async: JsonValue = None
    is_property: JsonValue = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, JsonValue]) -> SymbolRow | None:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            return None
        fields: dict[str, JsonValue] = {
            "path": path_value,
        }
        for key in TRACKED_KEYS:
            if key in payload:
                fields[key] = payload[key]
        return cls(**fields)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            key: getattr(self, key)
            for key in ("path", *sorted(TRACKED_KEYS))
            if getattr(self, key) is not None
        }


@dataclass(frozen=True, slots=True)
class ChangeEntry:
    """Difference for a single symbol between two snapshots."""

    path: str
    before: dict[str, JsonValue]
    after: dict[str, JsonValue]
    reasons: tuple[str, ...]

    def to_model(self) -> SymbolDeltaChangeModel:
        return SymbolDeltaChangeModel(
            path=self.path,
            before=dict(self.before),
            after=dict(self.after),
            reasons=list(self.reasons),
        )


@dataclass(frozen=True, slots=True)
class SymbolDeltaPayload:
    """Structured representation of the symbol delta payload."""

    base_sha: str | None
    head_sha: str | None
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[ChangeEntry, ...]

    def to_model(self) -> SymbolDeltaPayloadModel:
        return SymbolDeltaPayloadModel(
            base_sha=self.base_sha,
            head_sha=self.head_sha,
            added=list(self.added),
            removed=list(self.removed),
            changed=[entry.to_model() for entry in self.changed],
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return msgspec.to_builtins(self.to_model(), builtin_types=dict)


def _coerce_json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [cast(JsonValue, _coerce_json_value(item)) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    return str(value)


def _assign_subset(
    row: dict[str, JsonValue],
    payload: Mapping[str, JsonValue],
    keys: Iterable[str],
) -> None:
    for key in keys:
        if key in payload:
            row[key] = _coerce_json_value(payload[key])


def _validate_git_ref(ref: str) -> str:
    candidate = ref.strip()
    if not candidate:
        raise ValueError("Git reference must not be empty")
    if candidate.startswith("-"):
        raise ValueError("Git reference must not begin with '-'")
    return candidate


def _make_symbol_row(payload: Mapping[str, JsonValue]) -> SymbolRow | None:
    return SymbolRow.from_mapping(payload)


def _coerce_symbol_rows(data: object, *, source: str) -> list[SymbolRow]:
    if not isinstance(data, Sequence):
        raise TypeError(f"{source} is not a JSON array")
    rows: list[SymbolRow] = []
    for entry in data:
        if isinstance(entry, Mapping):
            candidate = _make_symbol_row(entry)
            if candidate is not None:
                rows.append(candidate)
    return rows


def _load_symbol_rows(path: Path) -> list[SymbolRow]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return _coerce_symbol_rows(raw, source=str(path))


def _symbols_from_git_blob(blob: str, *, source: str) -> list[SymbolRow]:
    try:
        data: object = json.loads(blob)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{source} does not contain valid JSON") from exc
    return _coerce_symbol_rows(data, source=source)


def _git_rev_parse(ref: str) -> str | None:
    try:
        result: ToolRunResult = run_tool(
            ["git", "rev-parse", ref],
            cwd=ENV.root,
            timeout=GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except ToolExecutionError:
        return None
    sha = result.stdout.strip()
    return sha or None


def _load_base_snapshot(arg: str) -> tuple[list[SymbolRow], str | None]:
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
            cwd=ENV.root,
            timeout=GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except ToolExecutionError:
        DELTA_LOG.warning(
            "No baseline symbols.json at git ref",
            extra={"status": "missing_baseline", "ref": ref},
        )
        return [], _git_rev_parse(ref)

    rows = _symbols_from_git_blob(result.stdout, source=f"git:{ref}")
    return rows, _git_rev_parse(ref)


def _index_rows(rows: list[SymbolRow]) -> dict[str, SymbolRow]:
    indexed: dict[str, SymbolRow] = {}
    for row in rows:
        indexed[row.path] = row
    return indexed


def _diff_rows(
    base: dict[str, SymbolRow],
    head: dict[str, SymbolRow],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[ChangeEntry, ...]]:
    base_paths = set(base)
    head_paths = set(head)

    added = tuple(sorted(head_paths - base_paths))
    removed = tuple(sorted(base_paths - head_paths))

    changes: list[ChangeEntry] = []
    for path in sorted(base_paths & head_paths):
        before_row = base[path]
        after_row = head[path]
        reasons: list[str] = []
        before_subset: dict[str, JsonValue] = {}
        after_subset: dict[str, JsonValue] = {}
        before_payload = before_row.to_dict()
        after_payload = after_row.to_dict()
        for key in sorted(TRACKED_KEYS):
            before_val = before_payload.get(key)
            after_val = after_payload.get(key)
            if before_val != after_val:
                reasons.append(key)
                if key in before_payload:
                    before_subset[key] = _coerce_json_value(before_payload[key])
                if key in after_payload:
                    after_subset[key] = _coerce_json_value(after_payload[key])
        if reasons:
            changes.append(
                ChangeEntry(
                    path=path,
                    before=before_subset,
                    after=after_subset,
                    reasons=tuple(reasons),
                )
            )

    return added, removed, tuple(changes)


def _build_delta(
    *,
    base_rows: list[SymbolRow],
    head_rows: list[SymbolRow],
    base_sha: str | None,
    head_sha: str | None,
) -> SymbolDeltaPayload:
    added, removed, changed = _diff_rows(_index_rows(base_rows), _index_rows(head_rows))
    return SymbolDeltaPayload(
        base_sha=base_sha,
        head_sha=head_sha,
        added=added,
        removed=removed,
        changed=changed,
    )


def write_delta(delta_path: Path, payload: SymbolDeltaPayload) -> bool:
    builtins_payload = payload.to_payload()
    validate_against_schema(
        builtins_payload,
        SYMBOL_DELTA_SCHEMA,
        artifact=delta_path.name,
        struct_type=SymbolDeltaPayloadModel,
    )
    serialized = json.dumps(builtins_payload, indent=2, ensure_ascii=False) + "\n"
    if delta_path.exists():
        existing = delta_path.read_text(encoding="utf-8")
        if existing == serialized:
            DELTA_LOG.info(
                "Delta unchanged",
                extra={"status": "unchanged", "destination": str(delta_path)},
            )
            return False
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(serialized, encoding="utf-8")
    DELTA_LOG.info(
        "Delta written",
        extra={
            "status": "updated",
            "destination": str(delta_path),
            "added": len(payload.added),
            "removed": len(payload.removed),
            "changed": len(payload.changed),
        },
    )
    return True


def _emit_problem(problem: Mapping[str, JsonValue] | None, *, default_message: str) -> None:
    payload = problem or build_problem_details(
        type="https://kgfoundry.dev/problems/docs-symbol-delta",
        title="Symbol delta computation failed",
        status=500,
        detail=default_message,
        instance="urn:docs:symbol-delta:unknown",
    )
    sys.stderr.write(render_problem(payload) + "\n")


@dataclass(slots=True)
class DeltaArgs:
    base: str
    output: str


def parse_args(argv: Sequence[str] | None) -> DeltaArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Git ref or path to the baseline symbols.json snapshot",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DELTA_PATH),
        help="Destination delta file",
    )
    namespace = parser.parse_args(argv)
    return DeltaArgs(base=str(namespace.base), output=str(namespace.output))


def main(argv: Sequence[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    args = parse_args(argv)
    delta_path = Path(args.output)

    if not SYMBOLS_PATH.exists():
        message = f"Missing current snapshot: {SYMBOLS_PATH}"
        _emit_problem(
            build_problem_details(
                type="https://kgfoundry.dev/problems/docs-symbol-delta",
                title="Symbol delta computation failed",
                status=404,
                detail=message,
                instance="urn:docs:symbol-delta:missing-current",
            ),
            default_message=message,
        )
        return 1

    with shared.observe_tool_run(["docs-symbol-delta"], cwd=ENV.root, timeout=None) as observation:
        try:
            head_rows = _load_symbol_rows(SYMBOLS_PATH)
            base_rows, base_sha = _load_base_snapshot(args.base)
            head_sha = _git_rev_parse("HEAD")
            payload = _build_delta(
                base_rows=base_rows,
                head_rows=head_rows,
                base_sha=base_sha,
                head_sha=head_sha,
            )
            write_delta(delta_path, payload)
        except ToolExecutionError as exc:
            observation.failure("failure", returncode=1)
            _emit_problem(exc.problem, default_message=str(exc))
            return 1
        except Exception as exc:  # pragma: no cover - defensive
            observation.failure("exception", returncode=1)
            problem = build_problem_details(
                type="https://kgfoundry.dev/problems/docs-symbol-delta",
                title="Symbol delta computation failed",
                status=500,
                detail=str(exc),
                instance="urn:docs:symbol-delta:unexpected-error",
            )
            _emit_problem(problem, default_message=str(exc))
            return 1
        else:
            observation.success(0)

    DELTA_LOG.info(
        "Delta computation complete",
        extra={
            "status": "success",
            "destination": str(delta_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
