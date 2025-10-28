#!/usr/bin/env python3
"""Compute the delta between symbol index snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS_BUILD = ROOT / "docs" / "_build"
SYMBOLS_PATH = DOCS_BUILD / "symbols.json"
DELTA_PATH = DOCS_BUILD / "symbols.delta.json"
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


def _read_json(path: Path) -> Any:
    """Load JSON data from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def _git_rev_parse(ref: str) -> str | None:
    """Return ``git rev-parse`` for ``ref`` if possible."""
    try:
        return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover - detached or non-git environments
        return None


def _load_base_snapshot(arg: str) -> tuple[list[dict[str, Any]], str | None]:
    """Return the base snapshot rows and resolved SHA from ``arg``."""
    candidate = Path(arg)
    if candidate.exists():
        data = _read_json(candidate)
        if not isinstance(data, list):
            raise SystemExit(f"Base snapshot at {candidate} is not a JSON array")
        return data, _git_rev_parse("HEAD")

    try:
        blob = subprocess.check_output(
            ["git", "show", f"{arg}:docs/_build/symbols.json"],
            cwd=ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        print(
            f"No symbols.json found at git ref '{arg}', assuming empty baseline.",
            file=sys.stderr,
        )
        return [], _git_rev_parse(arg)

    data = json.loads(blob)
    if not isinstance(data, list):
        raise SystemExit(f"Git ref '{arg}' does not contain a JSON array for symbols.json")
    return data, _git_rev_parse(arg)


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index rows by their ``path`` field."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = row.get("path")
        if isinstance(path, str):
            indexed[path] = row
    return indexed


def _diff_rows(
    base: dict[str, dict[str, Any]], head: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return (added, removed, changed) deltas between ``base`` and ``head`` maps."""
    base_paths = set(base)
    head_paths = set(head)

    added = sorted(head_paths - base_paths)
    removed = sorted(base_paths - head_paths)

    changed: list[dict[str, Any]] = []
    for path in sorted(base_paths & head_paths):
        before = base[path]
        after = head[path]
        reasons: list[str] = []
        before_subset: dict[str, Any] = {}
        after_subset: dict[str, Any] = {}
        for key in sorted(TRACKED_KEYS):
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val != after_val:
                reasons.append(key)
                before_subset[key] = before_val
                after_subset[key] = after_val
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


def _write_delta(delta: dict[str, Any]) -> None:
    """Write the delta file if it changed."""
    serialized = json.dumps(delta, indent=2, ensure_ascii=False) + "\n"
    if DELTA_PATH.exists():
        existing = DELTA_PATH.read_text(encoding="utf-8")
        if existing == serialized:
            print(f"Unchanged {DELTA_PATH}")
            return
    DELTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DELTA_PATH.write_text(serialized, encoding="utf-8")
    print(f"Updated {DELTA_PATH}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    global DELTA_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Git ref or path to the baseline symbols.json snapshot",
    )
    parser.add_argument(
        "--output",
        default=str(DELTA_PATH),
        help="Override the destination delta file",
    )
    args = parser.parse_args(argv)

    DELTA_PATH = Path(args.output)

    if not SYMBOLS_PATH.exists():
        raise SystemExit(f"Missing current snapshot: {SYMBOLS_PATH}")

    head_rows = _read_json(SYMBOLS_PATH)
    if not isinstance(head_rows, list):
        raise SystemExit(f"{SYMBOLS_PATH} is not a JSON array")

    base_rows, base_sha = _load_base_snapshot(args.base)
    head_sha = _git_rev_parse("HEAD")

    added, removed, changed = _diff_rows(_index_rows(base_rows), _index_rows(head_rows))

    delta = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "added": added,
        "removed": removed,
        "changed": changed,
    }

    _write_delta(delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
