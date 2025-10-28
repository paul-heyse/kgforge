#!/usr/bin/env python3
"""Build Test Map utilities (AST + coverage hybrid).

Outputs (all under docs/_build/):
  - test_map.json              # {symbol: [{file, lines[], windows[], reason}]}
  - test_map_coverage.json     # {symbol: {executed, ratio, hit_lines[], contexts[]}}
  - test_map_summary.json      # {module: {"untested_top10": [symbol, ...], "coverage_ratio_avg": float}}
  - test_map_lint.json         # [{"severity","symbol","rule","message","module","file"}]

Policy/env:
  TESTMAP_WINDOW=3             # #lines on either side for precision windows
  TESTMAP_FAIL_BUDGET=5        # max allowed untested public symbols before failing
  TESTMAP_FAIL_ON_UNTESTED=1   # exit(2) if budget exceeded
  TESTMAP_COVERAGE_JSON=docs/_build/coverage.json
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
OUTDIR = ROOT / "docs" / "_build"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE_MAP = OUTDIR / "test_map.json"
OUTFILE_COV = OUTDIR / "test_map_coverage.json"
OUTFILE_SUM = OUTDIR / "test_map_summary.json"
OUTFILE_LINT = OUTDIR / "test_map_lint.json"

# Restrict symbol discovery to known project namespaces to keep noise low.
KNOWN_PREFIXES = (
    "kgfoundry",
    "kgfoundry_common",
    "kg_builder",
    "search_api",
    "embeddings_dense",
    "embeddings_sparse",
    "ontology",
    "orchestration",
    "observability",
    "registry",
)

WINDOW = int(os.getenv("TESTMAP_WINDOW", "3"))
COV_JSON = Path(os.getenv("TESTMAP_COVERAGE_JSON", str(OUTDIR / "coverage.json")))
FAIL_BUDGET = int(os.getenv("TESTMAP_FAIL_BUDGET", "5"))
FAIL_ON_UNTESTED = os.getenv("TESTMAP_FAIL_ON_UNTESTED", "0") == "1"


def _load_json(path: Path) -> Any:
    """Load json.

    Parameters
    ----------
    path : Path
        Description.

    Returns
    -------
    Any
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _load_json(...)
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_symbol_candidates() -> set[str]:
    """Compute load symbol candidates.

    Carry out the load symbol candidates operation.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    
    
    
    
    
    
    candidates: set[str] = set()
    symbols_json = ROOT / "docs" / "_build" / "symbols.json"
    if symbols_json.exists():
        try:
            data = json.loads(symbols_json.read_text())
        except json.JSONDecodeError:
            data = []
        for row in data:
            path = row.get("path")
            if isinstance(path, str):
                candidates.add(path)
    if candidates:
        return candidates

    for pyfile in SRC.rglob("*.py"):
        rel = pyfile.relative_to(SRC)
        module = ".".join(rel.with_suffix("").parts)
        if module.startswith(KNOWN_PREFIXES):
            candidates.add(module)
    return candidates


def load_symbol_spans() -> dict[str, dict[str, Any]]:
    """Compute load symbol spans.

    Carry out the load symbol spans operation.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Description of return value.
    """
    
    
    
    
    
    
    out: dict[str, dict[str, Any]] = {}
    symbols_json = ROOT / "docs" / "_build" / "symbols.json"
    if not symbols_json.exists():
        return out
    try:
        rows = json.loads(symbols_json.read_text(encoding="utf-8"))
    except Exception:
        rows = []
    for r in rows:
        p = r.get("path")
        f = r.get("file")
        ln = r.get("lineno")
        en = r.get("endlineno")
        mod = r.get("module") or ".".join((p or "").split(".")[:-1])
        if isinstance(p, str) and isinstance(f, str) and isinstance(ln, int):
            out[p] = {"file": f, "lineno": ln, "endlineno": en or ln, "module": mod}
    return out


def load_public_symbols() -> set[str]:
    """Compute load public symbols.

    Carry out the load public symbols operation.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    
    
    
    
    
    
    nav = ROOT / "site" / "_build" / "navmap" / "navmap.json"
    if not nav.exists():
        return set()
    try:
        j = json.loads(nav.read_text(encoding="utf-8"))
    except Exception:
        return set()
    exports: set[str] = set()
    mods = (j.get("modules") or {}).items()
    for mod, entry in mods:
        ex = entry.get("exports") or []
        for s in ex:
            if isinstance(s, str):
                exports.add(f"{mod}.{s}" if not s.startswith(f"{mod}.") else s)
    return exports


def _names_from_ast(tree: ast.AST | None) -> set[str]:
    """Names from ast.

    Parameters
    ----------
    tree : ast.AST | None
        Description.

    Returns
    -------
    set[str]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _names_from_ast(...)
    """
    names: set[str] = set()
    if tree is None:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            for alias in node.names:
                if node.module:
                    names.add(f"{node.module}.{alias.name}")
                names.add(alias.name)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                names.add(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def scan_test_file(path: Path, symbols: set[str]) -> dict[str, list[dict[str, object]]]:
    """Compute scan test file.

    Carry out the scan test file operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.
    symbols : Set[str]
        Description for ``symbols``.

    Returns
    -------
    Mapping[str, List[Mapping[str, object]]]
        Description of return value.
    """
    
    
    
    
    
    
    try:
        text = path.read_text("utf-8")
    except OSError:
        return {}

    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None

    dotted_tokens = set(re.findall(r"[A-Za-z_][\w\.]+", text))
    ast_tokens = _names_from_ast(tree)
    matches: dict[str, list[dict[str, object]]] = {}

    for symbol in symbols:
        top = symbol.split(".", 1)[0]
        tail = symbol.split(".")[-1]
        reason = None
        if symbol in dotted_tokens:
            reason = "dotted"
        elif symbol in ast_tokens:
            reason = "ast_fqn"
        elif top in ast_tokens:
            reason = "ast_top"
        elif tail in dotted_tokens:
            reason = "tail"
        if reason:
            # locate up to 5 hit windows
            line_hits: list[int] = []
            for lineno, raw in enumerate(text.splitlines(), start=1):
                if symbol in raw or (tail and tail in raw):
                    line_hits.append(lineno)
                if len(line_hits) >= 5:
                    break
            windows = [{"start": max(1, n - WINDOW), "end": n + WINDOW} for n in line_hits]
            matches.setdefault(symbol, []).append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "lines": line_hits,
                    "windows": windows,
                    "reason": reason,
                }
            )
    return matches


# ---------------------------- coverage ingestion -----------------------------


def _normalize_repo_rel(path_like: str) -> str:
    """Normalize repo rel.

    Parameters
    ----------
    path_like : str
        Description.

    Returns
    -------
    str
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _normalize_repo_rel(...)
    """
    p = Path(path_like)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        # coverage JSON often records absolute paths; make repo-relative if possible
        try:
            raw = str(p)
            root = str(ROOT)
            idx = raw.find(root)
            if idx != -1:
                return raw[idx + len(root) :].lstrip(os.sep)
        except Exception:
            pass
        return str(p)


def load_coverage() -> tuple[dict[str, set[int]], dict[tuple[str, int], set[str]]]:
    """Compute load coverage.

    Carry out the load coverage operation.

    Returns
    -------
    Tuple[dict[str, Set[int]], dict[Tuple[str, int], Set[str]]]
        Description of return value.
    """
    
    
    
    
    
    
    if not COV_JSON.exists():
        return ({}, {})
    data = _load_json(COV_JSON)
    if not isinstance(data, dict):
        return ({}, {})
    files = data.get("files") or {}
    executed: dict[str, set[int]] = {}
    ctx_by_line: dict[tuple[str, int], set[str]] = {}
    for fpath, info in files.items():
        rel = _normalize_repo_rel(fpath)
        lines = set(info.get("executed_lines") or [])
        executed[rel] = lines
        # contexts are optional; enabled when [json] show_contexts = True
        # Coverage JSON may record contexts as mapping of context -> [lines]
        ctxs = info.get("contexts") or {}
        if isinstance(ctxs, dict):
            for ctx, ln_list in ctxs.items():
                for ln in ln_list or []:
                    ctx_by_line.setdefault((rel, ln), set()).add(str(ctx))
    return (executed, ctx_by_line)


# ---------------------------- builders & policy ------------------------------


def build_test_map(symbols: set[str]) -> dict[str, list[dict[str, object]]]:
    """Compute build test map.

    Carry out the build test map operation.

    Parameters
    ----------
    symbols : Set[str]
        Description for ``symbols``.

    Returns
    -------
    Mapping[str, List[Mapping[str, object]]]
        Description of return value.
    """
    
    
    
    
    
    
    table: dict[str, list[dict[str, object]]] = defaultdict(list)
    if not TESTS.exists():
        return {}

    for test_file in TESTS.rglob("test_*.py"):
        for symbol, rows in scan_test_file(test_file, symbols).items():
            table[symbol].extend(rows)

    # drop empties
    return {sym: rows for sym, rows in table.items() if rows}


def attach_coverage(
    symbol_spans: dict[str, dict[str, Any]],
    executed: dict[str, set[int]],
    ctx_by_line: dict[tuple[str, int], set[str]],
) -> dict[str, dict[str, Any]]:
    """Compute attach coverage.

    Carry out the attach coverage operation.

    Parameters
    ----------
    symbol_spans : Mapping[str, Mapping[str, Any]]
        Description for ``symbol_spans``.
    executed : Mapping[str, Set[int]]
        Description for ``executed``.
    ctx_by_line : Mapping[Tuple[str, int], Set[str]]
        Description for ``ctx_by_line``.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Description of return value.
    """
    
    
    
    
    
    
    result: dict[str, dict[str, Any]] = {}
    for sym, meta in symbol_spans.items():
        f = meta.get("file")
        ln = int(meta.get("lineno") or 0)
        en = int(meta.get("endlineno") or ln)
        rel = f if isinstance(f, str) else None
        hits: list[int] = []
        if rel and rel in executed:
            span = set(range(ln, en + 1))
            hits = sorted(list(executed[rel].intersection(span)))
        # contexts (optional)
        contexts: set[str] = set()
        if hits:
            for h in hits[:50]:  # limit
                contexts |= ctx_by_line.get((rel, h), set())
        ratio = 0.0
        if en >= ln:
            ratio = round(len(hits) / max(1, (en - ln + 1)), 4)
        result[sym] = {
            "executed": bool(hits),
            "ratio": ratio,
            "hit_lines": hits[:100],  # trim
            "file": rel,
            "span": [ln, en],
            "contexts": sorted(list(contexts))[:100],
        }
    return result


def summarize(
    public_syms: set[str],
    symbol_spans: dict[str, dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute summarize.

    Carry out the summarize operation.

    Parameters
    ----------
    public_syms : Set[str]
        Description for ``public_syms``.
    symbol_spans : Mapping[str, Mapping[str, Any]]
        Description for ``symbol_spans``.
    coverage : Mapping[str, Mapping[str, Any]]
        Description for ``coverage``.
    budget : int
        Description for ``budget``.

    Returns
    -------
    Tuple[dict[str, Any], List[dict[str, Any]]]
        Description of return value.
    """
    
    
    
    
    
    
    # group by module
    by_mod: dict[str, list[str]] = defaultdict(list)
    for s in public_syms:
        mod = symbol_spans.get(s, {}).get("module") or ".".join(s.split(".")[:-1])
        by_mod[mod].append(s)

    summary: dict[str, Any] = {}
    lints: list[dict[str, Any]] = []
    untested_total = 0
    for mod, syms in by_mod.items():
        ratios = []
        untested = []
        for s in syms:
            cov = coverage.get(s, {})
            ratios.append(float(cov.get("ratio") or 0.0))
            if not cov.get("executed"):
                untested.append(s)
        untested_total += len(untested)
        summary[mod] = {
            "untested_top10": untested[:10],
            "coverage_ratio_avg": round(sum(ratios) / max(1, len(ratios)), 4),
            "public_count": len(syms),
            "untested_count": len(untested),
        }

    if untested_total > budget:
        lints.append(
            {
                "severity": "error",
                "symbol": "*",
                "rule": "untested_public_budget",
                "message": f"Untested public symbols {untested_total} exceed budget {budget}",
                "module": "*",
                "file": "*",
            }
        )
    return summary, lints


def main() -> None:
    """Run the build-test-map command-line entry point.

    Returns
    -------
    None
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> main(...)
    """
    symbols = load_symbol_candidates()
    spans = load_symbol_spans()
    public_syms = load_public_symbols()

    # 1) heuristic test map
    heuristic = build_test_map(symbols)

    # 2) coverage overlay
    executed, ctx_map = load_coverage()
    cov = attach_coverage(spans, executed, ctx_map)

    # 3) policy & summaries
    summary, lints = summarize(public_syms or set(), spans, cov, FAIL_BUDGET)

    # write artifacts
    OUTFILE_MAP.write_text(json.dumps(heuristic, indent=2) + "\n", encoding="utf-8")
    OUTFILE_COV.write_text(json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    OUTFILE_SUM.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    OUTFILE_LINT.write_text(json.dumps(lints, indent=2) + "\n", encoding="utf-8")

    # strict mode
    if FAIL_ON_UNTESTED and any(x["severity"] == "error" for x in lints):
        print(f"[testmap] FAIL: {sum(1 for x in lints if x['severity'] == 'error')} error(s)")
        sys.exit(2)
    print(
        f"[testmap] wrote {OUTFILE_MAP.name}, {OUTFILE_COV.name}, {OUTFILE_SUM.name}, {OUTFILE_LINT.name}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
