"""Build Test Map utilities."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
OUTFILE = ROOT / "docs" / "_build" / "test_map.json"

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


def _names_from_ast(tree: ast.AST) -> set[str]:
    """Compute names from ast.

    Carry out the names from ast operation.

    Parameters
    ----------
    tree : ast.AST
        Description for ``tree``.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    names: set[str] = set()
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
    ast_tokens = _names_from_ast(tree) if tree is not None else set()
    matches: dict[str, list[dict[str, object]]] = {}

    for symbol in symbols:
        top = symbol.split(".", 1)[0]
        tail = symbol.split(".")[-1]
        if (
            symbol in dotted_tokens
            or symbol in ast_tokens
            or top in ast_tokens
            or tail in dotted_tokens
        ):
            line_hits: list[int] = []
            for lineno, raw in enumerate(text.splitlines(), start=1):
                if symbol in raw or (tail and tail in raw):
                    line_hits.append(lineno)
                if len(line_hits) >= 5:
                    break
            matches.setdefault(symbol, []).append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "lines": line_hits,
                }
            )
    return matches


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

    return {sym: rows for sym, rows in table.items() if rows}


def main() -> None:
    """Compute main.

    Carry out the main operation.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    symbols = load_symbol_candidates()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(build_test_map(symbols), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
