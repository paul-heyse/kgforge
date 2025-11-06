"""Shared Tree-sitter powered utilities exposed via MCP and HTTP bridges."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from codeintel.indexer.tscore import (
    LANGUAGE_NAMES,
    get_language,
    load_langs,
    parse_bytes,
    run_query,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_DIR = REPO_ROOT / "codeintel" / "queries"

PY_SYMBOL_QUERY = (QUERIES_DIR / "python.scm").read_text(encoding="utf-8")

CALL_QUERY = """
(call
  function: (identifier) @call.name
  arguments: (argument_list) @call.args) @call.node
""".strip()

ERROR_QUERY = """
(ERROR) @syntax.error
(MISSING) @syntax.missing
""".strip()


@dataclass(frozen=True)
class QueryResult:
    """Standard envelope for query responses."""

    captures: list[dict[str, Any]]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def _resolve_directory(path: str) -> Path:
    directory = _resolve_path(path)
    if not directory.is_dir():  # pragma: no cover - caller validates paths
        message = f"Directory '{path}' does not exist."
        raise FileNotFoundError(message)
    return directory


def _group_captures(
    captures: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for capture in captures:
        match_id = int(capture.get("match_id", -1))
        grouped.setdefault(match_id, {}).setdefault(capture["capture"], []).append(capture)
    return grouped


def _extract_definitions(capture: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    names = capture.get("def.name")
    if not names:
        return []
    params = capture.get("def.params", [])
    nodes = capture.get("def.node", names)
    entries: list[dict[str, Any]] = []
    for idx, name_entry in enumerate(names):
        node_entry = nodes[idx] if idx < len(nodes) else name_entry
        param_entry = params[idx] if idx < len(params) else {}
        entries.append(
            {
                "name": name_entry.get("text"),
                "params": param_entry.get("text"),
                "span": {
                    "start": node_entry.get("start_point"),
                    "end": node_entry.get("end_point"),
                },
            }
        )
    return entries


def _extract_calls(
    capture: Mapping[str, list[Mapping[str, Any]]],
    *,
    file_path: str,
    callee: str | None,
) -> list[dict[str, Any]]:
    names = capture.get("call.name", [])
    if not names:
        return []
    args = capture.get("call.args", [])
    nodes = capture.get("call.node", names)
    edges: list[dict[str, Any]] = []
    for idx, name_entry in enumerate(names):
        name = name_entry.get("text")
        if callee and name != callee:
            continue
        args_entry = args[idx] if idx < len(args) else {}
        node_entry = nodes[idx] if idx < len(nodes) else name_entry
        edges.append(
            {
                "file": file_path,
                "callee": name,
                "arguments": args_entry.get("text"),
                "span": {
                    "start": name_entry.get("start_point"),
                    "end": node_entry.get("end_point"),
                },
            }
        )
    return edges


@cache
def _load_python_symbols_query() -> str:
    return PY_SYMBOL_QUERY


def run_ts_query(path: str, *, language: str, query: str) -> QueryResult:
    """Execute a Tree-sitter query against a single file.

    Parameters
    ----------
    path : str
        Absolute or repository-relative file path.
    language : str
        Tree-sitter language identifier.
    query : str
        S-expression query to execute.

    Returns
    -------
    QueryResult
        Capture payload from :func:`run_query`.

    Raises
    ------
    ValueError
        If the language is not supported by the indexer.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    target = _resolve_path(path)
    data = target.read_bytes()
    langs = load_langs()
    lang = get_language(langs, language)
    tree = parse_bytes(lang, data)
    captures = run_query(lang, query, tree, data)
    return QueryResult(captures=captures)


def list_python_symbols(directory: str) -> list[dict[str, Any]]:
    """Collect Tree-sitter symbol definitions for Python source files.

    Parameters
    ----------
    directory : str
        Directory path to scan for Python source files.

    Returns
    -------
    list[dict[str, Any]]
        Collection of files paired with their discovered symbol metadata.

    Notes
    -----
    Raises :exc:`FileNotFoundError` if the directory does not exist (raised by
    internal directory resolution helper).
    """
    query_text = _load_python_symbols_query()
    langs = load_langs()
    lang = get_language(langs, "python")
    root = _resolve_directory(directory)
    results: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*.py")):
        data = file_path.read_bytes()
        tree = parse_bytes(lang, data)
        captures = run_query(lang, query_text, tree, data)
        grouped = _group_captures(captures)
        defs: list[dict[str, Any]] = []
        for capture in grouped.values():
            defs.extend(_extract_definitions(capture))
        if defs:
            results.append({"file": str(file_path), "defs": defs})
    return results


def list_calls(
    directory: str, *, language: str = "python", callee: str | None = None
) -> list[dict[str, Any]]:
    """Enumerate call expressions discovered within ``directory``.

    Parameters
    ----------
    directory : str
        Directory to inspect.
    language : str, optional
        Tree-sitter language identifier. Defaults to ``"python"``.
    callee : str | None, optional
        Optional callee filter.

    Returns
    -------
    list[dict[str, Any]]
        Capture metadata describing call edges.

    Raises
    ------
    ValueError
        If ``language`` is not supported.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    root = _resolve_directory(directory)
    langs = load_langs()
    lang = get_language(langs, language)
    edges: list[dict[str, Any]] = []
    pattern = "*.py" if language == "python" else "*"
    for file_path in sorted(root.rglob(pattern)):
        if not file_path.is_file():
            continue
        data = file_path.read_bytes()
        tree = parse_bytes(lang, data)
        captures = run_query(lang, CALL_QUERY, tree, data)
        matches = _group_captures(captures)
        for info in matches.values():
            edges.extend(_extract_calls(info, file_path=str(file_path), callee=callee))
    return edges


def list_errors(path: str, *, language: str = "python") -> list[dict[str, Any]]:
    """Report Tree-sitter syntax errors for a source file.

    Parameters
    ----------
    path : str
        File path to analyze for syntax errors.
    language : str, optional
        Tree-sitter language identifier. Defaults to ``"python"``.

    Returns
    -------
    list[dict[str, Any]]
        Captures describing error spans and the extracted text.

    Notes
    -----
    Raises :exc:`ValueError` if the language is not supported, or
    :exc:`FileNotFoundError` if the file cannot be found (both raised by internal
    query execution and path resolution helpers).
    """
    captures = run_ts_query(path, language=language, query=ERROR_QUERY).captures
    return [
        {
            "capture": capture["capture"],
            "span": {
                "start": capture["start_point"],
                "end": capture["end_point"],
            },
            "text": capture["text"],
        }
        for capture in captures
    ]
