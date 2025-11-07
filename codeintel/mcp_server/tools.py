"""Shared Tree-sitter powered utilities exposed via MCP and HTTP bridges."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from codeintel.config import LIMITS

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode
else:
    # Runtime type stub for Tree-sitter nodes
    class TSNode(Protocol):
        """Protocol for Tree-sitter node objects."""

        type: str
        start_point: tuple[int, int]
        end_point: tuple[int, int]
        children: list[TSNode]


from codeintel.indexer.tscore import (
    LANGUAGE_NAMES,
    get_language,
    load_langs,
    parse_bytes,
    run_query,
)

REPO_ROOT = Path(os.environ.get("KGF_REPO_ROOT", Path.cwd())).resolve()
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

EXCLUDES = [
    "**/.git/**",
    "**/.venv/**",
    "**/_build/**",
    "**/__pycache__/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/node_modules/**",
]


class SandboxError(ValueError):
    """Raised when a path resolution violates the repository sandbox."""


@dataclass(frozen=True)
class QueryResult:
    """Standard envelope for query responses."""

    captures: list[dict[str, Any]]


def repo_relative(p: Path) -> str:
    """Return repository-relative path string.

    Parameters
    ----------
    p : Path
        Absolute path within repository.

    Returns
    -------
    str
        Repository-relative path string.

    Raises
    ------
    SandboxError
        If path is outside repository root.
    """
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        message = f"Path outside repository: {p}"
        raise SandboxError(message) from exc


def resolve_path(rel: str) -> Path:
    """Resolve and validate a repository-relative path.

    Parameters
    ----------
    rel : str
        Repository-relative path string.

    Returns
    -------
    Path
        Resolved absolute path within repository.

    Raises
    ------
    SandboxError
        If path resolves outside repository root.
    FileNotFoundError
        If path does not exist.
    """
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT)):
        message = f"Path outside repository: {rel}"
        raise SandboxError(message)
    if p.is_dir():
        return p
    if not p.exists():
        raise FileNotFoundError(rel)
    return p


def resolve_directory(rel: str | None) -> Path:
    """Resolve and validate a repository-relative directory path.

    Parameters
    ----------
    rel : str | None
        Repository-relative directory path, or None for root.

    Returns
    -------
    Path
        Resolved absolute directory path within repository.

    Raises
    ------
    SandboxError
        If path resolves outside repository or is not a directory.
    """
    d = REPO_ROOT if not rel else resolve_path(rel)
    if not d.is_dir():
        message = f"Not a directory: {rel}"
        raise SandboxError(message)
    return d


def _bounded_limit(n: int | None) -> int:
    """Enforce limit bounds from configuration.

    Parameters
    ----------
    n : int | None
        Requested limit, or None for default.

    Returns
    -------
    int
        Bounded limit value between 1 and LIMITS.list_limit_max.
    """
    if n is None:
        return LIMITS.list_limit_default
    return max(1, min(int(n), LIMITS.list_limit_max))


def _group_captures(
    captures: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, list[Mapping[str, Any]]]]:
    """Group captures by match ID.

    Parameters
    ----------
    captures : Iterable[Mapping[str, Any]]
        Capture records from Tree-sitter query.

    Returns
    -------
    dict[int, dict[str, list[Mapping[str, Any]]]]
        Captures grouped by match_id, then by capture name.
    """
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for capture in captures:
        match_id = int(capture.get("match_id", -1))
        grouped.setdefault(match_id, {}).setdefault(capture["capture"], []).append(capture)
    return grouped


def _extract_definitions(
    capture: Mapping[str, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Extract symbol definitions from grouped captures.

    Parameters
    ----------
    capture : Mapping[str, list[Mapping[str, Any]]]
        Grouped captures for a single match.

    Returns
    -------
    list[dict[str, Any]]
        Symbol definition records with name, params, and span.
    """
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
                    "start": name_entry.get("start_point"),
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
    """Extract call expressions from grouped captures.

    Parameters
    ----------
    capture : Mapping[str, list[Mapping[str, Any]]]
        Grouped captures for a single match.
    file_path : str
        Source file path.
    callee : str | None
        Optional callee name filter.

    Returns
    -------
    list[dict[str, Any]]
        Call edge records with callee, arguments, and span.
    """
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
    """Load Python symbol query from file.

    Returns
    -------
    str
        Query S-expression string.
    """
    return PY_SYMBOL_QUERY


def run_ts_query(path: str, *, language: str, query: str) -> QueryResult:
    """Execute a Tree-sitter query against a single file.

    Parameters
    ----------
    path : str
        Repository-relative file path.
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
    SandboxError
        If path is outside repository.
    FileNotFoundError
        If file does not exist.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    # Resolve path - exceptions propagate from helper
    try:
        target = resolve_path(path)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
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
        Repository-relative directory path to scan.

    Returns
    -------
    list[dict[str, Any]]
        Collection of files paired with their discovered symbol metadata.

    Raises
    ------
    SandboxError
        If directory path is outside repository.
    FileNotFoundError
        If directory does not exist.
    """
    query_text = _load_python_symbols_query()
    langs = load_langs()
    lang = get_language(langs, "python")
    # Resolve directory - exceptions propagate from helper
    try:
        root = resolve_directory(directory)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
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
            rel_path = repo_relative(file_path)
            results.append({"file": rel_path, "defs": defs})
    return results


def list_calls(
    directory: str, *, language: str = "python", callee: str | None = None
) -> list[dict[str, Any]]:
    """Enumerate call expressions discovered within ``directory``.

    Parameters
    ----------
    directory : str
        Repository-relative directory to inspect.
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
    SandboxError
        If directory path is outside repository.
    FileNotFoundError
        If directory does not exist.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    # Resolve directory - exceptions propagate from helper
    try:
        root = resolve_directory(directory)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
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
            rel_path = repo_relative(file_path)
            edges.extend(_extract_calls(info, file_path=rel_path, callee=callee))
    return edges


def list_errors(path: str, *, language: str = "python") -> list[dict[str, Any]]:
    """Report Tree-sitter syntax errors for a source file.

    Parameters
    ----------
    path : str
        Repository-relative file path to analyze.
    language : str, optional
        Tree-sitter language identifier. Defaults to ``"python"``.

    Returns
    -------
    list[dict[str, Any]]
        Captures describing error spans and the extracted text.

    Raises
    ------
    ValueError
        If the language is not supported.
    SandboxError
        If path is outside repository.
    FileNotFoundError
        If file does not exist.
    """
    # Call run_ts_query - exceptions propagate from helper
    try:
        captures = run_ts_query(path, language=language, query=ERROR_QUERY).captures
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
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


def list_files(
    directory: str | None = None, glob: str | None = None, limit: int | None = None
) -> list[str]:
    """List repository files with optional filters.

    Parameters
    ----------
    directory : str | None, optional
        Repository-relative directory to scan, or None for root.
    glob : str | None, optional
        Optional glob pattern filter.
    limit : int | None, optional
        Maximum number of files to return.

    Returns
    -------
    list[str]
        Repository-relative file paths.

    Raises
    ------
    SandboxError
        If directory path is outside repository.
    FileNotFoundError
        If directory does not exist.
    """
    # Resolve directory - exceptions propagate from helper
    try:
        root = resolve_directory(directory)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    cap = _bounded_limit(limit)
    out: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = repo_relative(p)
        if any(fnmatch(rel, pat.replace("**/", "")) for pat in EXCLUDES):
            continue
        if glob and not fnmatch(rel, glob):
            continue
        out.append(rel)
        if len(out) >= cap:
            break
    return out


def get_file(path: str, offset: int = 0, length: int | None = None) -> dict[str, Any]:
    """Read a file segment with UTF-8 decoding.

    Parameters
    ----------
    path : str
        Repository-relative file path.
    offset : int, optional
        Byte offset to start reading, by default 0.
    length : int | None, optional
        Maximum bytes to read, or None for remainder.

    Returns
    -------
    dict[str, Any]
        File metadata and decoded text segment.

    Raises
    ------
    ValueError
        If offset is invalid.
    SandboxError
        If path is outside repository.
    FileNotFoundError
        If file does not exist.
    """
    # Resolve path - exceptions propagate from helper
    try:
        p = resolve_path(path)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    buf = p.read_bytes()
    if offset < 0 or offset > len(buf):
        message = "invalid offset"
        raise ValueError(message)
    end = len(buf) if length is None else min(len(buf), offset + max(0, length))
    rel_path = repo_relative(p)
    return {
        "path": rel_path,
        "size": len(buf),
        "offset": offset,
        "data": buf[offset:end].decode("utf-8", errors="replace"),
    }


def get_ast(path: str, language: str, fmt: str = "json") -> dict[str, Any]:
    """Return a bounded AST snapshot for a file.

    Parameters
    ----------
    path : str
        Repository-relative file path.
    language : str
        Tree-sitter language identifier.
    fmt : str, optional
        Output format: "json" or "sexpr", by default "json".

    Returns
    -------
    dict[str, Any]
        AST representation with path and format metadata.

    Raises
    ------
    ValueError
        If file exceeds size limit, language is unsupported, or format is invalid.
    SandboxError
        If path is outside repository.
    FileNotFoundError
        If file does not exist.
    """
    # Resolve path - exceptions propagate from helper
    try:
        p = resolve_path(path)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    data = p.read_bytes()
    if len(data) > LIMITS.max_ast_bytes:
        message = f"file too large: {len(data)} bytes (limit {LIMITS.max_ast_bytes})"
        raise ValueError(message)
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    langs = load_langs()
    lang = get_language(langs, language)
    tree = parse_bytes(lang, data)
    rel_path = repo_relative(p)
    if fmt == "sexpr":
        return {"path": rel_path, "format": "sexpr", "ast": tree.root_node.sexp()}
    # Shallow JSON walk (node type + span only) to keep size bounded
    ast_obj = _walk_ast_to_json(tree.root_node)
    return {"path": rel_path, "format": "json", "ast": ast_obj}


def _walk_ast_to_json(node: TSNode, budget: int = 200000) -> dict[str, Any] | None:
    """Recursively walk AST nodes with budget limit.

    Parameters
    ----------
    node : TSNode
        Tree-sitter node to walk.
    budget : int, optional
        Remaining node budget, by default 200000.

    Returns
    -------
    dict[str, Any] | None
        Node representation or None if budget exhausted.
    """
    if budget <= 0:
        return None
    obj: dict[str, Any] = {
        "type": node.type,
        "span": {
            "start": {"row": node.start_point[0] + 1, "col": node.start_point[1] + 1},
            "end": {"row": node.end_point[0] + 1, "col": node.end_point[1] + 1},
        },
    }
    children: list[dict[str, Any]] = []
    remaining = budget - 1
    for ch in node.children:
        child_obj = _walk_ast_to_json(ch, remaining)
        if child_obj is not None:
            children.append(child_obj)
            remaining -= 1
        if remaining <= 0:
            break
    if children:
        obj["children"] = children
    return obj


def _extract_outline_items(
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]], max_items: int
) -> list[dict[str, Any]]:
    """Extract outline items from grouped captures.

    Parameters
    ----------
    grouped : dict[int, dict[str, list[Mapping[str, Any]]]]
        Grouped captures from Tree-sitter query.
    max_items : int
        Maximum number of items to extract.

    Returns
    -------
    list[dict[str, Any]]
        List of outline items with name, kind, and span.
    """
    items: list[dict[str, Any]] = []
    item_count = 0
    for g in grouped.values():
        if item_count >= max_items:
            break
        names = list(g.get("def.name", []))
        if not names:
            continue
        name = names[0].get("text", "")
        node_captures = g.get("def.node", names)
        node = node_captures[0] if node_captures else names[0]
        items.append(
            {
                "name": name,
                "kind": "function",  # Could infer from node.type
                "span": {
                    "start": node.get("start_point"),
                    "end": node.get("end_point"),
                },
            }
        )
        item_count += 1
    return items


def get_outline(path: str, language: str = "python") -> dict[str, Any]:
    """Return a hierarchical outline (functions/classes) for a file.

    Parameters
    ----------
    path : str
        Repository-relative file path.
    language : str, optional
        Tree-sitter language identifier, by default "python".

    Returns
    -------
    dict[str, Any]
        Outline with path and items list.

    Raises
    ------
    ValueError
        If language is unsupported.
    SandboxError
        If path is outside repository.
    FileNotFoundError
        If file does not exist.
    """
    # Resolve path - exceptions propagate from helper
    try:
        p = resolve_path(path)
    except SandboxError as exc:
        raise SandboxError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    data = p.read_bytes()
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    query_file = QUERIES_DIR / f"{language}.scm"
    if not query_file.exists():
        # Return empty outline if query file missing
        rel_path = repo_relative(p)
        return {"path": rel_path, "items": []}
    query_text = query_file.read_text(encoding="utf-8")
    langs = load_langs()
    lang = get_language(langs, language)
    tree = parse_bytes(lang, data)
    captures = run_query(lang, query_text, tree, data)
    grouped = _group_captures(captures)
    items = _extract_outline_items(grouped, LIMITS.max_outline_items)
    rel_path = repo_relative(p)
    return {"path": rel_path, "items": items}


def get_health() -> dict[str, Any]:
    """Return server health status.

    Returns
    -------
    dict[str, Any]
        Health metrics including loaded languages and available queries.
    """
    query_files = list(QUERIES_DIR.glob("*.scm"))
    return {
        "files_indexed": 0,  # Would come from index store if available
        "langs_loaded": len(LANGUAGE_NAMES),
        "queries_available": len(query_files),
    }
