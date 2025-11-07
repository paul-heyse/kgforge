"""Shared Tree-sitter powered utilities exposed via MCP and HTTP bridges."""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from codeintel.config import LIMITS
from codeintel.errors import ManifestError, SandboxError
from codeintel.index.store import IndexStore
from codeintel.indexer.tscore import (
    LANGUAGE_NAMES,
    _load_manifest,
    get_language,
    load_langs,
    parse_bytes,
    run_query,
)
from codeintel.queries import list_available_queries

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

_HEALTH_PRIORITY: dict[str, int] = {"healthy": 0, "degraded": 1, "unhealthy": 2}


def _merge_status(current: str, candidate: str) -> str:
    """Return the status with higher severity.

    Parameters
    ----------
    current : str
        Previously accumulated status value.
    candidate : str
        New component status value.

    Returns
    -------
    str
        Highest-priority status between ``current`` and ``candidate``.
    """
    if _HEALTH_PRIORITY[candidate] > _HEALTH_PRIORITY[current]:
        return candidate
    return current


def _set_component(
    components: dict[str, dict[str, Any]],
    name: str,
    status: str,
    **extras: object,
) -> None:
    """Record a health component entry.

    This function mutates the components dictionary in place, adding or updating
    a component entry with the specified status and additional metadata fields.
    It is used internally by :func:`get_health` to build the health report.

    Parameters
    ----------
    components : dict[str, dict[str, Any]]
        Mutable mapping of component names to metadata dictionaries.
    name : str
        Component identifier.
    status : str
        Component status string (e.g., "healthy", "unhealthy", "degraded").
    **extras : object
        Additional metadata fields to merge into the component entry.
    """
    components[name] = {"status": status, **extras}


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

    This function resolves the file path, loads the language grammar, parses
    the file, and executes the provided Tree-sitter query. It returns capture
    metadata from the query execution.

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
        If the language identifier is not supported. The error message includes
        a list of available language choices.

    Notes
    -----
    Errors from :func:`resolve_path`, :func:`load_langs`, and :func:`run_query`
    propagate unchanged to the caller.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    target = resolve_path(path)
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

    Notes
    -----
    Errors from :func:`resolve_directory` and :func:`repo_relative` propagate
    unchanged to the caller.
    """
    query_text = _load_python_symbols_query()
    langs = load_langs()
    lang = get_language(langs, "python")
    root = resolve_directory(directory)
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

    This function scans a directory for source files, parses them with Tree-sitter,
    and extracts call expressions. It can optionally filter by callee name to find
    specific function calls.

    Parameters
    ----------
    directory : str
        Repository-relative directory to inspect.
    language : str, optional
        Tree-sitter language identifier. Defaults to ``"python"``.
    callee : str | None, optional
        Optional callee filter. If provided, only calls to this function name
        are included in the results.

    Returns
    -------
    list[dict[str, Any]]
        Capture metadata describing call edges, including callee name, arguments,
        and source location spans.

    Raises
    ------
    ValueError
        If the language identifier is not supported. The error message includes
        a list of available language choices.

    Notes
    -----
    Errors from :func:`resolve_directory` and :func:`run_query` propagate to the
    caller without modification.
    """
    if language not in LANGUAGE_NAMES:
        choices = ", ".join(LANGUAGE_NAMES)
        message = f"Unsupported language '{language}'. Choices: {choices}"
        raise ValueError(message)
    root = resolve_directory(directory)
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

    Notes
    -----
    Errors from :func:`run_ts_query` propagate directly to the caller.
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

    Notes
    -----
    Errors from :func:`resolve_directory` and :func:`repo_relative` propagate to
    the caller.
    """
    root = resolve_directory(directory)
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

    This function reads a portion of a file starting at the specified byte offset.
    It validates the offset, reads the requested segment, and decodes it as UTF-8
    with error replacement for invalid byte sequences.

    Parameters
    ----------
    path : str
        Repository-relative file path.
    offset : int, optional
        Byte offset to start reading, by default 0.
    length : int | None, optional
        Maximum bytes to read, or None for remainder of file.

    Returns
    -------
    dict[str, Any]
        File metadata including path, size, offset, and decoded text segment.
        The text is decoded with UTF-8 error replacement for invalid sequences.

    Raises
    ------
    ValueError
        If offset is negative or exceeds the file size.

    Notes
    -----
    Errors from :func:`resolve_path` and :func:`repo_relative` propagate to the
    caller without modification.
    """
    p = resolve_path(path)
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

    Notes
    -----
    Errors from :func:`resolve_path` and :func:`repo_relative` propagate directly
    to the caller.
    """
    p = resolve_path(path)
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

    This function parses a file with Tree-sitter and extracts a hierarchical
    outline of top-level definitions (functions, classes, etc.). The outline
    is limited by configured maximum items to prevent unbounded responses.

    Parameters
    ----------
    path : str
        Repository-relative file path.
    language : str, optional
        Tree-sitter language identifier, by default "python".

    Returns
    -------
    dict[str, Any]
        Outline with path and items list. Each item includes name, kind, and
        source location span.

    Raises
    ------
    ValueError
        If language is unsupported. The function explicitly raises ValueError
        with a message listing available language choices.

    Notes
    -----
    Errors from :func:`resolve_path` and :func:`repo_relative` propagate to the
    caller.
    """
    p = resolve_path(path)
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
    """Return aggregated health diagnostics for CodeIntel services.

    The report covers the language manifest, grammar loading, query registry,
    optional index database, and sandbox configuration.

    Returns
    -------
    dict[str, Any]
        Health summary containing the overall status, per-component details,
        and an issues list when degraded states are detected.
    """
    components: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    overall_status = "healthy"

    try:
        manifest = _load_manifest()
    except (ManifestError, OSError, ValueError) as exc:
        _set_component(components, "manifest", "unhealthy", error=str(exc))
        issues.append(f"Manifest error: {exc}")
        overall_status = _merge_status(overall_status, "unhealthy")
    else:
        _set_component(
            components,
            "manifest",
            "healthy",
            languages_count=len(manifest),
            languages=sorted(manifest.keys()),
        )

    try:
        load_langs()
    except (ManifestError, RuntimeError, ValueError) as exc:
        _set_component(components, "grammars", "unhealthy", error=str(exc))
        issues.append(f"Grammar loading error: {exc}")
        overall_status = _merge_status(overall_status, "unhealthy")
    else:
        _set_component(components, "grammars", "healthy", loaded=list(LANGUAGE_NAMES))

    available_queries = list_available_queries()
    missing_queries = [lang for lang in LANGUAGE_NAMES if lang not in available_queries]
    query_status = "healthy" if not missing_queries else "degraded"
    query_extras: dict[str, Any] = {"available": available_queries}
    if missing_queries:
        query_extras["missing"] = missing_queries
        issues.append("Missing queries for: " + ", ".join(sorted(missing_queries)))
        overall_status = _merge_status(overall_status, "degraded")
    _set_component(components, "queries", query_status, **query_extras)

    index_path = REPO_ROOT / ".kgf" / "codeintel.db"
    if index_path.exists():
        try:
            with IndexStore(index_path) as store:
                symbol_count = store.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
                file_count = store.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        except (OSError, sqlite3.DatabaseError) as exc:
            _set_component(
                components,
                "index",
                "degraded",
                path=str(index_path),
                error=str(exc),
            )
            issues.append(f"Index database error: {exc}")
            overall_status = _merge_status(overall_status, "degraded")
        else:
            _set_component(
                components,
                "index",
                "healthy",
                path=str(index_path),
                symbols=symbol_count,
                files=file_count,
            )
    else:
        _set_component(
            components,
            "index",
            "not_configured",
            message="No index found. Run 'codeintel index build' to create one.",
        )

    _set_component(
        components,
        "sandbox",
        "healthy",
        repo_root=str(REPO_ROOT),
        writable=os.access(REPO_ROOT, os.W_OK),
    )

    return {
        "status": overall_status,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "components": components,
        "issues": issues or None,
    }
