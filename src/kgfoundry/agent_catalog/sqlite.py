"""Persistence helpers for storing the Agent Catalog in SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import environ
from pathlib import Path

from kgfoundry.agent_catalog.search import SearchDocument, documents_from_catalog

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

CatalogPayload = Mapping[str, JsonValue]

ModuleRow = tuple[str, str, str]
SymbolRow = tuple[str, str, str, str, str, str]
AnchorRow = tuple[str, JsonValue, JsonValue, JsonValue, str]
RankingRow = tuple[str, JsonValue, JsonValue, JsonValue, JsonValue, int]
CallRow = tuple[str | None, str | None, str, str | None, str | None]
FtsRow = tuple[str, str, str, str, str]
CallGraph = tuple[Mapping[str, set[str]], Mapping[str, set[str]]]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@contextmanager
def _sqlite_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection configured for JSON operations.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Describe ``path``.

    Returns
    -------
    sqlite3.Connection
        Describe return value.
"""
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        yield connection
    finally:  # pragma: no cover - defensive close
        connection.close()


DDL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE packages (
    name TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE modules (
    qualified TEXT PRIMARY KEY,
    package TEXT NOT NULL REFERENCES packages(name) ON DELETE CASCADE,
    data TEXT NOT NULL
);

CREATE TABLE symbols (
    symbol_id TEXT PRIMARY KEY,
    package TEXT NOT NULL,
    module TEXT NOT NULL,
    qname TEXT NOT NULL,
    kind TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE anchors (
    symbol_id TEXT PRIMARY KEY REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    start_line INTEGER,
    end_line INTEGER,
    cst_fingerprint TEXT,
    remap_order TEXT
);

CREATE TABLE calls (
    caller_symbol_id TEXT,
    callee_symbol_id TEXT,
    callee_qname TEXT NOT NULL,
    confidence TEXT,
    kind TEXT,
    PRIMARY KEY (caller_symbol_id, callee_symbol_id, callee_qname)
);

CREATE TABLE ranking_features (
    symbol_id TEXT PRIMARY KEY REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    coverage REAL,
    complexity REAL,
    churn INTEGER,
    stability TEXT,
    deprecated INTEGER
);

CREATE VIRTUAL TABLE symbol_fts USING fts5 (
    symbol_id UNINDEXED,
    package,
    module,
    qname,
    text
);
"""


def _json_dumps(value: JsonValue) -> str:
    """Serialise ``value`` using JSON ensuring ASCII is preserved.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    value : NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``value``.

    Returns
    -------
    str
        Describe return value.
"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None) -> JsonValue:
    """Return JSON-decoded ``value`` handling ``NULL`` gracefully.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    value : str | NoneType
        Describe ``value``.

    Returns
    -------
    NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe return value.
"""
    if value is None:
        return None
    return json.loads(value)


def _stringify(value: JsonValue) -> str | None:
    """Return ``value`` as a string if possible.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    value : NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``value``.

    Returns
    -------
    str | NoneType
        Describe return value.
"""
    if value is None:
        return None
    return str(value)


def _iter_packages(payload: CatalogPayload) -> Sequence[Mapping[str, JsonValue]]:
    """Document  iter packages.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    payload : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Configure the payload.

    Returns
    -------
    str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe return value.
"""
    packages = payload.get("packages")
    if isinstance(packages, Sequence):
        return packages  # type: ignore[return-value]
    return []


def _iter_modules(package: Mapping[str, JsonValue]) -> Sequence[Mapping[str, JsonValue]]:
    """Document  iter modules.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    package : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Configure the package.

    Returns
    -------
    str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe return value.
"""
    modules = package.get("modules")
    if isinstance(modules, Sequence):
        return modules  # type: ignore[return-value]
    return []


def _build_metadata_rows(payload: CatalogPayload) -> list[tuple[str, str]]:
    """Return rows for the ``metadata`` table.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    payload : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``payload``.

    Returns
    -------
    list[tuple[str, str]]
        Describe return value.
"""
    return [
        ("version", _json_dumps(payload.get("version"))),
        ("generated_at", _json_dumps(payload.get("generated_at"))),
        ("repo", _json_dumps(payload.get("repo"))),
        ("link_policy", _json_dumps(payload.get("link_policy"))),
        ("artifacts", _json_dumps(payload.get("artifacts"))),
        ("shards", _json_dumps(payload.get("shards"))),
        ("semantic_index", _json_dumps(payload.get("semantic_index"))),
        ("search", _json_dumps(payload.get("search"))),
    ]


@dataclass(slots=True)
class _SymbolExtraction:
    """Container bundling rows generated from symbol payloads.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    modules : list[tuple[str, str, str]]
        Describe ``modules``.
    symbols : list[tuple[str, str, str, str, str, str]]
        Describe ``symbols``.
    anchors : list[tuple[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], str]]
        Describe ``anchors``.
    ranking : list[tuple[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], int]]
        Describe ``ranking``.
    lookup : dict[str, str]
        Describe ``lookup``.
"""

    modules: list[ModuleRow]
    symbols: list[SymbolRow]
    anchors: list[AnchorRow]
    ranking: list[RankingRow]
    lookup: dict[str, str]


def _collect_symbol_rows(packages: Sequence[Mapping[str, JsonValue]]) -> _SymbolExtraction:
    """Aggregate module, symbol, anchor, and ranking rows.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    packages : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``packages``.

    Returns
    -------
    _SymbolExtraction
        Describe return value.
"""
    module_rows: list[ModuleRow] = []
    symbol_rows: list[SymbolRow] = []
    anchor_rows: list[AnchorRow] = []
    ranking_rows: list[RankingRow] = []
    symbol_lookup: dict[str, str] = {}
    for package in packages:
        package_name = _stringify(package.get("name")) or ""
        for module in _iter_modules(package):
            qualified = _stringify(module.get("qualified")) or _stringify(module.get("name")) or ""
            module_payload: JsonObject = {
                "name": module.get("name"),
                "qualified": qualified,
                "source": module.get("source"),
                "pages": module.get("pages"),
                "imports": module.get("imports"),
                "graph": module.get("graph"),
            }
            module_rows.append((qualified, package_name, _json_dumps(module_payload)))
            symbols = module.get("symbols")
            if not isinstance(symbols, Sequence):
                continue
            for symbol in symbols:
                if not isinstance(symbol, Mapping):
                    continue
                symbol_id = _stringify(symbol.get("symbol_id")) or ""
                qname_value = _stringify(symbol.get("qname")) or symbol_id
                if qname_value:
                    symbol_lookup[qname_value] = symbol_id
                symbol_payload: JsonObject = {
                    key: symbol.get(key)
                    for key in (
                        "qname",
                        "kind",
                        "symbol_id",
                        "docfacts",
                        "quality",
                        "metrics",
                        "agent_hints",
                        "change_impact",
                        "exemplars",
                    )
                }
                symbol_rows.append(
                    (
                        symbol_id,
                        package_name,
                        qualified,
                        qname_value,
                        _stringify(symbol_payload.get("kind")) or "object",
                        _json_dumps(symbol_payload),
                    )
                )
                anchors = symbol.get("anchors")
                anchor_payload = anchors if isinstance(anchors, Mapping) else {}
                anchor_rows.append(
                    (
                        symbol_id,
                        anchor_payload.get("start_line"),
                        anchor_payload.get("end_line"),
                        anchor_payload.get("cst_fingerprint"),
                        _json_dumps(anchor_payload.get("remap_order")),
                    )
                )
                quality = symbol.get("quality")
                metrics = symbol.get("metrics")
                change_impact = symbol.get("change_impact")
                coverage = (
                    quality.get("docstring_coverage") if isinstance(quality, Mapping) else None
                )
                complexity = metrics.get("complexity") if isinstance(metrics, Mapping) else None
                churn = (
                    change_impact.get("churn_last_n")
                    if isinstance(change_impact, Mapping)
                    else None
                )
                stability = metrics.get("stability") if isinstance(metrics, Mapping) else None
                deprecated = 1 if isinstance(metrics, Mapping) and metrics.get("deprecated") else 0
                ranking_rows.append((symbol_id, coverage, complexity, churn, stability, deprecated))
    return _SymbolExtraction(module_rows, symbol_rows, anchor_rows, ranking_rows, symbol_lookup)


def _collect_call_rows(
    packages: Sequence[Mapping[str, JsonValue]],
    symbol_lookup: Mapping[str, str],
) -> list[CallRow]:
    """Return call graph rows using symbol identifiers where available.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    packages : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``packages``.
    symbol_lookup : str | str
        Describe ``symbol_lookup``.

    Returns
    -------
    list[tuple[str | NoneType, str | NoneType, str, str | NoneType, str | NoneType]]
        Describe return value.
"""
    rows: list[CallRow] = []
    for package in packages:
        for module in _iter_modules(package):
            graph = module.get("graph")
            if not isinstance(graph, Mapping):
                continue
            calls = graph.get("calls")
            if not isinstance(calls, Sequence):
                continue
            for edge in calls:
                if not isinstance(edge, Mapping):
                    continue
                caller_qname = _stringify(edge.get("caller"))
                callee_qname = _stringify(edge.get("callee"))
                if not callee_qname:
                    continue
                rows.append(
                    (
                        symbol_lookup.get(caller_qname or ""),
                        symbol_lookup.get(callee_qname),
                        callee_qname,
                        _stringify(edge.get("confidence")),
                        _stringify(edge.get("kind")),
                    )
                )
    return rows


def _build_package_rows(packages: Sequence[Mapping[str, JsonValue]]) -> list[tuple[str, str]]:
    """Return serialized package rows.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    packages : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``packages``.

    Returns
    -------
    list[tuple[str, str]]
        Describe return value.
"""
    rows: list[tuple[str, str]] = []
    for package in packages:
        name = _stringify(package.get("name")) or ""
        rows.append((name, _json_dumps({"name": name})))
    return rows


def _build_fts_rows(documents: Sequence[SearchDocument]) -> list[FtsRow]:
    """Return FTS rows from search documents.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    documents : SearchDocument
        Describe ``documents``.

    Returns
    -------
    list[tuple[str, str, str, str, str]]
        Describe return value.
"""
    rows: list[FtsRow] = []
    for document in documents:
        text = " ".join(part for part in (document.summary, document.docstring) if part)
        rows.append(
            (
                document.symbol_id,
                document.package,
                document.module,
                document.qname,
                text or document.text,
            )
        )
    return rows


def _write_many(
    connection: sqlite3.Connection, sql: str, rows: Sequence[tuple[JsonValue, ...]]
) -> None:
    """Execute ``executemany`` when rows are provided.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.
    sql : str
        Describe ``sql``.
    rows : tuple[NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue], ...]
        Describe ``rows``.
"""
    if rows:
        connection.executemany(sql, rows)


def write_sqlite_catalog(
    payload: CatalogPayload,
    path: Path,
    *,
    packages_override: Sequence[Mapping[str, JsonValue]] | None = None,
) -> None:
    """Persist ``payload`` as an optimised SQLite catalogue.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    payload : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
        Describe ``payload``.
    path : Path
        Describe ``path``.
    packages_override : str | NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | NoneType, optional
        Describe ``packages_override``.
        Defaults to ``None``.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    packages = (
        list(packages_override) if packages_override is not None else list(_iter_packages(payload))
    )
    payload_for_docs: dict[str, JsonValue] = dict(payload)
    payload_for_docs["packages"] = list(packages)
    documents = list(documents_from_catalog(payload_for_docs))
    metadata_rows = _build_metadata_rows(payload)
    symbol_rows = _collect_symbol_rows(packages)
    fts_rows = _build_fts_rows(documents)
    call_rows = _collect_call_rows(packages, symbol_rows.lookup)
    with _sqlite_connection(path) as connection:
        connection.executescript(DDL)
        _write_many(connection, "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata_rows)
        _write_many(
            connection,
            "INSERT INTO packages(name, data) VALUES (?, ?)",
            _build_package_rows(packages),
        )
        _write_many(
            connection,
            "INSERT INTO modules(qualified, package, data) VALUES (?, ?, ?)",
            symbol_rows.modules,
        )
        _write_many(
            connection,
            """
            INSERT INTO symbols(symbol_id, package, module, qname, kind, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            symbol_rows.symbols,
        )
        _write_many(
            connection,
            "INSERT INTO anchors(symbol_id, start_line, end_line, cst_fingerprint, remap_order) VALUES (?, ?, ?, ?, ?)",
            symbol_rows.anchors,
        )
        _write_many(
            connection,
            "INSERT INTO ranking_features(symbol_id, coverage, complexity, churn, stability, deprecated) VALUES (?, ?, ?, ?, ?, ?)",
            symbol_rows.ranking,
        )
        _write_many(
            connection,
            "INSERT INTO symbol_fts(symbol_id, package, module, qname, text) VALUES (?, ?, ?, ?, ?)",
            fts_rows,
        )
        _write_many(
            connection,
            "INSERT OR IGNORE INTO calls(caller_symbol_id, callee_symbol_id, callee_qname, confidence, kind) VALUES (?, ?, ?, ?, ?)",
            call_rows,
        )
        connection.commit()


def _fetch_metadata(connection: sqlite3.Connection) -> JsonObject:
    """Return metadata rows keyed by attribute name.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.

    Returns
    -------
    dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]
        Describe return value.
"""
    return {
        row["key"]: _json_loads(row["value"])
        for row in connection.execute("SELECT key, value FROM metadata")
    }


def _load_packages_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Load package rows and attach empty module containers.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.

    Returns
    -------
    dict[str, dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]]
        Describe return value.
"""
    packages: dict[str, JsonObject] = {}
    for row in connection.execute("SELECT name, data FROM packages ORDER BY name"):
        package_data = _json_loads(row["data"])
        data: JsonObject = (
            dict(package_data) if isinstance(package_data, dict) else {"name": row["name"]}
        )
        data.setdefault("modules", [])
        packages[row["name"]] = data
    return packages


def _load_modules_table(
    connection: sqlite3.Connection,
    packages: dict[str, JsonObject],
) -> dict[str, JsonObject]:
    """Load module rows and attach them to their packages.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.
    packages : dict[str, dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]]
        Describe ``packages``.

    Returns
    -------
    dict[str, dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]]
        Describe return value.
"""
    modules: dict[str, JsonObject] = {}
    for row in connection.execute(
        "SELECT qualified, package, data FROM modules ORDER BY qualified"
    ):
        module_data = _json_loads(row["data"])
        if isinstance(module_data, dict):
            data = dict(module_data)
        else:
            data = {"qualified": row["qualified"]}
        data.setdefault("symbols", [])
        modules[row["qualified"]] = data
        package_entry = packages.get(row["package"])
        if package_entry is not None:
            module_list = package_entry.setdefault("modules", [])
            if isinstance(module_list, list):
                module_list.append(data)
    return modules


def _load_anchor_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Return anchor metadata keyed by symbol identifier.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.

    Returns
    -------
    dict[str, dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]]
        Describe return value.
"""
    anchors: dict[str, JsonObject] = {}
    for row in connection.execute("SELECT * FROM anchors"):
        anchors[row["symbol_id"]] = {
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "cst_fingerprint": row["cst_fingerprint"],
            "remap_order": _json_loads(row["remap_order"]),
        }
    return anchors


def _load_ranking_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Return ranking metadata keyed by symbol identifier.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.

    Returns
    -------
    dict[str, dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]]
        Describe return value.
"""
    ranking: dict[str, JsonObject] = {}
    for row in connection.execute("SELECT * FROM ranking_features"):
        ranking[row["symbol_id"]] = {
            "docstring_coverage": row["coverage"],
            "complexity": row["complexity"],
            "churn_last_n": row["churn"],
            "stability": row["stability"],
            "deprecated": bool(row["deprecated"]),
        }
    return ranking


def _load_call_graph(connection: sqlite3.Connection) -> CallGraph:
    """Return caller/callee relationships keyed by symbol identifier.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.

    Returns
    -------
    tuple[str | set[str], str | set[str]]
        Describe return value.
"""
    callers: dict[str, set[str]] = defaultdict(set)
    callees: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        "SELECT caller_symbol_id, callee_symbol_id, callee_qname FROM calls"
    ):
        caller = row["caller_symbol_id"]
        callee = row["callee_symbol_id"] or row["callee_qname"]
        if caller:
            callees[caller].add(callee)
        if callee and caller:
            callers[callee].add(caller)
    return callers, callees


def _attach_symbols_to_modules(
    connection: sqlite3.Connection,
    modules: Mapping[str, JsonObject],
    anchors: Mapping[str, JsonObject],
    ranking: Mapping[str, JsonObject],
    call_graph: CallGraph,
) -> None:
    """Populate module dictionaries with symbol payloads from SQLite rows.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    connection : sqlite3.Connection
        Describe ``connection``.
    modules : str | dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]
        Describe ``modules``.
    anchors : str | dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]
        Describe ``anchors``.
    ranking : str | dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]
        Describe ``ranking``.
    call_graph : tuple[str | set[str], str | set[str]]
        Describe ``call_graph``.
"""
    callers, callees = call_graph
    for row in connection.execute(
        "SELECT symbol_id, package, module, qname, kind, data FROM symbols"
    ):
        symbol_data = _json_loads(row["data"])
        if not isinstance(symbol_data, dict):
            continue
        symbol_id = row["symbol_id"]
        symbol_payload = dict(symbol_data)
        symbol_payload.setdefault("anchors", anchors.get(symbol_id, {}))
        ranking_entry = ranking.get(symbol_id, {})
        metrics = symbol_payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics.setdefault("complexity", ranking_entry.get("complexity"))
        metrics.setdefault("stability", ranking_entry.get("stability"))
        metrics.setdefault("deprecated", ranking_entry.get("deprecated"))
        symbol_payload["metrics"] = metrics
        change_impact = symbol_payload.get("change_impact")
        if not isinstance(change_impact, dict):
            change_impact = {}
        change_impact.setdefault("callers", sorted(callers.get(symbol_id, [])))
        change_impact.setdefault("callees", sorted(callees.get(symbol_id, [])))
        change_impact.setdefault("churn_last_n", ranking_entry.get("churn_last_n"))
        symbol_payload["change_impact"] = change_impact
        module_entry = modules.get(row["module"])
        if module_entry is None:
            continue
        symbols = module_entry.setdefault("symbols", [])
        if isinstance(symbols, list):
            symbols.append(symbol_payload)


def load_catalog_from_sqlite(path: Path) -> JsonObject:
    """Load a catalogue payload from ``path`` and materialise as dictionaries.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Describe ``path``.

    Returns
    -------
    dict[str, NoneType | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]]
        Describe return value.
"""
    with _sqlite_connection(path) as connection:
        metadata = _fetch_metadata(connection)
        packages = _load_packages_table(connection)
        modules = _load_modules_table(connection, packages)
        anchors = _load_anchor_table(connection)
        ranking = _load_ranking_table(connection)
        call_graph = _load_call_graph(connection)
        _attach_symbols_to_modules(connection, modules, anchors, ranking, call_graph)
        return {
            "version": metadata.get("version"),
            "generated_at": metadata.get("generated_at"),
            "repo": metadata.get("repo") or {},
            "link_policy": metadata.get("link_policy") or {},
            "artifacts": metadata.get("artifacts") or {},
            "packages": list(packages.values()),
            "shards": metadata.get("shards"),
            "semantic_index": metadata.get("semantic_index"),
            "search": metadata.get("search"),
        }


def sqlite_candidates(json_path: Path) -> Iterable[Path]:
    """Yield plausible SQLite catalogue locations for ``json_path``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    json_path : Path
        Describe ``json_path``.

    Returns
    -------
    Path
        Describe return value.
"""
    if json_path.suffix == ".sqlite":
        yield json_path
        return
    try:
        env_value = environ.get("CATALOG_ROOT")
    except OSError:  # pragma: no cover - restricted environments may block env access
        env_value = None
    if env_value:
        env_path = Path(env_value)
        yield env_path / "catalog.sqlite"
    catalog_root = Path(json_path.parent)
    for raw in (json_path.with_suffix(".sqlite"), catalog_root / "catalog.sqlite"):
        yield Path(raw)


__all__ = [
    "load_catalog_from_sqlite",
    "sqlite_candidates",
    "write_sqlite_catalog",
]
