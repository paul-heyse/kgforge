"""Persistence helpers for storing the Agent Catalog in SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, assert_never, cast

from kgfoundry.agent_catalog.search import documents_from_catalog
from kgfoundry_common.errors import SymbolAttachmentError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from kgfoundry.agent_catalog.search import SearchDocument

JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
JsonObject = dict[str, JsonValue]

SearchPayloadValue = str | int | float | bool | list[object] | dict[str, object] | None
SearchPayload = Mapping[str, SearchPayloadValue]


def _to_json_object(mapping: Mapping[str, JsonValue]) -> JsonObject:
    """Create a JSON object copy from ``mapping``.

    Parameters
    ----------
    mapping : Mapping[str, JsonValue]
        Source mapping to copy.

    Returns
    -------
    JsonObject
        Copy of the mapping as a dictionary.
    """
    return dict(mapping)


def _to_search_payload(value: JsonValue) -> SearchPayloadValue:
    """Convert ``value`` into a search payload compatible structure.

    Parameters
    ----------
    value : JsonValue
        Value to convert.

    Returns
    -------
    SearchPayloadValue
        Converted value compatible with search payload structure.
    """
    if isinstance(value, list):
        return [cast("object", _to_search_payload(item)) for item in value]
    if isinstance(value, dict):
        return {key: cast("object", _to_search_payload(item)) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    assert_never(value)


PackagePayload = Mapping[str, JsonValue]
ModulePayload = Mapping[str, JsonValue]
SymbolPayload = Mapping[str, JsonValue]
CallEdgePayload = Mapping[str, JsonValue]

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

    Creates a SQLite connection with foreign keys enabled for safe
    relational operations. Ensures the connection is closed on exit.

    Parameters
    ----------
    path : Path
        Path to SQLite database file.

    Yields
    ------
    sqlite3.Connection
        SQLite connection with foreign keys enabled.
    """
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
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

    Serializes a JSON value to a string with sorted keys for deterministic
    output. Preserves non-ASCII characters.

    Parameters
    ----------
    value : JsonValue
        JSON value to serialize.

    Returns
    -------
    str
        JSON string representation of the value.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None) -> JsonValue:
    """Return JSON-decoded ``value`` handling ``NULL`` gracefully.

    Deserializes a JSON string, returning None if the input is None.

    Parameters
    ----------
    value : str | None
        JSON string to decode, or None.

    Returns
    -------
    JsonValue
        Decoded JSON value, or None if input is None.
    """
    if value is None:
        return None
    return cast("JsonValue", json.loads(value))


def _stringify(value: JsonValue) -> str | None:
    """Return ``value`` as a string if possible.

    Converts a JSON value to a string representation. Returns None if
    the value is None, otherwise converts to string or JSON.

    Parameters
    ----------
    value : JsonValue
        Value to convert to string.

    Returns
    -------
    str | None
        String representation of value, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _json_dumps(value)


def _iter_packages(payload: CatalogPayload) -> list[PackagePayload]:
    """Return package payload mappings extracted from ``payload``.

    Parameters
    ----------
    payload : CatalogPayload
        Catalog payload to extract packages from.

    Returns
    -------
    list[PackagePayload]
        List of package payload mappings.
    """
    packages = payload.get("packages")
    if not isinstance(packages, Sequence):
        return []
    return [cast("PackagePayload", item) for item in packages if isinstance(item, Mapping)]


def _iter_modules(package: PackagePayload) -> list[ModulePayload]:
    """Return module payloads for ``package``.

    Parameters
    ----------
    package : PackagePayload
        Package payload to extract modules from.

    Returns
    -------
    list[ModulePayload]
        List of module payload mappings.
    """
    modules = package.get("modules")
    if not isinstance(modules, Sequence):
        return []
    return [cast("ModulePayload", module) for module in modules if isinstance(module, Mapping)]


def _iter_symbols(module: ModulePayload) -> list[SymbolPayload]:
    """Return typed symbol payloads from ``module``.

    Parameters
    ----------
    module : ModulePayload
        Module payload to extract symbols from.

    Returns
    -------
    list[SymbolPayload]
        List of symbol payload mappings.
    """
    symbols = module.get("symbols")
    if not isinstance(symbols, Sequence):
        return []
    return [cast("SymbolPayload", symbol) for symbol in symbols if isinstance(symbol, Mapping)]


def _iter_call_edges(module: ModulePayload) -> list[CallEdgePayload]:
    """Return typed call edges from a module graph.

    Parameters
    ----------
    module : ModulePayload
        Module payload to extract call edges from.

    Returns
    -------
    list[CallEdgePayload]
        List of call edge payload mappings.
    """
    graph = module.get("graph")
    if not isinstance(graph, Mapping):
        return []
    calls = graph.get("calls")
    if not isinstance(calls, Sequence):
        return []
    return [cast("CallEdgePayload", edge) for edge in calls if isinstance(edge, Mapping)]


def _build_metadata_rows(payload: CatalogPayload) -> list[tuple[str, str]]:
    """Return rows for the ``metadata`` table.

    Extracts metadata fields from the catalog payload and returns them
    as key-value tuples for database insertion.

    Parameters
    ----------
    payload : CatalogPayload
        Catalog payload containing metadata fields.

    Returns
    -------
    list[tuple[str, str]]
        List of (key, value) tuples for metadata table.
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

    Aggregates extracted rows for modules, symbols, anchors, ranking,
    and lookup mapping from symbol processing.

    Parameters
    ----------
    modules : list[ModuleRow]
        Module rows extracted from packages.
    symbols : list[SymbolRow]
        Symbol rows extracted from modules.
    anchors : list[AnchorRow]
        Anchor rows for symbol source locations.
    ranking : list[RankingRow]
        Ranking feature rows for symbols.
    lookup : dict[str, str]
        Mapping from qname to symbol_id.
    """

    modules: list[ModuleRow]
    symbols: list[SymbolRow]
    anchors: list[AnchorRow]
    ranking: list[RankingRow]
    lookup: dict[str, str]


@dataclass(slots=True)
class _SqliteRows:
    """Aggregated row sets for writing the SQLite catalog."""

    metadata: list[tuple[str, str]]
    packages: list[tuple[str, str]]
    modules: list[ModuleRow]
    symbols: list[SymbolRow]
    anchors: list[AnchorRow]
    ranking: list[RankingRow]
    fts: list[FtsRow]
    calls: list[CallRow]


def _collect_symbol_rows(packages: Sequence[PackagePayload]) -> _SymbolExtraction:
    """Aggregate module, symbol, anchor, and ranking rows.

    Iterates through packages and modules to extract all symbol-related
    rows for database insertion.

    Parameters
    ----------
    packages : Sequence[PackagePayload]
        Sequence of package payloads to process.

    Returns
    -------
    _SymbolExtraction
        Aggregated rows for modules, symbols, anchors, ranking, and lookup.
    """
    extraction = _SymbolExtraction([], [], [], [], {})
    for package in packages:
        package_name = _stringify(package.get("name")) or ""
        for module in _iter_modules(package):
            _accumulate_module_rows(extraction, package_name, module)
    return extraction


def _accumulate_module_rows(
    extraction: _SymbolExtraction,
    package_name: str,
    module: ModulePayload,
) -> None:
    qualified = _stringify(module.get("qualified")) or _stringify(module.get("name")) or ""
    module_payload: JsonObject = {
        "name": module.get("name"),
        "qualified": qualified,
        "source": module.get("source"),
        "pages": module.get("pages"),
        "imports": module.get("imports"),
        "graph": module.get("graph"),
    }
    extraction.modules.append((qualified, package_name, _json_dumps(module_payload)))
    for symbol in _iter_symbols(module):
        _accumulate_symbol_rows(extraction, package_name, qualified, symbol)


def _accumulate_symbol_rows(
    extraction: _SymbolExtraction,
    package_name: str,
    qualified_module: str,
    symbol: SymbolPayload,
) -> None:
    symbol_id = _stringify(symbol.get("symbol_id")) or ""
    qname_value = _stringify(symbol.get("qname")) or symbol_id
    if qname_value:
        extraction.lookup[qname_value] = symbol_id
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
    extraction.symbols.append(
        (
            symbol_id,
            package_name,
            qualified_module,
            qname_value,
            _stringify(symbol_payload.get("kind")) or "object",
            _json_dumps(symbol_payload),
        )
    )
    _append_anchor_row(extraction, symbol_id, symbol)
    _append_ranking_row(extraction, symbol_id, symbol)


def _append_anchor_row(
    extraction: _SymbolExtraction,
    symbol_id: str,
    symbol: SymbolPayload,
) -> None:
    anchors = symbol.get("anchors")
    anchor_payload = anchors if isinstance(anchors, Mapping) else {}
    extraction.anchors.append(
        (
            symbol_id,
            anchor_payload.get("start_line"),
            anchor_payload.get("end_line"),
            anchor_payload.get("cst_fingerprint"),
            _json_dumps(anchor_payload.get("remap_order")),
        )
    )


def _append_ranking_row(
    extraction: _SymbolExtraction,
    symbol_id: str,
    symbol: SymbolPayload,
) -> None:
    quality_raw = symbol.get("quality")
    quality = quality_raw if isinstance(quality_raw, Mapping) else None
    metrics_raw = symbol.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, Mapping) else None
    change_impact_raw = symbol.get("change_impact")
    change_impact = change_impact_raw if isinstance(change_impact_raw, Mapping) else None
    coverage = quality.get("docstring_coverage") if isinstance(quality, Mapping) else None
    complexity = metrics.get("complexity") if isinstance(metrics, Mapping) else None
    churn = change_impact.get("churn_last_n") if isinstance(change_impact, Mapping) else None
    stability = metrics.get("stability") if isinstance(metrics, Mapping) else None
    deprecated_flag = 1 if isinstance(metrics, Mapping) and metrics.get("deprecated") else 0
    extraction.ranking.append((symbol_id, coverage, complexity, churn, stability, deprecated_flag))


def _collect_call_rows(
    packages: Sequence[PackagePayload],
    symbol_lookup: Mapping[str, str],
) -> list[CallRow]:
    """Return call graph rows using symbol identifiers where available.

    Extracts call edges from module graphs and converts them to database
    rows using symbol lookup to resolve qnames to symbol_ids.

    Parameters
    ----------
    packages : Sequence[PackagePayload]
        Sequence of package payloads containing call graphs.
    symbol_lookup : Mapping[str, str]
        Mapping from qname to symbol_id for resolving identifiers.

    Returns
    -------
    list[CallRow]
        List of call rows (caller_symbol_id, callee_symbol_id, callee_qname,
        confidence, kind).
    """
    rows: list[CallRow] = []
    for package in packages:
        for module in _iter_modules(package):
            for edge in _iter_call_edges(module):
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


def _build_package_rows(packages: Sequence[PackagePayload]) -> list[tuple[str, str]]:
    """Return serialized package rows.

    Extracts package names and serializes them as JSON for database storage.

    Parameters
    ----------
    packages : Sequence[PackagePayload]
        Sequence of package payloads.

    Returns
    -------
    list[tuple[str, str]]
        List of (package_name, json_data) tuples.
    """
    rows: list[tuple[str, str]] = []
    for package in packages:
        name = _stringify(package.get("name")) or ""
        rows.append((name, _json_dumps({"name": name})))
    return rows


def _build_fts_rows(documents: Sequence[SearchDocument]) -> list[FtsRow]:
    """Return FTS rows from search documents.

    Extracts full-text search data from search documents for FTS5 index.

    Parameters
    ----------
    documents : Sequence[SearchDocument]
        Sequence of search documents to index.

    Returns
    -------
    list[FtsRow]
        List of (symbol_id, package, module, qname, text) tuples.
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

    Executes a bulk insert operation only if rows are available.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to execute against.
    sql : str
        SQL INSERT statement with placeholders.
    rows : Sequence[tuple[JsonValue, ...]]
        Sequence of row tuples to insert.
    """
    if rows:
        connection.executemany(sql, rows)


def _resolve_packages(
    payload: CatalogPayload,
    packages_override: Sequence[Mapping[str, JsonValue]] | None,
) -> list[PackagePayload]:
    if packages_override is not None:
        return _coerce_packages(packages_override)
    return _iter_packages(payload)


def _build_search_documents(
    payload: CatalogPayload,
    packages: Sequence[PackagePayload],
) -> list[SearchDocument]:
    payload_for_docs: JsonObject = dict(payload)
    payload_for_docs["packages"] = [dict(pkg) for pkg in packages]
    payload_for_docs_obj: dict[str, SearchPayloadValue] = {
        key: _to_search_payload(value) for key, value in payload_for_docs.items()
    }
    search_payload = cast("SearchPayload", payload_for_docs_obj)
    return list(documents_from_catalog(search_payload))


def _prepare_sqlite_rows(
    payload: CatalogPayload,
    packages: Sequence[PackagePayload],
    documents: Sequence[SearchDocument],
) -> _SqliteRows:
    symbol_rows = _collect_symbol_rows(packages)
    return _SqliteRows(
        metadata=_build_metadata_rows(payload),
        packages=_build_package_rows(packages),
        modules=symbol_rows.modules,
        symbols=symbol_rows.symbols,
        anchors=symbol_rows.anchors,
        ranking=symbol_rows.ranking,
        fts=_build_fts_rows(documents),
        calls=_collect_call_rows(packages, symbol_rows.lookup),
    )


def write_sqlite_catalog(
    payload: CatalogPayload,
    path: Path,
    *,
    packages_override: Sequence[Mapping[str, JsonValue]] | None = None,
) -> None:
    """Persist ``payload`` as an optimised SQLite catalogue.

    Creates a SQLite database from catalog payload, including all tables
    for packages, modules, symbols, anchors, ranking features, FTS index,
    and call graph.

    Parameters
    ----------
    payload : CatalogPayload
        Catalog payload to persist.
    path : Path
        Path where SQLite database will be created.
    packages_override : Sequence[Mapping[str, JsonValue]] | None, optional
        Optional override for packages list. Defaults to None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    packages = _resolve_packages(payload, packages_override)
    documents = _build_search_documents(payload, packages)
    rows = _prepare_sqlite_rows(payload, packages, documents)
    with _sqlite_connection(path) as connection:
        connection.executescript(DDL)
        _write_many(connection, "INSERT INTO metadata(key, value) VALUES (?, ?)", rows.metadata)
        _write_many(
            connection,
            "INSERT INTO packages(name, data) VALUES (?, ?)",
            rows.packages,
        )
        _write_many(
            connection,
            "INSERT INTO modules(qualified, package, data) VALUES (?, ?, ?)",
            rows.modules,
        )
        _write_many(
            connection,
            """
            INSERT INTO symbols(symbol_id, package, module, qname, kind, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows.symbols,
        )
        _write_many(
            connection,
            "INSERT INTO anchors(symbol_id, start_line, end_line, cst_fingerprint, remap_order) VALUES (?, ?, ?, ?, ?)",
            rows.anchors,
        )
        _write_many(
            connection,
            "INSERT INTO ranking_features(symbol_id, coverage, complexity, churn, stability, deprecated) VALUES (?, ?, ?, ?, ?, ?)",
            rows.ranking,
        )
        _write_many(
            connection,
            "INSERT INTO symbol_fts(symbol_id, package, module, qname, text) VALUES (?, ?, ?, ?, ?)",
            rows.fts,
        )
        _write_many(
            connection,
            "INSERT OR IGNORE INTO calls(caller_symbol_id, callee_symbol_id, callee_qname, confidence, kind) VALUES (?, ?, ?, ?, ?)",
            rows.calls,
        )
        connection.commit()


def _fetch_metadata(connection: sqlite3.Connection) -> JsonObject:
    """Return metadata rows keyed by attribute name.

    Loads metadata key-value pairs from the database.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.

    Returns
    -------
    JsonObject
        Dictionary of metadata key-value pairs.
    """
    result: JsonObject = {}
    rows = cast(
        "list[tuple[str, str]]", connection.execute("SELECT key, value FROM metadata").fetchall()
    )
    for key, value_str in rows:
        result[key] = _json_loads(value_str)
    return result


def _load_packages_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Load package rows and attach empty module containers.

    Loads packages from the database and initializes empty module lists
    for each package.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.

    Returns
    -------
    dict[str, JsonObject]
        Dictionary mapping package names to package payloads with empty
        module lists.
    """
    packages: dict[str, JsonObject] = {}
    rows = cast(
        "list[tuple[str, str]]",
        connection.execute("SELECT name, data FROM packages ORDER BY name").fetchall(),
    )
    for package_name, data_raw in rows:
        package_data = _json_loads(data_raw)
        data: JsonObject
        if isinstance(package_data, dict):
            data = _to_json_object(cast("Mapping[str, JsonValue]", package_data))
        else:
            data = cast("JsonObject", {"name": package_name})
        modules_list: list[JsonObject] = [
            _to_json_object(module) for module in _iter_modules(cast("PackagePayload", data))
        ]
        data["modules"] = cast("JsonValue", modules_list)
        packages[package_name] = data
    return packages


def _load_modules_table(
    connection: sqlite3.Connection,
    packages: dict[str, JsonObject],
) -> dict[str, JsonObject]:
    """Load module rows and attach them to their packages.

    Loads modules from the database and attaches them to their parent
    packages, initializing empty symbol lists.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.
    packages : dict[str, JsonObject]
        Dictionary of package payloads to update.

    Returns
    -------
    dict[str, JsonObject]
        Dictionary mapping module qualified names to module payloads.
    """
    modules: dict[str, JsonObject] = {}
    rows = cast(
        "list[tuple[str, str, str]]",
        connection.execute(
            "SELECT qualified, package, data FROM modules ORDER BY qualified"
        ).fetchall(),
    )
    for qualified, package_name, data_raw in rows:
        module_data = _json_loads(data_raw)
        data: JsonObject
        if isinstance(module_data, dict):
            data = _to_json_object(cast("Mapping[str, JsonValue]", module_data))
        else:
            data = cast("JsonObject", {"qualified": qualified})
        symbols_list: list[JsonObject] = [
            _to_json_object(symbol) for symbol in _iter_symbols(cast("ModulePayload", data))
        ]
        data["symbols"] = cast("JsonValue", symbols_list)
        modules[qualified] = data
        package_entry = packages.get(package_name)
        if package_entry is not None:
            module_list_value = package_entry.get("modules")
            module_list: list[JsonObject]
            if isinstance(module_list_value, list):
                module_list = cast("list[JsonObject]", module_list_value)
            else:
                module_list = cast("list[JsonObject]", [])
                package_entry["modules"] = cast("JsonValue", module_list)
            module_list.append(data)
    return modules


def _load_anchor_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Return anchor metadata keyed by symbol identifier.

    Loads anchor information (start_line, end_line, fingerprint, remap_order)
    from the database.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.

    Returns
    -------
    dict[str, JsonObject]
        Dictionary mapping symbol_id to anchor metadata.
    """
    anchors: dict[str, JsonObject] = {}
    rows = cast(
        "list[tuple[str, int | None, int | None, str | None, str | None]]",
        connection.execute(
            "SELECT symbol_id, start_line, end_line, cst_fingerprint, remap_order FROM anchors"
        ).fetchall(),
    )
    for symbol_id, start_raw, end_raw, fingerprint_raw, remap_raw in rows:
        anchors[symbol_id] = {
            "start_line": start_raw,
            "end_line": end_raw,
            "cst_fingerprint": fingerprint_raw,
            "remap_order": _json_loads(remap_raw),
        }
    return anchors


def _load_ranking_table(connection: sqlite3.Connection) -> dict[str, JsonObject]:
    """Return ranking metadata keyed by symbol identifier.

    Loads ranking features (coverage, complexity, churn, stability, deprecated)
    from the database.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.

    Returns
    -------
    dict[str, JsonObject]
        Dictionary mapping symbol_id to ranking metadata.
    """
    ranking: dict[str, JsonObject] = {}
    rows = cast(
        "list[tuple[str, float | None, float | None, int | None, str | None, int]]",
        connection.execute(
            "SELECT symbol_id, coverage, complexity, churn, stability, deprecated FROM ranking_features"
        ).fetchall(),
    )
    for symbol_id, coverage_raw, complexity_raw, churn_raw, stability_raw, deprecated_raw in rows:
        ranking[symbol_id] = {
            "docstring_coverage": coverage_raw,
            "complexity": complexity_raw,
            "churn_last_n": churn_raw,
            "stability": stability_raw,
            "deprecated": bool(deprecated_raw),
        }
    return ranking


def _load_call_graph(connection: sqlite3.Connection) -> CallGraph:
    """Return caller/callee relationships keyed by symbol identifier.

    Loads call graph edges from the database and builds bidirectional
    mappings for callers and callees.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query.

    Returns
    -------
    CallGraph
        Tuple of (callers mapping, callees mapping) where each maps
        symbol_id to sets of related symbol IDs.
    """
    callers: dict[str, set[str]] = defaultdict(set)
    callees: dict[str, set[str]] = defaultdict(set)
    rows = cast(
        "list[tuple[str | None, str | None, str]]",
        connection.execute(
            "SELECT caller_symbol_id, callee_symbol_id, callee_qname FROM calls"
        ).fetchall(),
    )
    for caller, callee_id, callee_qname in rows:
        callee = callee_id or callee_qname
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

    Loads symbols from the database and attaches them to their modules,
    enriching with anchor, ranking, and call graph data.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite connection to query symbols from.
    modules : Mapping[str, JsonObject]
        Dictionary of module payloads to update.
    anchors : Mapping[str, JsonObject]
        Dictionary of anchor metadata by symbol_id.
    ranking : Mapping[str, JsonObject]
        Dictionary of ranking metadata by symbol_id.
    call_graph : CallGraph
        Tuple of (callers, callees) mappings.

    Raises
    ------
    SymbolAttachmentError
        If database query fails.
    """
    callers, callees = call_graph
    try:
        rows = cast(
            "list[tuple[str, str, str, str, str, str]]",
            connection.execute(
                "SELECT symbol_id, package, module, qname, kind, data FROM symbols"
            ).fetchall(),
        )
    except sqlite3.DatabaseError as e:
        msg = "Failed to query symbols from database"
        raise SymbolAttachmentError(msg, cause=e) from e

    for symbol_id, _package_name, module_name, _qname, _kind, data_raw in rows:
        try:
            symbol_data = _json_loads(data_raw)
        except ValueError as e:
            logger.warning("Skipping symbol %s with invalid JSON data: %s", symbol_id, e)
            continue

        if not isinstance(symbol_data, dict):
            continue

        symbol_payload = _to_json_object(cast("Mapping[str, JsonValue]", symbol_data))

        # Attach anchor data
        if "anchors" not in symbol_payload:
            symbol_payload["anchors"] = anchors.get(symbol_id, {})

        # Assemble metrics from ranking and symbol payload
        ranking_entry = ranking.get(symbol_id, {})
        _attach_metrics(symbol_payload, ranking_entry)

        # Assemble change impact from call graph
        _attach_change_impact(symbol_payload, symbol_id, callers, callees, ranking_entry)

        # Add symbol to module
        _append_symbol_to_module(modules, module_name, symbol_payload)


def _attach_metrics(symbol_payload: JsonObject, ranking_entry: JsonObject) -> None:
    """Attach ranking metrics to symbol payload.

    Merges ranking features into the symbol's metrics dictionary.

    Parameters
    ----------
    symbol_payload : JsonObject
        Symbol payload dict to update (modified in-place).
    ranking_entry : JsonObject
        Ranking entry containing complexity, stability, and deprecated flags.
    """
    metrics = symbol_payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    if "complexity" not in metrics:
        metrics["complexity"] = ranking_entry.get("complexity")
    if "stability" not in metrics:
        metrics["stability"] = ranking_entry.get("stability")
    if "deprecated" not in metrics:
        metrics["deprecated"] = ranking_entry.get("deprecated")

    symbol_payload["metrics"] = metrics


def _attach_change_impact(
    symbol_payload: JsonObject,
    symbol_id: str,
    callers: Mapping[str, set[str]],
    callees: Mapping[str, set[str]],
    ranking_entry: JsonObject,
) -> None:
    """Attach change impact data to symbol payload.

    Merges call graph relationships and churn data into the symbol's
    change_impact dictionary.

    Parameters
    ----------
    symbol_payload : JsonObject
        Symbol payload dict to update (modified in-place).
    symbol_id : str
        ID of the symbol.
    callers : Mapping[str, set[str]]
        Mapping of symbol_id to set of caller IDs.
    callees : Mapping[str, set[str]]
        Mapping of symbol_id to set of callee IDs.
    ranking_entry : JsonObject
        Ranking entry containing churn_last_n information.
    """
    change_impact = symbol_payload.get("change_impact")
    if not isinstance(change_impact, dict):
        change_impact = {}

    if "callers" not in change_impact:
        callers_list = sorted(callers.get(symbol_id, []))
        change_impact["callers"] = cast("JsonValue", callers_list)

    if "callees" not in change_impact:
        callees_list = sorted(callees.get(symbol_id, []))
        change_impact["callees"] = cast("JsonValue", callees_list)

    if "churn_last_n" not in change_impact:
        change_impact["churn_last_n"] = ranking_entry.get("churn_last_n")

    symbol_payload["change_impact"] = change_impact


def _append_symbol_to_module(
    modules: Mapping[str, JsonObject], module_name: str, symbol_payload: JsonObject
) -> None:
    """Append symbol payload to a module's symbol list.

    Adds a symbol to the symbols list of its parent module.

    Parameters
    ----------
    modules : Mapping[str, JsonObject]
        Dictionary of module payloads (modified in-place).
    module_name : str
        Qualified name of the module.
    symbol_payload : JsonObject
        Symbol payload to append.
    """
    module_entry = modules.get(module_name)
    if module_entry is None:
        return

    symbols_value = module_entry.get("symbols")
    if isinstance(symbols_value, list):
        symbol_list = cast("list[JsonObject]", symbols_value)
    else:
        symbol_list = []
        module_entry["symbols"] = cast("JsonValue", symbol_list)

    symbol_list.append(symbol_payload)


def load_catalog_from_sqlite(path: Path) -> JsonObject:
    """Load a catalogue payload from ``path`` as JSON-compatible dictionaries.

    Parameters
    ----------
    path : Path
        Path to SQLite catalog database.

    Returns
    -------
    JsonObject
        Catalog payload as JSON-compatible dictionaries.
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

    Parameters
    ----------
    json_path : Path
        Path to JSON catalog file.

    Yields
    ------
    Path
        Candidate SQLite database paths.
    """
    if json_path.suffix == ".sqlite":
        yield json_path
        return
    try:
        env_value = environ.get("CATALOG_ROOT")
    except OSError:  # pragma: no cover
        env_value = None
    if env_value:
        env_path = Path(env_value)
        yield env_path / "catalog.sqlite"
    catalog_root = Path(json_path.parent)
    for candidate in (json_path.with_suffix(".sqlite"), catalog_root / "catalog.sqlite"):
        yield Path(candidate)


__all__ = [
    "load_catalog_from_sqlite",
    "sqlite_candidates",
    "write_sqlite_catalog",
]


def _coerce_packages(packages: Sequence[Mapping[str, JsonValue]]) -> list[PackagePayload]:
    """Return a list of package payloads from ``packages``.

    Parameters
    ----------
    packages : Sequence[Mapping[str, JsonValue]]
        Sequence of package mappings.

    Returns
    -------
    list[PackagePayload]
        List of package payload mappings.
    """
    return list(packages)
