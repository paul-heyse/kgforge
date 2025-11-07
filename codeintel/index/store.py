"""Persistent SQLite index for symbols and references."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Self

from codeintel.indexer.tscore import get_language, load_langs, parse_bytes, run_query

EXT_TO_LANG = {
    ".py": "python",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".json": "json",
}

EXCLUDES = [
    "**/.git/**",
    "**/.venv/**",
    "**/_build/**",
    "**/__pycache__/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/node_modules/**",
]


@dataclass(frozen=True)
class FileMeta:
    """File metadata for incremental indexing."""

    lang: str
    mtime_ns: int
    size_bytes: int


class IndexStore:
    """SQLite-backed index store for symbols and references."""

    def __init__(self, db_path: Path) -> None:
        """Initialize index store.

        Parameters
        ----------
        db_path : Path
            Path to SQLite database file.
        """
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        """Enter context manager and open database connection.

        Returns
        -------
        IndexStore
            Self for context manager protocol.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager and close database connection."""
        if self.conn:
            self.conn.close()

    def execute(self, sql: str, args: tuple[object, ...] = ()) -> sqlite3.Cursor:
        """Execute SQL query.

        Parameters
        ----------
        sql : str
            SQL query string.
        args : tuple[object, ...], optional
            Query parameters, by default ().

        Returns
        -------
        sqlite3.Cursor
            Query result cursor.

        Raises
        ------
        RuntimeError
            If store is not opened (use as context manager).
        """
        if self.conn is None:
            message = "IndexStore not opened (use as context manager)"
            raise RuntimeError(message)
        return self.conn.execute(sql, args)

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        """Execute SQL query with multiple parameter sets.

        Parameters
        ----------
        sql : str
            SQL query string.
        rows : list[tuple[object, ...]]
            Parameter sets for batch execution.

        Raises
        ------
        RuntimeError
            If store is not opened (use as context manager).
        """
        if self.conn is None:
            message = "IndexStore not opened (use as context manager)"
            raise RuntimeError(message)
        self.conn.executemany(sql, rows)

    def commit(self) -> None:
        """Commit pending transactions.

        Raises
        ------
        RuntimeError
            If store is not opened (use as context manager).
        """
        if self.conn is None:
            message = "IndexStore not opened (use as context manager)"
            raise RuntimeError(message)
        self.conn.commit()


def ensure_schema(store: IndexStore) -> None:
    """Create database schema if it doesn't exist.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.

    Raises
    ------
    RuntimeError
        If store is not opened (use as context manager).
    """
    schema_file = Path(__file__).with_name("schema.sql")
    schema = schema_file.read_text(encoding="utf-8")
    if store.conn is None:
        message = "IndexStore not opened (use as context manager)"
        raise RuntimeError(message)
    store.conn.executescript(schema)
    store.commit()


def detect_lang(path: Path) -> str | None:
    """Detect language from file extension.

    Parameters
    ----------
    path : Path
        File path to inspect.

    Returns
    -------
    str | None
        Language identifier or None if unsupported.
    """
    return EXT_TO_LANG.get(path.suffix.lower())


def stat_meta(path: Path, lang: str) -> FileMeta:
    """Extract file metadata for indexing.

    Parameters
    ----------
    path : Path
        File path to stat.
    lang : str
        Detected language identifier.

    Returns
    -------
    FileMeta
        File metadata record.
    """
    st = path.stat()
    return FileMeta(lang=lang, mtime_ns=st.st_mtime_ns, size_bytes=st.st_size)


def needs_reindex(store: IndexStore, path: Path, meta: FileMeta) -> bool:
    """Check if file needs reindexing based on metadata.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.
    path : Path
        File path to check.
    meta : FileMeta
        Current file metadata.

    Returns
    -------
    bool
        True if file needs reindexing.
    """
    row = store.execute(
        "SELECT mtime_ns, size_bytes FROM files WHERE path=?", (str(path),)
    ).fetchone()
    if not row:
        return True
    return (row[0], row[1]) != (meta.mtime_ns, meta.size_bytes)


def replace_file(store: IndexStore, path: Path, meta: FileMeta) -> None:
    """Replace file's symbols and references in index.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.
    path : Path
        File path to index.
    meta : FileMeta
        File metadata record.

    Notes
    -----
    Deletes existing symbols/refs for the file, then inserts new ones
    extracted via Tree-sitter queries.
    """
    # Delete old rows
    store.execute("DELETE FROM symbols WHERE path=?", (str(path),))
    store.execute("DELETE FROM refs WHERE path=?", (str(path),))
    store.execute(
        "INSERT OR REPLACE INTO files(path, lang, mtime_ns, size_bytes) VALUES(?,?,?,?)",
        (str(path), meta.lang, meta.mtime_ns, meta.size_bytes),
    )

    # Load query file
    queries_dir = Path(__file__).resolve().parents[1] / "queries"
    query_file = queries_dir / f"{meta.lang}.scm"
    if not query_file.exists():
        store.commit()
        return

    data = path.read_bytes()
    langs = load_langs()
    lang = get_language(langs, meta.lang)
    tree = parse_bytes(lang, data)
    query_text = query_file.read_text(encoding="utf-8")
    caps = run_query(lang, query_text, tree, data)

    # Normalize captures to symbols & refs
    sym_rows: list[tuple[object, ...]] = []
    ref_rows: list[tuple[object, ...]] = []
    for c in caps:
        cap = c["capture"]
        row = c["start_point"]["row"]  # 0-based, but we'll store as-is
        if cap == "def.name":
            name = c.get("text", "") or ""
            qual = f"{path}:{name}"
            sym_rows.append(
                (
                    str(path),
                    meta.lang,
                    "function",
                    name,
                    qual,
                    row,
                    row,
                    None,
                    None,
                )
            )
        elif cap == "call.name":
            callee = c.get("text", "")
            ref_rows.append(
                (
                    str(path),
                    meta.lang,
                    "call",
                    f"{path}::<scope?>",
                    callee,
                    row,
                )
            )

    if sym_rows:
        store.executemany(
            """INSERT INTO symbols(path, lang, kind, name, qualname, start_line, end_line, signature, docstring)
                             VALUES(?,?,?,?,?,?,?,?,?)""",
            sym_rows,
        )
    if ref_rows:
        store.executemany(
            """INSERT INTO refs(path, lang, kind, src_qualname, dst_qualname, line)
                             VALUES(?,?,?,?,?,?)""",
            ref_rows,
        )
    store.commit()


def discover_files(root: Path) -> list[Path]:
    """Discover indexable files in repository.

    Parameters
    ----------
    root : Path
        Repository root directory.

    Returns
    -------
    list[Path]
        List of file paths to index.
    """
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(fnmatch(str(p), pat.replace("**/", "")) for pat in EXCLUDES)
    ]


def index_incremental(store: IndexStore, repo_root: Path, *, changed_only: bool = True) -> int:
    """Perform incremental indexing of repository files.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.
    repo_root : Path
        Repository root directory.
    changed_only : bool, optional
        If True, only index changed files, by default True.

    Returns
    -------
    int
        Number of files indexed.
    """
    count = 0
    for p in discover_files(repo_root):
        lang = detect_lang(p)
        if not lang:
            continue
        meta = stat_meta(p, lang)
        if changed_only and not needs_reindex(store, p, meta):
            continue
        replace_file(store, p, meta)
        count += 1
    return count


def search_symbols(
    store: IndexStore,
    query: str,
    kind: str | None = None,
    lang: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Search for symbols matching query pattern.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.
    query : str
        Search pattern (SQL LIKE).
    kind : str | None, optional
        Optional symbol kind filter.
    lang : str | None, optional
        Optional language filter.
    limit : int, optional
        Maximum results, by default 100.

    Returns
    -------
    list[dict[str, object]]
        Symbol records with path, kind, name, qualname, start, end.
    """
    sql = "SELECT path, kind, name, qualname, start_line, end_line FROM symbols WHERE name LIKE ?"
    args: list[object] = [f"%{query}%"]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    if lang:
        sql += " AND lang=?"
        args.append(lang)
    sql += " LIMIT ?"
    args.append(limit)
    return [
        {
            "path": r[0],
            "kind": r[1],
            "name": r[2],
            "qualname": r[3],
            "start": r[4],
            "end": r[5],
        }
        for r in store.execute(sql, tuple(args)).fetchall()
    ]


def find_references(store: IndexStore, qualname: str, limit: int = 100) -> list[dict[str, object]]:
    """Find references to a symbol by qualname.

    Parameters
    ----------
    store : IndexStore
        Open index store instance.
    qualname : str
        Symbol qualname to search for.
    limit : int, optional
        Maximum results, by default 100.

    Returns
    -------
    list[dict[str, object]]
        Reference records with path, kind, src, dst, line.
    """
    sql = (
        "SELECT path, kind, src_qualname, dst_qualname, line FROM refs WHERE dst_qualname=? LIMIT ?"
    )
    return [
        {
            "path": r[0],
            "kind": r[1],
            "src": r[2],
            "dst": r[3],
            "line": r[4],
        }
        for r in store.execute(sql, (qualname, limit)).fetchall()
    ]
