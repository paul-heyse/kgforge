Absolutely—here’s a deep, repo-tailored build-out for **Steps 3 → 5** with copy-pasteable code and the “why” behind each choice. This continues directly from the overall plan you attached. 

---

# 3) Tie the MCP server into your standardized CLI (façade + Typer)

## Goals

* One command to run the MCP stdio server with your **CLI façade** (correlation IDs, envelopes, metrics).
* Friendly developer targets (`make codeintel-serve`) and predictable artifacts.
* Clean hand-off of **repo root** to the server/tools via environment.

### 3.1 Create `codeintel/cli.py`

If you don’t already have a top-level CLI for `codeintel`, add one (this mirrors your façade from other CLIs).

```python
# codeintel/cli.py
from __future__ import annotations
import os
import anyio
import typer
from pathlib import Path

# Your shared façade (names may vary if you customized them)
from tools._shared.cli_integration import cli_operation
from tools._shared.cli_runtime import CliContext, EnvelopeBuilder

# MCP stdio server entry
from codeintel.mcp_server.server import amain as mcp_amain

app = typer.Typer(no_args_is_help=True)
mcp = typer.Typer()
app.add_typer(mcp, name="mcp")

def _infer_repo_root(repo: str | None) -> str:
    return str(Path(repo or ".").resolve())

@mcp.command("serve")
@cli_operation(echo_args=True, echo_env=True)
def serve(ctx: CliContext, env: EnvelopeBuilder, repo: str = ".") -> None:
    """
    Start the CodeIntel MCP server over stdio.
    """
    repo_root = _infer_repo_root(repo)
    # Make repo visible to tools in a predictable way:
    os.environ["KGF_REPO_ROOT"] = repo_root

    # Run server inside AnyIO; façade gives you correlation ids + envelopes
    anyio.run(mcp_amain)
    env.set_result(summary=f"MCP server stopped (repo={repo_root})")
```

> **Why:** We export `KGF_REPO_ROOT` so `mcp_server/tools.py` can use the same root consistently; your façade wraps the run for logs/envelopes/metrics.

### 3.2 (Optional) Add index subcommands here or reuse your existing `indexer/cli.py`

If you want them together:

```python
index = typer.Typer()
app.add_typer(index, name="index")

@index.command("build")
@cli_operation(echo_args=True)
def build_index(ctx: CliContext, env: EnvelopeBuilder, repo: str = ".", fresh: bool = False):
    from codeintel.index.store import IndexStore, ensure_schema, index_incremental
    db = Path(repo) / ".kgf" / "codeintel.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with IndexStore(db) as store:
        ensure_schema(store)
        index_incremental(store, Path(repo), changed_only=not fresh)
    env.add_artifact(kind="index-db", path=db)
    env.set_result(summary=f"Indexed repo={Path(repo).resolve()} into {db}")
```

### 3.3 Makefile helpers

```makefile
.PHONY: codeintel-serve codeintel-index
codeintel-serve:
\tpython -m codeintel.cli mcp serve --repo .
codeintel-index:
\tpython -m codeintel.cli index build --repo .
```

### 3.4 Envelope placement (optional)

If you want CLI envelopes under a canonical route (e.g., `docs/_data/cli/codeintel/mcp/serve/…`), ensure your façade uses the **command path** `["codeintel","mcp","serve"]` when deriving the output directory.

---

# 4) (Optional) Persistent index for symbols & references

You can ship without this; MCP tools can parse files on demand. If you want **fast cross-file search** and `findReferences`, add a light SQLite store.

## 4.1 Storage schema (SQLite)

```sql
-- codeintel/index/schema.sql
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  lang TEXT NOT NULL,
  mtime_ns BIGINT NOT NULL,
  size_bytes BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
  path TEXT NOT NULL,
  lang TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  qualname TEXT NOT NULL,
  start_line INT NOT NULL,
  end_line INT NOT NULL,
  signature TEXT,
  docstring TEXT,
  PRIMARY KEY (path, start_line, kind, name)
);
CREATE TABLE IF NOT EXISTS refs (
  path TEXT NOT NULL,
  lang TEXT NOT NULL,
  kind TEXT NOT NULL,
  src_qualname TEXT NOT NULL,
  dst_qualname TEXT,
  line INT NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_src ON refs(src_qualname);
CREATE INDEX IF NOT EXISTS refs_dst ON refs(dst_qualname);
```

## 4.2 Minimal store API

```python
# codeintel/index/store.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, sqlite3, time

@dataclass
class FileMeta:
    lang: str
    mtime_ns: int
    size_bytes: int

class IndexStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        return self

    def __exit__(self, *exc):
        if self.conn: self.conn.close()

    def execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, args)

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        self.conn.executemany(sql, rows)

    def commit(self) -> None:
        self.conn.commit()

def ensure_schema(store: IndexStore) -> None:
    schema = (Path(__file__).with_name("schema.sql")).read_text()
    store.conn.executescript(schema)
    store.commit()

# --- metadata helpers ---
EXT_TO_LANG = {
    ".py": "python", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".json": "json",
}

def detect_lang(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())

def stat_meta(path: Path, lang: str) -> FileMeta:
    st = path.stat()
    return FileMeta(lang=lang, mtime_ns=st.st_mtime_ns, size_bytes=st.st_size)

def needs_reindex(store: IndexStore, path: Path, meta: FileMeta) -> bool:
    row = store.execute("SELECT mtime_ns, size_bytes FROM files WHERE path=?", (str(path),)).fetchone()
    if not row: return True
    return (row[0], row[1]) != (meta.mtime_ns, meta.size_bytes)
```

## 4.3 Extract & upsert one file

```python
# codeintel/index/store.py (cont.)
from codeintel.indexer.tscore import run_query

def replace_file(store: IndexStore, path: Path, meta: FileMeta) -> None:
    # delete old rows
    store.execute("DELETE FROM symbols WHERE path=?", (str(path),))
    store.execute("DELETE FROM refs WHERE path=?", (str(path),))
    store.execute("INSERT OR REPLACE INTO files(path, lang, mtime_ns, size_bytes) VALUES(?,?,?,?)",
                  (str(path), meta.lang, meta.mtime_ns, meta.size_bytes))

    data = path.read_bytes()
    q = (Path(__file__).parents[1] / "queries" / f"{meta.lang}.scm").read_text()
    caps = run_query(meta.lang, data, q)

    # Normalize captures to symbols & refs (mirror your capture names)
    sym_rows, ref_rows = [], []
    for c in caps:
        cap = c["capture"]
        row = c["start_point"]["row"]  # 1-based
        if cap == "def.name":
            name = c.get("text", "") or ""
            # example: compute qualname as "path::name" (or better: module + class.function)
            qual = f"{path}:{name}"
            sym_rows.append((str(path), meta.lang, "function", name, qual, row, row, None, None))
        elif cap == "call.name":
            callee = c.get("text", "")
            ref_rows.append((str(path), meta.lang, "call", f"{path}::<scope?>", callee, row))

    if sym_rows:
        store.executemany("""INSERT INTO symbols(path, lang, kind, name, qualname, start_line, end_line, signature, docstring)
                             VALUES(?,?,?,?,?,?,?,?,?)""", sym_rows)
    if ref_rows:
        store.executemany("""INSERT INTO refs(path, lang, kind, src_qualname, dst_qualname, line)
                             VALUES(?,?,?,?,?,?)""", ref_rows)
    store.commit()
```

> **Note:** You can refine `qualname` using module/package detection and surrounding class names if you add a light “ancestor stack” during query collection. The above is the minimal working skeleton.

## 4.4 Incremental indexer

```python
# codeintel/index/store.py (cont.)
from fnmatch import fnmatch

EXCLUDES = ["**/.git/**", "**/.venv/**", "**/_build/**", "**/__pycache__/**"]

def discover_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not any(fnmatch(str(p), pat.replace("**/", "")) for pat in EXCLUDES):
            out.append(p)
    return out

def index_incremental(store: IndexStore, repo_root: Path, changed_only: bool = True) -> int:
    count = 0
    for p in discover_files(repo_root):
        lang = detect_lang(p)
        if not lang: continue
        meta = stat_meta(p, lang)
        if changed_only and not needs_reindex(store, p, meta):
            continue
        replace_file(store, p, meta)
        count += 1
    return count
```

## 4.5 “search” and “findReferences” (wired later into MCP if you want)

```python
def search_symbols(store: IndexStore, query: str, kind: str | None = None, lang: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT path, kind, name, qualname, start_line, end_line FROM symbols WHERE name LIKE ?"
    args = [f"%{query}%"]
    if kind: sql += " AND kind=?"; args.append(kind)
    if lang: sql += " AND lang=?"; args.append(lang)
    sql += " LIMIT ?"; args.append(limit)
    return [{"path": r[0], "kind": r[1], "name": r[2], "qualname": r[3], "start": r[4], "end": r[5]}
            for r in store.execute(sql, tuple(args)).fetchall()]

def find_references(store: IndexStore, qualname: str, limit: int = 100) -> list[dict]:
    sql = "SELECT path, kind, src_qualname, dst_qualname, line FROM refs WHERE dst_qualname=? LIMIT ?"
    return [{"path": r[0], "kind": r[1], "src": r[2], "dst": r[3], "line": r[4]}
            for r in store.execute(sql, (qualname, limit)).fetchall()]
```

> **Why SQLite?** Zero-config, fast enough for medium repos, WAL is safe for read-heavy loads. If you prefer DuckDB (consistent with the rest of KGF), you can swap with nearly identical SQL and a `duckdb.connect`.

---

# 5) Security & resource limits (ship-blockers)

Lock these in before exposing the server to clients.

## 5.1 Central config (env + defaults)

Create `codeintel/config.py`:

```python
# codeintel/config.py
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ServerLimits:
    max_ast_bytes: int = int(os.environ.get("CODEINTEL_MAX_AST_BYTES", "1048576"))  # 1 MiB
    max_outline_items: int = int(os.environ.get("CODEINTEL_MAX_OUTLINE_ITEMS", "2000"))
    list_limit_default: int = int(os.environ.get("CODEINTEL_LIMIT_DEFAULT", "100"))
    list_limit_max: int = int(os.environ.get("CODEINTEL_LIMIT_MAX", "1000"))
    tool_timeout_s: float = float(os.environ.get("CODEINTEL_TOOL_TIMEOUT_S", "10.0"))
    rate_limit_qps: float = float(os.environ.get("CODEINTEL_RATE_LIMIT_QPS", "5.0"))
    rate_limit_burst: int = int(os.environ.get("CODEINTEL_RATE_LIMIT_BURST", "10"))
    enable_ts_query: bool = os.environ.get("CODEINTEL_ENABLE_TS_QUERY", "0") == "1"

LIMITS = ServerLimits()
```

Use `LIMITS` throughout `tools.py` and `server.py` instead of magic numbers.

## 5.2 Rate limiting (simple token bucket)

```python
# codeintel/mcp_server/ratelimit.py
from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class TokenBucket:
    rate: float       # tokens per second
    burst: int
    tokens: float = 0.0
    last: float = time.monotonic()

    def acquire(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
```

Wire into `MCPServer`:

```python
# codeintel/mcp_server/server.py (inside MCPServer.__init__)
from codeintel.config import LIMITS
from codeintel.mcp_server.ratelimit import TokenBucket
self._bucket = TokenBucket(rate=LIMITS.rate_limit_qps, burst=LIMITS.rate_limit_burst)
```

Before handling each `"tools/call"`:

```python
if not self._bucket.acquire():
    await self._send_error(request_id, code=-32001, message='{"type":"urn:kgf:problem:rate-limit","title":"Too many requests","status":429,"code":"KGF-CI-RATE"}')
    return
```

## 5.3 Tool timeouts & cancellation

Wrap each handler call with a timeout:

```python
import anyio
...
try:
    with anyio.move_on_after(LIMITS.tool_timeout_s) as scope:
        result = await handler(arguments)
    if scope.cancel_called:
        await self._send_error(request_id, code=-32001, message='{"type":"urn:kgf:problem:timeout","title":"Tool timeout","status":504,"code":"KGF-CI-TIMEOUT"}')
        return
except anyio.get_cancelled_exc_class():
    await self._send_error(request_id, code=-32001, message='{"type":"urn:kgf:problem:cancelled","title":"Cancelled","status":499,"code":"KGF-CANCELLED"}')
    return
```

## 5.4 Gate “advanced” Tree-sitter queries

In the `_tool_ts_query` handler, check `LIMITS.enable_ts_query`. If false, return PD explaining how to enable:

```python
from codeintel.config import LIMITS
if not LIMITS.enable_ts_query:
    return {"status":"error","problem":{"type":"urn:kgf:problem:disabled","title":"TS query disabled","detail":"Set CODEINTEL_ENABLE_TS_QUERY=1 to enable","status":403,"code":"KGF-CI-TSQ-DISABLED"}}
```

## 5.5 Path sandbox (every tool)

You already tightened `_resolve_path()` in `tools.py`. Ensure **every** handler calls it and **never** uses `open()` directly on untrusted input. Add a one-liner helper in `tools.py`:

```python
def repo_relative(p: Path) -> str:
    return str(p.resolve().relative_to(REPO_ROOT))
```

…and return **only** repo-relative paths in results.

## 5.6 Size & result caps

* **AST**: enforce `LIMITS.max_ast_bytes` before parsing; cap traversal nodes (e.g., 200k).
* **Outline**: stop after `LIMITS.max_outline_items`.
* **listFiles/search/findReferences**: enforce `LIMITS.list_limit_max` via `_bounded_limit()`.

---

## Minimal smoke after 3–5

* `python -m codeintel.cli mcp serve --repo .`
  In ChatGPT MCP (or CLI):

  * `tools/list` → shows `code.listFiles`, `code.getFile`, `code.getOutline`, `code.getAST`, `ts.*`
  * `tools/call` `code.listFiles` → returns a bounded list
  * `tools/call` `code.getOutline` → returns a few function/class items for a Python file
  * `tools/call` `code.getAST` → returns bounded AST; large files return PD (size)

* (Optional) `python -m codeintel.cli index build --repo .`
  Then (if you wire search/find into MCP):

  * `tools/call` `code.searchSymbols` → returns rows from SQLite
  * `tools/call` `code.findReferences` → returns ref rows

---

## Quick “ready to commit” checklist

* [ ] `codeintel/cli.py` added; façade wraps `mcp serve`
* [ ] `Makefile` targets `codeintel-serve` / `codeintel-index`
* [ ] `codeintel/index/` with `schema.sql`, `store.py` (ensure_schema, index_incremental, search/find)
* [ ] `codeintel/config.py` with limits & feature flags
* [ ] `codeintel/mcp_server/ratelimit.py` token bucket
* [ ] `mcp_server/server.py` enforces rate limit + timeouts + PD errors; registers new tools
* [ ] `mcp_server/tools.py` uses sandbox, size caps, repo-relative paths only
* [ ] Tests: path sandbox, AST size cap, rate limit timeout, basic round-trip `tools/list`/`code.getOutline`

If you’d like, I can turn this into a small PR patch set (new files + small edits) and a smoke test that spins up the server and exercises `code.health` and `code.getOutline`.
