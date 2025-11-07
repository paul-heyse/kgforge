Awesome — I dug into your zip and you’ve already got a solid base:

* `codeintel/build_languages.py` + `build/languages.json` (manifest-driven language loading)
* `codeintel/indexer/tscore.py` (grammar loading, parse, query)
* `codeintel/mcp_server/server.py` (JSON-RPC stdio loop with `tools/list` + `tools/call`)
* `codeintel/mcp_server/tools.py` (Tree-sitter utilities for defs/calls/errors)
* `codeintel/indexer/cli.py` & `codeintel/indexer/cli_context.py` (Typer + context)

Here’s a **customized, end-to-end elaboration of steps 0–2** with concrete code you can paste and why each piece matters.

---

# 0) Finish the language runtime

### 0.1 Lock the grammar manifest and load consistently

You already have `build_languages.py` and a `build/languages.json` manifest. Make that manifest your **single source of truth** for `indexer/tscore.py`:

```python
# codeintel/indexer/tscore.py  (add near top)
from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from tree_sitter import Language, Parser

ROOT = Path(__file__).resolve().parents[1]
LANG_MANIFEST = ROOT / "build" / "languages.json"

@dataclass(frozen=True)
class LangSpec:
    name: str           # "python"
    module: str         # "tree_sitter_python" or "tree_sitter_languages"
    symbol: str         # "language" or "python" (exported attr name)
    version: str        # distribution version

@dataclass(frozen=True)
class Langs:
    grammars: Mapping[str, Language]   # name -> Language
```

**Loader — read manifest → import module → get Language:**

```python
def _load_manifest() -> list[LangSpec]:
    import json
    data = json.loads(LANG_MANIFEST.read_text())
    out: list[LangSpec] = []
    for item in data.get("languages", []):
        out.append(LangSpec(
            name=item["name"],
            module=item["module"],
            symbol=item.get("symbol", "language"),
            version=item.get("version", "unknown"),
        ))
    return out

@cache
def load_langs() -> Langs:
    grams: dict[str, Language] = {}
    for spec in _load_manifest():
        mod = import_module(spec.module)
        # ‘tree_sitter_*’ wheels usually export “language”
        lang_callable = getattr(mod, spec.symbol, None)
        if not callable(lang_callable):
            raise RuntimeError(f"Language factory not found: {spec.module}.{spec.symbol}")
        grams[spec.name] = lang_callable()
    return Langs(grammars=grams)

@cache
def get_language(name: str) -> Language:
    langs = load_langs()
    try:
        return langs.grammars[name]
    except KeyError as e:
        raise ValueError(f"Unknown language '{name}'. Available={list(langs.grammars)}") from e
```

**Why:** This makes language versions reproducible and ensures every consumer uses the same grammar load path.

### 0.2 Parse once, query many, with safe coordinate extraction

```python
def parse_bytes(lang_name: str, data: bytes) -> tuple[Parser, Any, Any]:
    """Create a parser for `lang_name`, parse `data`, return (parser, tree, root)."""
    parser = Parser()
    parser.set_language(get_language(lang_name))
    tree = parser.parse(data)
    return parser, tree, tree.root_node

def _node_span(node) -> dict[str, Any]:
    # start_point/end_point are (row, column), 0-based; convert to 1-based for UX
    return {
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": {"row": node.start_point[0]+1, "col": node.start_point[1]+1},
        "end_point": {"row": node.end_point[0]+1, "col": node.end_point[1]+1},
    }

def run_query(lang_name: str, data: bytes, query_src: str) -> list[dict[str, Any]]:
    """Compile `query_src` and run it against `data`, returning capture dicts."""
    from tree_sitter import Query
    lang = get_language(lang_name)
    parser, tree, root = parse_bytes(lang_name, data)
    query = Query(lang, query_src)
    # You can do query.matches(root) if you want match granularity
    captures = query.captures(root)
    out: list[dict[str, Any]] = []
    for node, cap_name in captures:
        entry = {"capture": cap_name, **_node_span(node)}
        out.append(entry)
    return out
```

**Why:** This yields a stable, language-agnostic capture format with both byte and “point” coordinates. Downstream tools (outline, AST, refs) can rely on uniform fields.

### 0.3 Organize language queries & stubs

You already have `queries/python.scm`. Add minimalist stubs for TOML/YAML/Markdown to enable basic outlines:

* `queries/toml.scm` — top-level keys
* `queries/yaml.scm` — first-level mappings/anchors
* `queries/markdown.scm` — headings & fenced blocks

Even if you don’t fill them now, the loader won’t explode and tools can advertise multiple languages.

---

# 1) Harden the MCP tools

You’ve got good building blocks in `mcp_server/tools.py`. Let’s make path sandboxing, caps, and outlines fully deterministic.

### 1.1 Rock-solid path sandbox

Replace any ad-hoc resolvers with this (tight and explicit):

```python
# codeintel/mcp_server/tools.py (top)
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Any
import os

REPO_ROOT = Path(os.environ.get("KGF_REPO_ROOT", Path.cwd())).resolve()

class SandboxError(ValueError): ...

def _resolve_path(rel: str) -> Path:
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT)):
        raise SandboxError(f"Path outside repository: {rel}")
    if p.is_dir():
        return p
    if not p.exists():
        raise FileNotFoundError(rel)
    return p

def _resolve_directory(rel: str | None) -> Path:
    d = REPO_ROOT if not rel else _resolve_path(rel)
    if not d.is_dir():
        raise SandboxError(f"Not a directory: {rel}")
    return d
```

**Why:** Prevents `../` traversal and symlink escapes. All tools should call this before touching files.

### 1.2 Caps & defaults (central constants)

```python
MAX_AST_BYTES = int(os.environ.get("CODEINTEL_MAX_AST_BYTES", "1048576")) # 1 MiB
LIMIT_DEFAULT = 100
LIMIT_MAX = 1000

def _bounded_limit(n: int | None) -> int:
    if n is None: return LIMIT_DEFAULT
    return max(1, min(int(n), LIMIT_MAX))
```

### 1.3 Listing files with filters

```python
from fnmatch import fnmatch

EXCLUDES = [
    "**/.git/**", "**/.venv/**", "**/_build/**", "**/__pycache__/**",
    "**/.mypy_cache/**", "**/.pytest_cache/**", "**/node_modules/**",
]

def list_files(directory: str | None = None, glob: str | None = None, limit: int | None = None) -> list[str]:
    root = _resolve_directory(directory)
    cap = _bounded_limit(limit)
    out: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file(): continue
        rel = str(p.relative_to(REPO_ROOT))
        if any(fnmatch(rel, pat.replace("**/", "")) for pat in EXCLUDES):
            continue
        if glob and not fnmatch(rel, glob):  # optional user filter
            continue
        out.append(rel)
        if len(out) >= cap: break
    return out
```

### 1.4 File read (chunked) and AST snapshot (with cap)

```python
def get_file(path: str, offset: int = 0, length: int | None = None) -> dict[str, Any]:
    p = _resolve_path(path)
    buf = p.read_bytes()
    if offset < 0 or offset > len(buf):
        raise ValueError("invalid offset")
    end = len(buf) if length is None else min(len(buf), offset + max(0, length))
    return {"path": path, "size": len(buf), "offset": offset, "data": buf[offset:end].decode("utf-8", errors="replace")}

def get_ast(path: str, language: str, format: str = "json") -> dict[str, Any]:
    p = _resolve_path(path)
    data = p.read_bytes()
    if len(data) > MAX_AST_BYTES:
        raise ValueError(f"file too large: {len(data)} bytes (limit {MAX_AST_BYTES})")
    from codeintel.indexer.tscore import parse_bytes
    _, tree, root = parse_bytes(language, data)
    if format == "sexpr":
        return {"path": path, "format": "sexpr", "ast": root.sexp()}
    # Shallow JSON walk (node type + span only) to keep size bounded
    def walk(node, depth=0, budget=200000):
        if budget <= 0: return None, 0
        obj = {"type": node.type, "span": {
            "start": {"row": node.start_point[0]+1, "col": node.start_point[1]+1},
            "end":   {"row": node.end_point[0]+1,   "col": node.end_point[1]+1},
        }}
        children = []
        remaining = budget - 1
        for ch in node.children:
            child_obj, rem_used = walk(ch, depth+1, remaining)
            if child_obj is not None:
                children.append(child_obj)
                remaining -= rem_used
            if remaining <= 0: break
        if children:
            obj["children"] = children
        used = budget - remaining
        return obj, used
    ast_obj, _ = walk(root)
    return {"path": path, "format": "json", "ast": ast_obj}
```

**Why:** AST output is bounded — no memory explosions on large files.

### 1.5 Outline and definitions using your captures

You’re already extracting `def.name`, `def.params`, etc. Build a simple outline:

```python
def make_outline(captures: list[dict[str, Any]], file_path: str) -> dict[str, Any]:
    # Group by match_id (your grouping helper already does this)
    from collections import defaultdict
    grouped = defaultdict(lambda: {"defs": [], "calls": []})
    for cap in captures:
        mid = cap.get("match_id", -1)
        grouped[mid]["all"] = grouped[mid].get("all", []) + [cap]
        if cap["capture"].startswith("def."):
            grouped[mid]["defs"].append(cap)
        elif cap["capture"].startswith("call."):
            grouped[mid]["calls"].append(cap)

    items: list[dict[str, Any]] = []
    for mid, g in grouped.items():
        names = [c for c in g["defs"] if c["capture"] == "def.name"]
        if not names: continue
        name = names[0].get("text", "")
        node = next((c for c in g["defs"] if c["capture"] == "def.node"), names[0])
        items.append({
            "name": name,
            "kind": "function",         # or infer from node.type
            "span": {
                "start": node["start_point"], "end": node["end_point"]
            }
        })
    return {"path": file_path, "items": items}
```

**Tool wrapper:**

```python
def get_outline(path: str, language: str = "python") -> dict[str, Any]:
    p = _resolve_path(path)
    data = p.read_bytes()
    from codeintel.indexer.tscore import run_query
    from pathlib import Path
    q = (Path(__file__).resolve().parents[1]/"queries"/f"{language}.scm").read_text()
    caps = run_query(language, data, q)
    return make_outline(caps, str(p.relative_to(REPO_ROOT)))
```

---

# 2) MCP stdio server — finish the JSON-RPC loop & register tools

You already have a solid `MCPServer` with:

* method: `"tools/list"` returns schemas built via Pydantic (`QueryRequest`, `SymbolsRequest`, `CallsRequest`, `ErrorsRequest`)
* method: `"tools/call"` dispatches to `_tool_handlers` mapping with `{ "ts.query", "ts.symbols", "ts.calls", "ts.errors" }`

To **finish**:

### 2.1 Add the new “file” and “outline/ast” tools and wire the handlers

Extend the handler map in `__init__`:

```python
self._tool_handlers = {
    "ts.query": self._tool_ts_query,
    "ts.symbols": self._tool_ts_symbols,
    "ts.calls": self._tool_ts_calls,
    "ts.errors": self._tool_ts_errors,
    "code.listFiles": self._tool_list_files,
    "code.getFile": self._tool_get_file,
    "code.getOutline": self._tool_get_outline,
    "code.getAST": self._tool_get_ast,
}
```

Add simple request models:

```python
class ListFilesRequest(BaseModel):
    directory: str | None = Field(None)
    glob: str | None = Field(None)
    limit: int | None = Field(None)

class GetFileRequest(BaseModel):
    path: str
    offset: int = 0
    length: int | None = None

class OutlineRequest(BaseModel):
    path: str
    language: str = "python"

class ASTRequest(BaseModel):
    path: str
    language: str = "python"
    format: str = "json"
```

Implement handlers using your `tools.py` helpers **inside** `server.py` (they already import `codeintel.mcp_server.tools as tools`):

```python
@staticmethod
async def _tool_list_files(payload: dict[str, Any]) -> dict[str, Any]:
    req = ListFilesRequest.model_validate(payload)
    items = await to_thread.run_sync(
        tools.list_files, req.directory, req.glob, req.limit
    )
    return {"status": "ok", "files": items, "meta": {"count": len(items)}}

@staticmethod
async def _tool_get_file(payload: dict[str, Any]) -> dict[str, Any]:
    req = GetFileRequest.model_validate(payload)
    out = await to_thread.run_sync(tools.get_file, req.path, req.offset, req.length)
    return {"status": "ok", **out}

@staticmethod
async def _tool_get_outline(payload: dict[str, Any]) -> dict[str, Any]:
    req = OutlineRequest.model_validate(payload)
    out = await to_thread.run_sync(tools.get_outline, req.path, req.language)
    return {"status": "ok", **out}

@staticmethod
async def _tool_get_ast(payload: dict[str, Any]) -> dict[str, Any]:
    req = ASTRequest.model_validate(payload)
    out = await to_thread.run_sync(tools.get_ast, req.path, req.language, req.format)
    return {"status": "ok", **out}
```

Add their schemas to ` _tool_schemas()`:

```python
{
  "name": "code.listFiles",
  "description": "List repo files with optional filters.",
  "inputSchema": ListFilesRequest.model_json_schema(),
},
{
  "name": "code.getFile",
  "description": "Read a file segment (UTF-8).",
  "inputSchema": GetFileRequest.model_json_schema(),
},
{
  "name": "code.getOutline",
  "description": "Return an outline (functions/classes) for a file.",
  "inputSchema": OutlineRequest.model_json_schema(),
},
{
  "name": "code.getAST",
  "description": "Return a bounded AST snapshot.",
  "inputSchema": ASTRequest.model_json_schema(),
},
```

### 2.2 Standardize error returns to Problem Details

Right now you do:

```python
except (FileNotFoundError, ValueError) as exc:
    await self._send_error(request_id, code=-32001, message=str(exc))
```

That’s fine for JSON-RPC, but for MCP/agent consumers it’s nicer to return a **Problem Details** object. Create a helper:

```python
def _pd(code: str, title: str, detail: str, status: int = 400) -> dict[str, Any]:
    return {"type": f"urn:kgf:problem:codeintel:{code.lower()}",
            "title": title, "detail": detail, "status": status, "code": code}

# example usage inside _handle_request
except tools.SandboxError as exc:
    await self._send_error(request_id, code=-32001, message=json.dumps(_pd("KGF-CI-1001", "Sandbox violation", str(exc), 403)))
    return
```

(If you prefer, keep JSON-RPC errors as is, but make the **tool result** itself wrap PD when a tool fails. Either is acceptable; the key is the stable PD shape.)

### 2.3 Concurrency and cancellation

You’re already offloading CPU-bound ops with `anyio.to_thread.run_sync`. Add cancellation support:

* If `anyio.get_cancelled_exc_class()` propagates, catch it and return PD with `KGF-CANCELLED`.
* In long scans (e.g., listing files), periodically check `anyio.cancel_scope().cancel_called` if you use a task group.

### 2.4 Manual test (now)

From the repo root:

```bash
python -m codeintel.mcp_server.server &
SERVER_PID=$!

# tools/list
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n' | nc -N localhost 0 2>/dev/null || true

# tools/call: code.listFiles
printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"code.listFiles","arguments":{"directory":".","glob":"src/**/*.py","limit":5}}}\n' | \
python -m codeintel.mcp_server.server

kill $SERVER_PID
```

(Or just pipe a single line into `python -m codeintel.mcp_server.server` as you did during development.)

---

## Why these changes matter

* **Determinism & safety:** repo-root sandbox + size caps prevent nasty surprises and keep responses bounded.
* **Reusability:** manifest-based grammar loading + cached `get_language()` gives you stable, reproducible behavior across tools and tests.
* **Agent-readiness:** `tools/list` exposes input schemas; result envelopes are predictable; PD errors are machine-friendly.
* **Performance:** on-demand parsing + small JSON AST keep the server snappy without prebuilding an index (you can add the persistent index later).

If you want, I can also sketch the exact patch hunks for `tools.py` and `server.py` based on the snippets above so you can drop them straight in.
