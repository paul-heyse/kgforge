This zip helps a ton—I opened it and you’ve already got ~80% of a workable stack:

* `codeintel/mcp_server/`: a stdio server skeleton (`server.py`), tool implementations (`tools.py`), and an optional FastAPI bridge (`http_bridge.py`).
* `codeintel/indexer/`: Tree-sitter core (`tscore.py`), Typer CLI (`cli.py`), and a shared CLI context (`cli_context.py`).
* `codeintel/build_languages.py` + `codeintel/build/languages.json`: manifest-driven language loading (python/json/yaml/toml/markdown).
* `codeintel/queries/python.scm`: first queries for Python (defs + calls).
* `codeintel/ts_languages/tree-sitter-python`: grammar submodule pinned alongside your manifest approach.

Here’s a customized, end-to-end plan to get this to a **finished, production-quality MCP delivery**. It plugs into your standardized CLI façade, enforces safety, and makes the server easy for ChatGPT to use.

---

# 0) Finish the language runtime

**What you have**

* A manifest (`build/languages.json`) that lists language modules from PyPI (e.g., `tree_sitter_python`) and a builder (`build_languages.py`) to regenerate it.

**Action**

1. **Lock the manifest**: make `build_languages.py` your single source of truth; add a Makefile target:

   ```makefile
   .PHONY: codeintel-langs
   codeintel-langs:
   	python -m codeintel.build_languages
   ```
2. **Ensure `tscore` uses the manifest**:

   * In `tscore.py`, implement/confirm:

     * `load_langs()` reads the manifest and imports `language()` from each module.
     * `get_language(name)` returns the cached `Language`.
     * `parse_bytes(lang, data)` creates a per-lang `Parser`, parses, and returns `(tree, root)`.
     * `run_query(lang, data, query_scm)` compiles the `.scm` and returns captures with **byte/point spans** and **text slices**.
3. **Queries per language**:

   * You already have `queries/python.scm` (defs + calls). Add at least:

     * `imports.scm` (for import graphs),
     * `class.scm` (classes/inheritance),
     * `attr.scm` (attribute accesses),
     * basic `errors.scm` if needed.
   * Add pass-through stubs for JSON/TOML/YAML/Markdown (even if they only expose a file outline) so tools can advertise multi-language support without failing.

---

# 1) Harden the MCP tools (`codeintel/mcp_server/tools.py`)

**What you have**

* A good set of utilities (path resolution, `run_ts_query`, error listing, capture grouping).

**Action (concrete behaviors)**

1. **Path sandboxing (non-negotiable)**:

   * Resolve `Path(repo_root) / rel_path`, then `resolve()`. Reject if it does **not** start with `repo_root.resolve()`.
   * Deny `..` segments, symlink escapes, and hidden/system dirs (add an allowlist later if you need to).
2. **Size & complexity caps**:

   * For `getAST`/raw AST dumps: refuse files > 1MB (configurable), and early-terminate traversal after N nodes (e.g., 200k).
   * For all tools: `limit` defaults (100) and a hard max (1000).
3. **Tool list & shapes** (JSON envelopes, consistent with your CLI contracts):

   * `code.health()` → `{ files_indexed, langs_loaded, queries_available }`.
   * `code.listFiles(directory?, glob?, lang?, limit?)` → repo-relative paths.
   * `code.getFile(path, offset?, length?)` → chunked text reads (UTF-8).
   * `code.getOutline(path)` → hierarchical outline (classes/functions + spans).
   * `code.getAST(path, format="json"|"sexpr")` → capped snapshot; return **Problem Details** if over limit.
   * `code.runTSQuery(path, language, query)` → captures with spans & text.
   * `code.listErrors(path, language)` → syntax error captures.
   * (Optional) `code.searchSymbols(query, kind?, lang?, limit?)` & `code.findReferences(qualname, limit?)` if you add a persistent index (Phase 2.5 below).
4. **Problem Details on every failure**:

   * Wrap `FileNotFoundError`, `ValueError` (bad query), and sandbox violations with your standard PD shape + a `KGF-CI-****` code.
5. **Docstrings**:

   * Keep short Pydantic models in `tools.py` for tool IO; they’re easy to reuse in the HTTP bridge and ensure schemas stay stable.

---

# 2) MCP stdio server (`codeintel/mcp_server/server.py`)

**What you have**

* An AnyIO scaffold, Pydantic request models, and a placeholder JSON-RPC loop.

**Action**

1. **Wire a minimal JSON-RPC 2.0 loop** (stdio):

   * Read newline-delimited JSON objects from `stdin`.
   * Dispatch by method name to `tools.*` functions (map `method → handler`).
   * Validate params with your Pydantic request models; on errors, return PD.
   * Always return `{"jsonrpc": "2.0", "id": <id>, "result": {...}}` or an error with PD.
2. **Concurrency & cancellation**:

   * Use `anyio.to_thread.run_sync` for filesystem/Tree-sitter work (non-async).
   * Support client cancellation by catching `CancelledError` and returning `KGF-CANCELLED`.
3. **Correlation & logs**:

   * Add a tiny request context (id, method, started_ms) and log `mcp.request`/`mcp.response` with duration and status.
   * Propagate a correlation-ID header when you later call HTTP (if your tools ever fetch remote content).
4. **Entry points**:

   * `python -m codeintel.mcp_server.server` should start stdio mode by default.
   * Keep `http_bridge.py` as optional (for Actions); MCP client should prefer stdio.

**Test it manually**:

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"code.health","params":{}}\n' | \
python -m codeintel.mcp_server.server
```

---

# 3) Tie it into your standardized CLI

Add a Typer entry that launches the server, using your façade:

```python
# codeintel/cli.py
import typer
from tools._shared.cli_integration import cli_operation
from tools._shared.cli_runtime import CliContext, EnvelopeBuilder
from codeintel.mcp_server.server import amain as mcp_amain
import anyio

app = typer.Typer()
mcp = typer.Typer()
app.add_typer(mcp, name="mcp")

@mcp.command("serve")
@cli_operation(echo_args=True, echo_env=True)
def serve(ctx: CliContext, env: EnvelopeBuilder, repo: str = "."):
    # Optionally export repo root to tools via env/config if needed
    anyio.run(mcp_amain)
    env.set_result(summary="MCP server exited cleanly")
```

Makefile:

```makefile
.PHONY: codeintel-serve
codeintel-serve:
\tpython -m codeintel.cli mcp serve --repo .
```

---

# 4) Optional: persistent index (symbols/refs)

You can ship without this (MCP tools work directly on files), but if you want fast cross-ref:

1. Add `codeintel/index/` with a tiny SQLite schema (`files`, `symbols`, `refs`) and an incremental updater (mtime/size guard).
2. Implement `searchSymbols` and `findReferences` against that store.
3. Hook an optional watchdog to reindex changed files in the background while the MCP server is running.

(If you want, I can spec this part in a follow-up, but it’s not required for a first MCP delivery.)

---

# 5) Security & resource limits (ship blockers)

* **Path sandbox** (see §1.1) on every tool and resource.
* **Size/time caps**: AST/outline and large file reads must respect limits.
* **Advanced query guard**: only allow raw S-expressions when `CODEINTEL_ENABLE_TS_QUERY=1`.
* **Rate limiting** (simple): 5–10 tool calls/sec with a token bucket in memory to prevent self-DDOS.

---

# 6) Tests (minimum bar)

* Unit:

  * `run_ts_query` returns captures on a small Python fixture; error query returns PD.
  * Path sandbox rejects `../` and symlink escapes.
  * Size caps: `getAST` over limit returns PD.
* Integration:

  * Start `server.py` in a subprocess and round-trip JSON-RPC for `code.health`, `code.listFiles`, `code.getOutline`.
* CLI:

  * `codeintel.cli mcp serve` starts and responds.

---

# 7) Docs (agent-ready)

* `docs/modules/codeintel/`:

  * **MCP tools table** (name, params, result schema, limits, examples).
  * **Resources** (if you add them later): e.g., `kgf://file/<path>`, with chunking rules.
  * **Config**: env vars (`CODEINTEL_EXCLUDES`, `CODEINTEL_MAX_AST_BYTES`, etc.).
* Add a one-pager “How to connect ChatGPT to KGF CodeIntel (MCP)” (see §9).

---

# 8) CI guardrails

* Lint/typecheck on `codeintel/**`.
* Unit/integration tests from §6.
* “No direct Tree-sitter import drift”: simple grep to ensure all language loads pass via `tscore` + manifest.
* (Optional) deny direct `open()` on arbitrary paths under `mcp_server`, enforcing the sandbox helper.

---

# 9) Connect ChatGPT (MCP) to your server

1. Ensure the repo’s venv has dependencies:

   ```
   pip install tree_sitter anyio pydantic fastapi uvicorn typer
   # plus your internal kgfoundry_common/tools packages
   ```
2. In ChatGPT, add a **local MCP server**:

   * **Command**: `python -m codeintel.mcp_server.server`
   * **Working directory**: your repo root (so relative paths resolve).
   * (Optional) Env:

     * `CODEINTEL_EXCLUDES="**/.venv/**,**/_build/**,**/.git/**"`
     * `CODEINTEL_MAX_AST_BYTES="1048576"`
3. From ChatGPT, call:

   * `code.health` → should return language/queries status.
   * `code.listFiles { directory: "." }`
   * `code.getOutline { path: "src/.../module.py" }`
   * `code.runTSQuery { path: "...", language: "python", query: "(function_definition name: (identifier) @name)" }`

---

## What’s already “done” vs “to finish”

**Done (from your zip)**

* Manifest-driven language config and a Python grammar vendored.
* Core TS utilities scaffold (`tscore.py`).
* Tooling scaffolds and FastAPI bridge.
* Indexer CLI and CLI context aligned with your shared contracts.

**To finish (this plan)**

* Tighten path sandbox, caps, and PD mapping in `tools.py`.
* Complete the stdio JSON-RPC loop in `server.py` and wire all tool handlers.
* Add Typer `codeintel mcp serve` entry using your façade.
* (Optional) Expand Python queries; add minimal stubs for TOML/YAML/Markdown.
* Tests, docs, and a Makefile flow.

---

If you want, I can draft PR-ready stubs for:

* the JSON-RPC loop in `server.py` (method map + AnyIO stdio),
* the path-sandbox & caps in `tools.py`,
* the Typer `mcp serve` command,
* and a tiny test that spins the server and exercises `code.health`.

That’ll let you drop it in and iterate quickly.
