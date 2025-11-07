Love it — let’s finish strong. Below is a **deep, repo-tailored plan for Steps 6 → 9** that plugs directly into what you’ve already built and the earlier end-to-end outline you referenced. 

---

# 6) Tests (unit, integration, CLI smoke)

## 6.1 Test layout

```
tests/
  codeintel/
    conftest.py
    test_tools_sandbox.py
    test_tools_ast_limits.py
    test_tools_outline.py
    test_server_roundtrip.py
    test_rate_limit_timeout.py
    test_cli_entrypoints.py
```

### `tests/codeintel/conftest.py`

* Provide `REPO_FIXTURE` (a tiny synthetic repo: a few `.py`, `.toml`, `.md` files).
* Set env for sandbox & limits (`KGF_REPO_ROOT`, `CODEINTEL_MAX_AST_BYTES`, `CODEINTEL_LIMIT_MAX`, `CODEINTEL_ENABLE_TS_QUERY`).
* Optionally patch `REPO_ROOT` in `codeintel.mcp_server.tools` if it resolves at import (prefer reading from env to avoid this).

````python
# tests/codeintel/conftest.py
import os, shutil
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def repo_fixture(tmp_path_factory):
    r = tmp_path_factory.mktemp("repo")
    (r/"pkg").mkdir()
    (r/"pkg"/"mod.py").write_text("class A:\n    def f(self,x):\n        return x\n\n")
    (r/"README.md").write_text("# sample\n\n```python\nprint('hi')\n```\n")
    (r/"pyproject.toml").write_text('[tool]\nname="demo"\n')
    return r

@pytest.fixture(autouse=True)
def set_env(repo_fixture, monkeypatch):
    monkeypatch.setenv("KGF_REPO_ROOT", str(repo_fixture))
    monkeypatch.setenv("CODEINTEL_MAX_AST_BYTES", "65536")
    monkeypatch.setenv("CODEINTEL_LIMIT_MAX", "1000")
    monkeypatch.setenv("CODEINTEL_ENABLE_TS_QUERY", "1")  # enable advanced query in tests
````

## 6.2 Unit tests (tools)

### Sandbox

```python
# tests/codeintel/test_tools_sandbox.py
from codeintel.mcp_server import tools
import pytest
from pathlib import Path

def test_resolve_path_inside(repo_fixture):
    p = tools._resolve_path("pkg/mod.py")
    assert p.exists()

def test_resolve_path_outside_raises(tmp_path):
    outside = Path("/").resolve()  # definitely outside
    with pytest.raises(tools.SandboxError):
        tools._resolve_path("../../etc/passwd")
```

### AST size cap

```python
# tests/codeintel/test_tools_ast_limits.py
from codeintel.mcp_server import tools
import pytest

def test_get_ast_respects_size_limit(monkeypatch, repo_fixture):
    bigfile = repo_fixture/"pkg"/"big.py"
    bigfile.write_text("x='a'*" + "1"*100000)  # large literal; crude but works
    monkeypatch.setenv("CODEINTEL_MAX_AST_BYTES", "64")
    with pytest.raises(ValueError):
        tools.get_ast("pkg/big.py", "python", "json")
```

### Outline (Python)

```python
# tests/codeintel/test_tools_outline.py
from codeintel.mcp_server import tools

def test_outline_simple(repo_fixture):
    out = tools.get_outline("pkg/mod.py", "python")
    assert out["path"].endswith("pkg/mod.py")
    assert isinstance(out["items"], list)
    assert any(i["name"]=="f" for i in out["items"])
```

## 6.3 Integration tests (JSON-RPC round-trip)

Spin up the stdio server in-process via `anyio.run()` where possible, or use `subprocess` for true stdio behavior.

```python
# tests/codeintel/test_server_roundtrip.py
import json, subprocess, sys, time
from pathlib import Path

def _rpc(proc, payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "no response"
    return json.loads(line)

def test_tools_list(repo_fixture):
    proc = subprocess.Popen(
        [sys.executable, "-m", "codeintel.mcp_server.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(repo_fixture)
    )
    try:
        msg = {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
        res = _rpc(proc, msg)
        assert res["result"]["tools"]  # non-empty
    finally:
        proc.kill()
```

Add a second test that calls `tools/call`:

```python
def test_get_outline_call(repo_fixture):
    proc = subprocess.Popen([...], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=str(repo_fixture))
    try:
        # tools/list (to discover schema) — optional
        _ = _rpc(proc, {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}})
        # tools/call
        call = {
          "jsonrpc":"2.0","id":2,"method":"tools/call",
          "params":{"name":"code.getOutline","arguments":{"path":"pkg/mod.py","language":"python"}}
        }
        res = _rpc(proc, call)
        assert res["result"]["status"] == "ok"
        assert any(i["name"]=="f" for i in res["result"]["items"])
    finally:
        proc.kill()
```

## 6.4 Rate-limit & timeout behavior

Set tiny limits through env and hammer the server:

```python
# tests/codeintel/test_rate_limit_timeout.py
import json, subprocess, sys

def test_rate_limit(repo_fixture, monkeypatch):
    monkeypatch.setenv("CODEINTEL_RATE_LIMIT_QPS", "1")
    monkeypatch.setenv("CODEINTEL_RATE_LIMIT_BURST", "1")
    p = subprocess.Popen([sys.executable, "-m", "codeintel.mcp_server.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=str(repo_fixture))
    try:
        def rpc(id_): 
            req={"jsonrpc":"2.0","id":id_,"method":"tools/call","params":{"name":"code.listFiles","arguments":{"limit":1}}}
            p.stdin.write(json.dumps(req)+"\n"); p.stdin.flush(); return json.loads(p.stdout.readline())
        r1 = rpc(1); assert "result" in r1 or "error" in r1
        r2 = rpc(2); assert "error" in r2 and "rate-limit" in json.dumps(r2).lower()
    finally:
        p.kill()
```

## 6.5 CLI smoke

```python
# tests/codeintel/test_cli_entrypoints.py
from typer.testing import CliRunner
from codeintel.cli import app

def test_cli_serve_help():
    res = CliRunner().invoke(app, ["mcp", "serve", "--help"])
    assert res.exit_code == 0
```

---

# 7) Docs (agent-ready, generated, and drift-checked)

## 7.1 Author content pages

Create:

```
docs/modules/codeintel/
  index.md
  quickstart_mcp.md
  tools.md
  limits.md
  config.md
```

**`index.md` (example skeleton):**

```markdown
# CodeIntel (Tree-sitter + MCP)

CodeIntel exposes your repository structure to MCP clients (e.g., ChatGPT) via a local stdio server. It’s safe-by-default (sandboxed), fast, and agent-friendly.

- **Server:** `python -m codeintel.mcp_server.server`
- **CLI:** `python -m codeintel.cli mcp serve --repo .`
- **Tools:** outline, AST, TS query, file list/get (see [tools.md](./tools.md))
- **Limits:** size/time/rate caps (see [limits.md](./limits.md))
```

**`quickstart_mcp.md`:**

* How to run the server.
* How to add a “local MCP server” in ChatGPT (command + working directory).
* First calls to try (`tools/list`, `code.getOutline`, `ts.query`).

**`limits.md`:**

* Enumerate `CODEINTEL_MAX_AST_BYTES`, `CODEINTEL_LIMIT_MAX`, `CODEINTEL_TOOL_TIMEOUT_S`, `CODEINTEL_RATE_LIMIT_*`.
* Default values and how to override.

**`config.md`:**

* `KGF_REPO_ROOT`, excludes, advanced query flag (`CODEINTEL_ENABLE_TS_QUERY`).

## 7.2 Generate the tools reference from server schemas

Create a generator:

```python
# tools/mkdocs_suite/docs/_scripts/gen_codeintel_mcp_docs.py
from codeintel.mcp_server.server import MCPServer

HEADER = "# CodeIntel MCP Tools\n\nAuto-generated from server schemas.\n\n"
def main():
    s = MCPServer()
    tools = s._tool_schemas()
    lines = [HEADER]
    for t in tools:
        lines.append(f"## `{t['name']}`\n\n{t.get('description','')}\n")
        schema = t["inputSchema"]
        lines.append("**Parameters**:\n")
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        for k, v in props.items():
            typ = v.get("type", "any")
            desc = v.get("description", "")
            star = " *(required)*" if k in req else ""
            lines.append(f"- `{k}`: `{typ}`{star} — {desc}")
        lines.append("\n")
    (Path(__file__).parents[3]/"docs/modules/codeintel/tools.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    from pathlib import Path
    main()
```

Add an mkdocs hook or Makefile target:

```makefile
.PHONY: docs-codeintel
docs-codeintel:
\tpython tools/mkdocs_suite/docs/_scripts/gen_codeintel_mcp_docs.py
```

## 7.3 mkdocs.yml nav

```yaml
nav:
  - CodeIntel:
      - Overview: modules/codeintel/index.md
      - Quickstart (MCP): modules/codeintel/quickstart_mcp.md
      - Tools: modules/codeintel/tools.md
      - Limits: modules/codeintel/limits.md
      - Config: modules/codeintel/config.md
```

## 7.4 Drift checks

* CI runs `docs-codeintel` and **fails** if `tools.md` changed (means code drift → regenerate & commit).
* Optionally snapshot the JSON from `MCPServer._tool_schemas()` and fail on diffs to catch accidental schema breaking.

---

# 8) CI guardrails (lint, typecheck, tests, policy checks)

## 8.1 GitHub Actions (Python)

`.github/workflows/codeintel.yml`:

```yaml
name: codeintel

on:
  push:
    paths:
      - "codeintel/**"
      - "tools/_shared/**"
      - "docs/modules/codeintel/**"
      - ".github/workflows/codeintel.yml"
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: "pip"
      - name: Install deps
        run: |
          pip install -U pip
          pip install -e .[dev]  # include tree_sitter, anyio, pydantic, typer, pytest, ruff, mypy/pyright
      - name: Build TS manifest
        run: python -m codeintel.build_languages
      - name: Lint
        run: ruff check .
      - name: Typecheck
        run: |
          mypy codeintel || true
          # or: pyright
      - name: Unit & Integration tests
        env:
          CODEINTEL_ENABLE_TS_QUERY: "1"
        run: pytest -q tests/codeintel
      - name: Generate docs (tools)
        run: make docs-codeintel
      - name: Ensure docs drift-free
        run: |
          git diff --exit-code docs/modules/codeintel/tools.md
```

## 8.2 Pre-commit hooks

`.pre-commit-config.yaml` (excerpt):

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{ id: ruff, args: ["--fix"] }, { id: ruff-format }]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks: [{ id: mypy, additional_dependencies: ["pydantic"] }]
  - repo: local
    hooks:
      - id: forbid-direct-tree-sitter-imports
        name: Forbid direct tree-sitter imports outside indexer/tscore.py
        entry: bash -c "! grep -RIn --include='*.py' -E 'from tree_sitter|import tree_sitter' codeintel | grep -v 'indexer/tscore.py'"
        language: system
      - id: forbid-unsafe-open
        name: Forbid raw open() in mcp_server (use tools._resolve_path)
        entry: bash -c "! grep -RIn --include='*.py' -E '\\bopen\\(' codeintel/mcp_server || (echo 'Use sandboxed helpers' && exit 1)"
        language: system
```

**Why:** Prevent accidental bypass of your sandbox and ensure grammar usage stays centralized.

---

# 9) Connect ChatGPT (MCP) to your server

> UI labels can change, but the flow is consistent: add a **local MCP server** whose command is your stdio server, and set the working directory to your repo root.

## 9.1 Prepare the environment

* Ensure venv has the dependencies:

  ```
  pip install -e .[dev]
  python -m codeintel.build_languages
  ```

* Decide how you’ll run it:

  * direct: `python -m codeintel.mcp_server.server`
  * via CLI façade: `python -m codeintel.cli mcp serve --repo .`

* (Optional) set env caps:

  ```
  export CODEINTEL_MAX_AST_BYTES=1048576
  export CODEINTEL_RATE_LIMIT_QPS=5
  export CODEINTEL_RATE_LIMIT_BURST=10
  ```

## 9.2 Register in ChatGPT (local MCP)

* In ChatGPT, open **Settings** → **Capabilities / MCP** (or equivalent).
* Choose **Add local server**.
* **Command**: `python -m codeintel.mcp_server.server`
* **Working directory**: your repo root
* (Optional) add env vars for caps/excludes.

ChatGPT should handshake, call `tools/list`, and display your tools (`code.listFiles`, `code.getFile`, `code.getOutline`, `code.getAST`, `ts.query`, …).

## 9.3 First calls to try

* `code.listFiles { directory: ".", glob: "pkg/*.py", limit: 5 }`
* `code.getOutline { path: "pkg/mod.py", language: "python" }`
* `code.getAST { path: "pkg/mod.py", language: "python", format: "json" }`
* `ts.query { path: "pkg/mod.py", language: "python", query: "(function_definition name: (identifier) @def.name)" }`

## 9.4 Troubleshooting

* **No response**: ensure newline-terminated JSON on stdin (your server already reads line-delimited JSON).
* **Sandbox errors**: check working directory and `KGF_REPO_ROOT`; the server rejects paths outside the repo.
* **Large file PD**: increase `CODEINTEL_MAX_AST_BYTES`.
* **Rate limit PD**: raise `CODEINTEL_RATE_LIMIT_*`.
* **TS query disabled**: set `CODEINTEL_ENABLE_TS_QUERY=1`.

---

## Final acceptance checklist (6–9)

* [ ] Tests: unit (sandbox, AST cap, outline), integration (JSON-RPC), rate/timeout, CLI smoke.
* [ ] Docs: Index/Quickstart/Tools/Limits/Config, generator script, mkdocs nav.
* [ ] CI: lint, typecheck, run tests, generate docs, drift check, pre-commit guardrails.
* [ ] ChatGPT connection: local MCP server added; basic calls succeed.

This locks in a **best-in-class** delivery: reproducible grammars, safe & observable server, generated docs with drift checks, robust tests, and a turnkey MCP experience in ChatGPT. If you want, I can package these into PR-ready commits (tests, docs script, CI workflow, pre-commit hooks) so you can drop them straight into the repo.
