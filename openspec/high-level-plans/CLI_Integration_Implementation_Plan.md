Yes—let’s finish this properly. Below is a **comprehensive, end-to-end rollout plan** that takes the façade you just added and drives it to a **best-in-class, repo-wide standard**. It’s broken into phases, with specific file targets in your repo, code you can paste, and CI guardrails so it stays consistent.

---

# Phase 1 — Ground truth & dependencies

## 1.1. Confirm new modules are in place

You’ve added:

* `tools/_shared/paths.py`
* `tools/_shared/problems.py`
* `tools/_shared/observability.py`
* `tools/_shared/cli_runtime.py`
* `tools/_shared/cli_integration.py`
* `schema/tools/cli_envelope.json` bumped to **v1.1.0**

## 1.2. (Optional) Add JSON-Schema validation dependency for dev

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "jsonschema>=4.22",
  "pytest",
  "typer[all]",
  # your existing dev deps ...
]
```

> The façade tolerates missing `jsonschema` (logs a warning). Enable it in dev/CI to catch drift.

---

# Phase 2 — Finalize façade knobs (signals, codes, metrics)

## 2.1. Add gentle SIGINT/SIGTERM handling (optional but worth it)

**Edit** `tools/_shared/cli_runtime.py`:

* Add this in the file (near imports):

```python
import signal
from types import FrameType
```

* Add an **optional** signal handler inside `cli_run`:

```python
cancelled = {"value": False}

def _on_cancel(sig: int, frame: FrameType | None):
    cancelled["value"] = True
    # We don’t raise here; we mark and let your code notice or let finalize map it.
    # If you prefer, raise KeyboardInterrupt here instead.
    logger.info("cli_run_signal", extra={"signal": sig})

old_int = signal.getsignal(signal.SIGINT)
old_term = signal.getsignal(signal.SIGTERM)
signal.signal(signal.SIGINT, _on_cancel)
signal.signal(signal.SIGTERM, _on_cancel)
```

* In the `finally:` block (before writing envelope), if the run was otherwise “success” but `cancelled["value"] is True`, set a canonical problem:

```python
if status == "success" and cancelled["value"]:
    status = "error"
    env.set_problem(ProblemDetails(
        type="urn:kgf:problem:cancelled",
        title="Operation cancelled",
        status=499,
        code="KGF-CANCELLED",
    ))
```

* Restore signals at the very end:

```python
signal.signal(signal.SIGINT, old_int)
signal.signal(signal.SIGTERM, old_term)
```

## 2.2. Centralize error code mapping

Create a single map and **re-use it** so codes are consistent across all CLIs:

```python
# tools/_shared/error_codes_map.py
from .problems import ProblemDetails

class ConfigurationError(Exception): ...
class ExternalRateLimited(Exception): ...
# reuse your project-specific exceptions if you already have them

ERROR_CODE_MAP = {
    ConfigurationError: "KGF-1001",
    ExternalRateLimited: "KGF-1203",
    KeyboardInterrupt: "KGF-CANCELLED",
    # extend over time...
}
```

If you use `@cli_operation`, wire this map as the **default**: edit `tools/_shared/cli_integration.py`:

```python
from .error_codes_map import ERROR_CODE_MAP

def cli_operation(*, echo_args: bool = True, echo_env: bool = False, error_code_map = ERROR_CODE_MAP):
    ...
            cfg = CliRunConfig(
                command_path=route,
                args_summary=[f"{k}={v}" for k, v in kwargs.items()] if echo_args else None,
                env_summary=_collect_env_summary() if echo_env else None,
                error_code_map=error_code_map,
            )
```

## 2.3. Wire metrics emission to your stack

Provide a Prometheus emitter and inject it:

```python
# tools/_shared/metrics_prom.py
from prometheus_client import Summary, Counter
from .observability import MetricEmitter

_cli_run_sec = Summary("kgf_cli_run_duration_seconds", "CLI run duration", ["operation"])
_cli_run_total = Counter("kgf_cli_runs_total", "CLI runs total", ["operation", "status"])

class PromEmitter(MetricEmitter):
    def emit_cli_run(self, *, operation: str, status: str, duration_s: float) -> None:
        _cli_run_sec.labels(operation=operation).observe(duration_s)
        _cli_run_total.labels(operation=operation, status=status).inc()

emitter = PromEmitter()
```

Then in your **bootstrap** (e.g., `src/__init__.py` or a tiny `tools/bootstrap.py` you import from CLIs):

```python
from tools._shared import observability as _obs
from tools._shared.metrics_prom import emitter as prom_emitter
_obs.emitter = prom_emitter
```

> Keep label cardinality in check by **only** labeling by `operation` (dot-joined route).

---

# Phase 3 — Convert all CLIs to the façade

**Primary targets in this repo** (based on your codebase):

* `src/download/cli.py`
* `src/orchestration/cli.py`
* `codeintel/indexer/cli.py`
* `tools/typer_to_openapi_cli.py` (generator itself can be wrapped; the command runner is still a CLI)

## 3.1. Mechanical steps for each command

1. Import the decorator & types:

```python
from tools._shared.cli_integration import cli_operation
from tools._shared.cli_runtime import CliContext, EnvelopeBuilder
```

2. Decorate each Typer/Click command:

```python
@app.command()
@cli_operation(echo_args=True, echo_env=False)
def mycmd(ctx: CliContext, env: EnvelopeBuilder, arg1: str, flag: bool = False):
    ...
```

3. Replace bespoke try/except, path building, and “write envelope” logic with:

* `ctx.logger` for logs.
* `env.add_artifact(...)` for files produced.
* `env.set_result(...)` on success.
* Let `cli_run` do the rest.

4. If you had ad-hoc exit codes, keep them **only** for very specific semantics. The façade will exit(1) on error otherwise.

### Example rewrite (before → after)

**Before**

```python
@app.command()
def harvest(source: str, max_items: int = 200):
    logger.info("start", extra={"source": source})
    try:
        n = run_harvest(source, max_items)
        write_json("docs/_data/cli/download/harvest/....json", {...})
        print(f"Harvested {n} docs")
        raise SystemExit(0)
    except Exception as e:
        write_json(".../error.json", problem_details_for(e))
        raise SystemExit(1)
```

**After**

```python
@app.command()
@cli_operation()
def harvest(ctx: CliContext, env: EnvelopeBuilder, source: str, max_items: int = 200):
    ctx.logger.info("begin", extra={"source": source, "max_items": max_items})
    n = run_harvest(source, max_items)
    env.set_result(summary=f"Harvested {n} docs from {source}")
```

> The façade gives you: start/done logs, duration, stdout, JSON envelope with schema, metrics, exit code.

## 3.2. Codemod (optional accelerator)

Create `tools/cli/codemods/decorate_cli_commands.py` to add the decorator and prepend `(ctx, env, ...)` to function signatures. Use LibCST if you want safe rewrites; otherwise do it by hand—there are only a handful of CLIs.

---

# Phase 4 — OpenAPI & docs alignment

## 4.1. Update the Typer→OpenAPI generator

**Target:** `tools/typer_to_openapi_cli.py`

* Ensure every operation gets:

  * `operationId = ".".join(route)` (the façade’s `operation`)
  * `x-cli` extension with the `command_path` list
* Tag grouping: group by the first 1–2 segments (`download`, `download.harvest`) so the docs side nav remains readable.

Pseudo-snippet:

```python
def operation_meta_from_typer(cmd) -> dict:
    route = compute_route_for_command(cmd)  # same logic as current_route()
    operation = ".".join(route)
    return {
        "operationId": operation,
        "tags": [route[0], ".".join(route[:2]) if len(route) > 1 else route[0]],
        "x-cli": {"command_path": route, "operation": operation},
        "summary": cmd.help or "No summary",
        # parameters, requestBody, responses, etc...
    }
```

## 4.2. MkDocs plumbing

* Add a docs page “CLI Runs” that:

  * Lists available operations (by reading `docs/_data/cli/**`).
  * For each operation, shows the **latest envelope** and links to older ones.
* Optional: embed a collapsible JSON viewer of the latest envelope.

Simple loader (pseudo):

```python
# tools/mkdocs_suite/docs/_scripts/list_cli_runs.py
from pathlib import Path
import json

def latest_envelopes(root: Path):
    for op_dir in sorted(root.rglob("*")):
        if not op_dir.is_dir(): continue
        envs = sorted(op_dir.glob("*.json"))
        if not envs: continue
        latest = envs[-1]
        yield op_dir, json.loads(latest.read_text())
```

---

# Phase 5 — CI: smoke tests & drift guards

## 5.1. Pytest smoke for façade

Add `tests/test_cli_runtime.py` (sample from previous message). Also add **per-CLI** smoke using `typer.testing.CliRunner`:

```python
# tests/test_cli_commands.py
from typer.testing import CliRunner
from src.download.cli import app as download_app

runner = CliRunner()

def test_download_harvest_smoke(tmp_path, monkeypatch):
    from tools._shared.paths import Paths
    monkeypatch.setattr(Paths, "discover", staticmethod(
        lambda: Paths(repo_root=tmp_path, docs_data=tmp_path/"docs/_data", cli_out_root=tmp_path/"docs/_data/cli")
    ))
    result = runner.invoke(download_app, ["harvest", "--source", "openalex", "--max-items", "1"])
    assert result.exit_code == 0
    # Assert envelope written:
    assert list((tmp_path/"docs/_data/cli/download/harvest").glob("*.json"))
```

## 5.2. Envelope validator for recent runs (optional)

**Script:** `tools/cli/validate_envelopes.py`

* Walk `docs/_data/cli/**` for files modified in the last 2–3 days and validate against `schema/tools/cli_envelope.json`.
* CI step: run it; fail on validation error.

## 5.3. Adoption scanner (don’t regress)

**Script:** `tools/cli/check_cli_adoption.py`

* Parse AST to find Typer/Click command functions.
* Ensure each has the `@cli_operation` decorator (or a suppression comment `# no-facade-ok`).
* CI step: run and fail if any command is not wrapped.

Quick scaffold:

```python
# tools/cli/check_cli_adoption.py
import ast, sys, pathlib, re

DECORATOR = "cli_operation"

def file_has_raw_commands(path: pathlib.Path) -> list[tuple[str, int]]:
    src = path.read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            deco_names = {getattr(getattr(d, "func", d), "id", None) or getattr(d, "id", None) for d in node.decorator_list}
            if "command" in "".join(re.findall(r"@.*command", src)) and DECORATOR not in deco_names:
                offenders.append((path.as_posix(), node.lineno))
    return offenders

def main():
    root = pathlib.Path.cwd()
    offenders = []
    for path in root.rglob("*.py"):
        if any(p in path.as_posix() for p in ("/tests/", "/_shared/", "/.venv/")):
            continue
        offenders.extend(file_has_raw_commands(path))
    if offenders:
        print("Missing @cli_operation:", *offenders, sep="\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Hook both scripts in **CI** (GitHub Actions) after your unit tests.

---

# Phase 6 — Import-linter & layering guardrails

Update `.importlinter` (conceptually):

* Forbid `src/**` from importing `tools/**` (keep `tools/_shared` usable by CLIs and tooling, but not by core libraries—adjust to your architecture).
* Allow CLIs to import from `tools/_shared` and their own module packages, but not vice versa.
* Create a specific contract: all `cli.py` files **must not** import random internal subpackages that violate layering (e.g., `embeddings_http` importing service internals).

*(If you already have import-linter in place, add these as separate, named contracts so violations are visible.)*

---

# Phase 7 — Developer UX polish

## 7.1. Makefile targets

```makefile
.PHONY: cli-smoke cli-validate cli-adoption

cli-smoke:
\tpytest -q tests/test_cli_* -k smoke

cli-validate:
\tpython -m tools.cli.validate_envelopes

cli-adoption:
\tpython -m tools.cli.check_cli_adoption
```

## 7.2. “How to write a CLI” doc (short and crisp)

Create `docs/contrib/howto_cli.md`:

* Always use `@cli_operation`.
* First params must be `(ctx, env, ...)`.
* Use `ctx.logger` for logs, `env.add_artifact` for outputs, `env.set_result` for final payloads.
* Never write envelopes manually.
* Avoid unique metric labels; rely on `operation`.

---

# Phase 8 — Back-compat & deprecation plan

* **Envelope v1.1** includes `command` and `subcommand` as **nullable** (for consumers that still expect them).
* If any docs/tools assumed the **two-level** directory (`download/harvest/`), add short-term symlinks or update the readers to handle **multi-level** routes.
* Announce a 1–2 release deprecation window; then remove the symlinks and stop populating `command`/`subcommand`.

---

# Phase 9 — Security & safety review

* Confirm `normalize_token` forbids traversal and odd characters (already enforced).
* Keep `env_summary` **small and redacted**; do not include secrets.
* Keep `args_summary` **non-sensitive**; if any subcommand takes tokens/passwords, scrub them before passing to `CliRunConfig`.
* Ensure any exceptions you intentionally bubble (e.g., KeyboardInterrupt) get a sensible error code and title via `ERROR_CODE_MAP`.

---

# Phase 10 — Rollout checklist (copy/paste for tracking)

* [ ] Files added: `_shared` façade, schema v1.1, integration decorator.
* [ ] Prometheus emitter injected (or confirm noop is acceptable).
* [ ] **Convert CLIs**:

  * [ ] `src/download/cli.py` uses `@cli_operation`
  * [ ] `src/orchestration/cli.py` uses `@cli_operation`
  * [ ] `codeintel/indexer/cli.py` uses `@cli_operation`
  * [ ] `tools/typer_to_openapi_cli.py` uses `@cli_operation`
* [ ] Remove bespoke envelope-writing / try/except from each converted CLI.
* [ ] Typer→OpenAPI updated to emit `operation` & `x-cli.command_path`.
* [ ] MkDocs page “CLI Runs” lists latest envelopes by route.
* [ ] Tests:

  * [ ] façade unit tests pass (`tests/test_cli_runtime.py`)
  * [ ] per-CLI smoke tests pass (`typer.testing.CliRunner`)
* [ ] CI:

  * [ ] `cli-validate` validates recent envelopes against schema
  * [ ] `cli-adoption` ensures all commands are decorated
* [ ] Import-linter: contracts for CLI layering added/passing.
* [ ] Back-compat: any old readers updated (or symlinks created); announce deprecation.
* [ ] Docs: `docs/contrib/howto_cli.md` added.

---

## Final notes & quality bar

* **Determinism**: file names (`YYYYMMDD-HHMMSS-runid`) + route-mirrored directories guarantees easy diffing and historical trails.
* **Low duplication**: all CLIs share the same lifecycle (logs, metrics, JSON envelope, stdout).
* **Observability**: single low-cardinality label (`operation`) keeps metrics safe; add rollups by prefix in dashboard queries rather than labels.
* **Agent-readiness**: schema-validated, predictable envelopes and `x-cli` OpenAPI links make this trivial for AI agents to consume.

If you want, I can draft PR-ready diffs for `src/download/cli.py` and `tools/typer_to_openapi_cli.py` showing the precise changes (imports, decorators, signature adjustments, and removal of bespoke boilerplate).
