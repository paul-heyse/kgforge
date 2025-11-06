Love it. Here’s a **complete, repo-ready implementation plan** (with code you can paste) to deploy the **multi-level CLI façade** across kgfoundry. It’s organized so a junior dev can follow step-by-step and safely migrate every CLI.

---

# 0) What you’ll get

* A single façade that runs **any depth** of commands like `download.harvest.openalex.run`.
* Deterministic artifacts: **one JSON envelope per run** in `docs/_data/cli/<route...>/<timestamp>-<run>.json`.
* Zero duplication of logging, metrics, error mapping, JSON-Schema validation, atomic file writes, and stdout summaries.
* Easy Typer integration (`@cli_operation` decorator), and OpenAPI/Docs can read the new `operation` field.

---

# 1) Add the shared building blocks

Create these new files (or adapt if you already have similarly named modules).

## 1.1 `tools/_shared/paths.py`

```python
# tools/_shared/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    docs_data: Path
    cli_out_root: Path

    @staticmethod
    @lru_cache(maxsize=1)
    def discover(start: Path | None = None) -> "Paths":
        cur = (start or Path(__file__)).resolve()
        for p in [cur, *cur.parents]:
            if (p / ".git").exists():
                repo_root = p
                break
        else:
            repo_root = Path.cwd()

        docs_data = repo_root / "docs" / "_data"
        cli_out_root = docs_data / "cli"
        return Paths(repo_root=repo_root, docs_data=docs_data, cli_out_root=cli_out_root)

    def cli_envelope_dir(self, route: list[str]) -> Path:
        base = self.cli_out_root
        for seg in route:
            base = base / seg
        return base
```

## 1.2 `tools/_shared/problems.py`

```python
# tools/_shared/problems.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Mapping

@dataclass
class ProblemDetails:
    type: str                 # URI or urn:kgf:problem:...
    title: str
    detail: str | None = None
    status: int | None = None
    instance: str | None = None
    code: str | None = None   # canonical error code like KGF-1203
    ext: Mapping[str, Any] | None = None

def problem_from_exc(
    exc: BaseException,
    *,
    code_map: Mapping[type[BaseException], str] | None = None,
    operation: str | None = None,
    run_id: str | None = None,
) -> ProblemDetails:
    name = exc.__class__.__name__
    code = (code_map or {}).get(type(exc))
    # choose a stable type namespace; adjust to your taste
    type_uri = f"urn:kgf:problem:{name.lower()}"
    title = name.replace("_", " ")
    detail = str(exc) or None
    instance = f"urn:kgf:op:{operation}:run:{run_id}" if operation and run_id else None
    return ProblemDetails(type=type_uri, title=title, detail=detail, status=500, instance=instance, code=code)
```

## 1.3 `tools/_shared/observability.py`

```python
# tools/_shared/observability.py
from __future__ import annotations
from typing import Protocol

class MetricEmitter(Protocol):
    def emit_cli_run(self, *, operation: str, status: str, duration_s: float) -> None: ...

class _NoopEmitter:
    def emit_cli_run(self, *, operation: str, status: str, duration_s: float) -> None: ...

# Replace at import-time in your service/env bootstrap if you have Prometheus/StatsD, etc.
emitter: MetricEmitter = _NoopEmitter()
```

---

# 2) The façade itself

## 2.1 `tools/_shared/cli_runtime.py`

```python
# tools/_shared/cli_runtime.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, logging, os, re, time, uuid

from .paths import Paths
from .observability import emitter
from .problems import ProblemDetails, problem_from_exc

# --- Light logger fallback (works if you don't have a custom logger adapter) ---
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("kgf.cli")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logger

def _bind_fields(logger: logging.Logger, fields: Mapping[str, Any]) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra=fields)

# --- Route normalization ---
_TOKEN_RE = re.compile(r"[^a-z0-9_-]+")

def normalize_token(s: str) -> str:
    s = s.strip().lower().replace(" ", "-")
    s = s.replace(".", "-")
    s = _TOKEN_RE.sub("", s)
    if not s or s in {".", ".."}:
        raise ValueError("Invalid empty or traversal token in command route")
    if len(s) > 48:
        s = s[:48]
    return s

def normalize_route(segments: Sequence[str]) -> list[str]:
    route = [normalize_token(x) for x in segments]
    if not route:
        raise ValueError("command_path must have at least one segment")
    if len(route) > 6:
        raise ValueError("command_path too deep (max 6 segments)")
    return route

# --- Hash helper for artifacts ---
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Envelope builder ---
class EnvelopeBuilder:
    def __init__(self, *, command_path: list[str], operation: str, run_id: str, correlation_id: str):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self._env: dict[str, Any] = {
            "version": "1.1.0",
            "command_path": command_path,
            "operation": operation,
            # Back-compat (nullable):
            "command": command_path[0] if command_path else None,
            "subcommand": command_path[1] if len(command_path) > 1 else None,
            "run_id": run_id,
            "correlation_id": correlation_id,
            "started_at": now,
            "status": "success",
            "args": [],
            "env": {},
            "artifacts": [],
        }

    def set_args(self, argv: Sequence[str]) -> None:
        self._env["args"] = list(map(str, argv))

    def set_env(self, env: Mapping[str, str]) -> None:
        # Only include a few safe summaries (already redacted upstream)
        self._env["env"] = dict(env)

    def add_artifact(self, *, kind: str, path: Path, sha256: str | None = None) -> None:
        entry = {"kind": kind, "path": str(path)}
        if sha256:
            entry["sha256"] = sha256
        self._env["artifacts"].append(entry)

    def set_result(self, *, summary: str, payload: Any | None = None) -> None:
        self._env["result"] = {"summary": summary}
        if payload is not None:
            self._env["result"]["payload"] = payload

    def set_problem(self, problem: ProblemDetails) -> None:
        self._env["status"] = "error"
        self._env["problem"] = {k: v for k, v in asdict(problem).items() if v is not None}

    def finalize(self) -> dict[str, Any]:
        self._env["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return self._env

# --- Config & Context ---
@dataclass(frozen=True)
class CliRunConfig:
    command_path: Sequence[str]
    envelope_dir: Path | None = None
    correlation_id: str | None = None
    write_envelope_on: str = "always"   # "always" | "error" | "success"
    stdout_format: str = "minimal"      # "none" | "minimal" | "full"
    exit_on_error: bool = True
    error_code_map: Mapping[type[BaseException], str] = field(default_factory=dict)
    extra_context: Mapping[str, Any] = field(default_factory=dict)
    args_summary: Sequence[str] | None = None
    env_summary: Mapping[str, str] | None = None

    @classmethod
    def from_route(cls, *segments: str, **kw: Any) -> "CliRunConfig":
        return cls(command_path=normalize_route(segments), **kw)

@dataclass
class CliContext:
    command_path: list[str]
    operation: str
    run_id: str
    correlation_id: str
    started_monotonic: float
    logger: logging.LoggerAdapter
    paths: Paths

# --- CLI façade ---
def _now_id() -> str:
    return uuid.uuid4().hex[:6]

def _envelope_path(paths: Paths, route: list[str], started_at: str, run_id: str) -> Path:
    # started_at: "YYYY-MM-DDTHH:MM:SSZ"
    date = started_at[:10].replace("-", "")        # YYYYMMDD
    timepart = started_at[11:19].replace(":", "")  # HHMMSS
    directory = paths.cli_envelope_dir(route)
    filename = f"{date}-{timepart}-{run_id}.json"
    return directory / filename

def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)

def _validate_envelope(paths: Paths, envelope: dict[str, Any], logger: logging.LoggerAdapter) -> None:
    # Optional JSON-Schema validation if jsonschema is available
    try:
        import jsonschema, json
        schema_path = paths.repo_root / "schema" / "tools" / "cli_envelope.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
            jsonschema.validate(envelope, schema)
    except Exception as e:
        logger.warning("envelope_validation_failed", extra={"error": str(e)})

@contextmanager
def cli_run(cfg: CliRunConfig) -> Iterator[tuple[CliContext, EnvelopeBuilder]]:
    route = normalize_route(cfg.command_path)
    operation = ".".join(route)
    correlation_id = cfg.correlation_id or _now_id()
    run_id = _now_id()
    paths = Paths.discover()
    logger = _bind_fields(_get_logger(), {
        "operation": operation,
        "route": route,
        "run_id": run_id,
        "correlation_id": correlation_id,
        **(cfg.extra_context or {}),
    })
    env = EnvelopeBuilder(command_path=route, operation=operation, run_id=run_id, correlation_id=correlation_id)
    if cfg.args_summary: env.set_args(cfg.args_summary)
    if cfg.env_summary:  env.set_env(cfg.env_summary)

    started = time.monotonic()
    logger.info("cli_run_start")
    status = "success"
    problem: ProblemDetails | None = None
    try:
        yield (
            CliContext(
                command_path=route,
                operation=operation,
                run_id=run_id,
                correlation_id=correlation_id,
                started_monotonic=started,
                logger=logger,
                paths=paths,
            ),
            env,
        )
    except BaseException as exc:
        status = "error"
        problem = problem_from_exc(exc, code_map=cfg.error_code_map, operation=operation, run_id=run_id)
        env.set_problem(problem)
        if cfg.stdout_format != "none":
            code = f"{problem.code}: " if problem.code else ""
            print(f"{operation} [run={run_id}] ❌ {code}{problem.title}")
        if not cfg.exit_on_error:
            # We'll re-raise after finalize
            _reraise = True
        else:
            _reraise = False
    else:
        if cfg.stdout_format != "none":
            dur = time.monotonic() - started
            print(f"{operation} [run={run_id}] ✅ in {dur:.2f}s (corr={correlation_id})")
        _reraise = False
    finally:
        finished = time.monotonic()
        env_dict = env.finalize()
        env_dict["duration_ms"] = int((finished - started) * 1000)
        _validate_envelope(paths, env_dict, logger)

        out_dir = cfg.envelope_dir or paths.cli_envelope_dir(route)
        out_path = _envelope_path(paths, route, env_dict["started_at"], run_id)

        should_write = (
            cfg.write_envelope_on == "always" or
            (cfg.write_envelope_on == "error" and status == "error") or
            (cfg.write_envelope_on == "success" and status == "success")
        )
        if should_write:
            _atomic_write_json(out_path, env_dict)

        try:
            emitter.emit_cli_run(operation=operation, status=status, duration_s=(finished - started))
        except Exception:
            logger.warning("metrics_emit_failed")

        logger.info("cli_run_done", extra={"status": status, "duration_ms": env_dict["duration_ms"]})

        if status == "error" and cfg.exit_on_error:
            raise SystemExit(1)
        if status == "error" and _reraise:
            raise
```

---

# 3) Typer/Click Integration (automatic routing)

## 3.1 `tools/_shared/cli_integration.py`

```python
# tools/_shared/cli_integration.py
from __future__ import annotations
from typing import Any, Callable, Sequence
import functools
import click

from .cli_runtime import CliRunConfig, cli_run

def current_route(include_root: bool = False) -> list[str]:
    """
    Walks Click/Typer context to capture the full command path.
    Excludes the binary/root command unless include_root=True.
    """
    ctx = click.get_current_context()
    route: list[str] = []
    while ctx is not None:
        if ctx.info_name:  # group/command name
            route.append(ctx.info_name)
        ctx = ctx.parent
    route.reverse()
    if not include_root and route:
        # By convention, drop the top-most root (binary label) if present
        route = route[1:] if len(route) > 1 else route
    return route

def cli_operation(*, echo_args: bool = True, echo_env: bool = False):
    """
    Decorator that wraps a Typer/Click command in the CLI façade.

    Command function signature becomes:
      fn(ctx, env, *args, **kwargs)
    where:
      - ctx : CliContext
      - env : EnvelopeBuilder
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            route = current_route()
            cfg = CliRunConfig(
                command_path=route,
                args_summary=[f"{k}={v}" for k, v in kwargs.items()] if echo_args else None,
                env_summary=_collect_env_summary() if echo_env else None,
            )
            with cli_run(cfg) as (ctx, env):
                return fn(ctx, env, *args, **kwargs)
        return wrapper
    return deco

def _collect_env_summary() -> dict[str, str]:
    import os
    # Keep this small and safe; redact if you add tokens
    keys = ["PYTHONPATH", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES"]
    return {k: os.environ.get(k, "") for k in keys}
```

---

# 4) Update the envelope JSON-Schema

Edit `schema/tools/cli_envelope.json` to **v1.1** (only showing the new parts):

```json
{
  "$id": "https://kgfoundry/schema/tools/cli_envelope.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KGF CLI Envelope",
  "type": "object",
  "required": ["version", "command_path", "operation", "run_id", "status", "started_at", "finished_at"],
  "properties": {
    "version": { "type": "string", "const": "1.1.0" },
    "command_path": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string", "pattern": "^[a-z0-9_-]{1,48}$" }
    },
    "operation": { "type": "string", "pattern": "^[a-z0-9_-]+(\\.[a-z0-9_-]+)*$" },
    "command": { "type": ["string", "null"] },
    "subcommand": { "type": ["string", "null"] },
    "run_id": { "type": "string" },
    "correlation_id": { "type": "string" },
    "status": { "enum": ["success", "error"] },
    "args": { "type": "array", "items": { "type": "string" } },
    "env": { "type": "object", "additionalProperties": { "type": "string" } },
    "artifacts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["kind", "path"],
        "properties": {
          "kind": { "type": "string" },
          "path": { "type": "string" },
          "sha256": { "type": "string" }
        }
      }
    },
    "result": { "type": "object" },
    "problem": {
      "type": "object",
      "properties": {
        "type": { "type": "string" },
        "title": { "type": "string" },
        "detail": { "type": ["string", "null"] },
        "status": { "type": ["integer", "null"] },
        "instance": { "type": ["string", "null"] },
        "code": { "type": ["string", "null"] }
      }
    },
    "started_at": { "type": "string" },
    "finished_at": { "type": "string" },
    "duration_ms": { "type": "integer" }
  },
  "additionalProperties": true
}
```

---

# 5) Convert one CLI (example)

Assume a Typer layout like `src/download/cli.py`:

### Before (typical)

```python
import typer
app = typer.Typer()

@app.command()
def harvest(source: str, max_items: int = 200):
    # ... bespoke logging, output, try/except ...
    ...
```

### After (with façade)

```python
# src/download/cli.py
from __future__ import annotations
import typer
from tools._shared.cli_integration import cli_operation
from tools._shared.cli_runtime import CliContext, EnvelopeBuilder  # types only

app = typer.Typer()

@app.command()
@cli_operation(echo_args=True, echo_env=False)
def harvest(ctx: CliContext, env: EnvelopeBuilder, source: str, max_items: int = 200):
    ctx.logger.info("begin", extra={"source": source, "max_items": max_items})
    # ... run the real work ...
    # Register any produced artifacts:
    # env.add_artifact(kind="manifest", path=ctx.paths.docs_data / "download" / "manifest.json")
    env.set_result(summary=f"Harvested {max_items} docs from {source}")
```

Run it:

```
python -m src.download.cli harvest --source openalex --max-items 50
# -> docs/_data/cli/download/harvest/20251106-091210-8f3a1c.json
# -> stdout: download.harvest [run=8f3a1c] ✅ in 0.32s (corr=3b6f…)
```

### Nested commands

If you’ve nested Typer apps:

```python
download = typer.Typer()
harvest = typer.Typer()
openalex = typer.Typer()
app.add_typer(download, name="download")
download.add_typer(harvest, name="harvest")
harvest.add_typer(openalex, name="openalex")

@openalex.command("run")
@cli_operation()
def run_openalex(ctx: CliContext, env: EnvelopeBuilder, max_items: int = 200):
    ...
```

Route becomes `["download","harvest","openalex","run"]`, and envelope directory mirrors the route.

---

# 6) Roll out to all CLIs

**Find commands** (examples):

```bash
grep -RIn --include="*.py" -E "@(app|.*)\.command\(" src tools
grep -RIn --include="*.py" -E "Typer\(" src tools
```

For each command function:

1. Add:

   ```python
   from tools._shared.cli_integration import cli_operation
   from tools._shared.cli_runtime import CliContext, EnvelopeBuilder
   ```
2. Add the decorator:

   ```python
   @cli_operation(echo_args=True, echo_env=False)
   ```
3. Change signature:

   ```python
   def mycmd(ctx: CliContext, env: EnvelopeBuilder, <existing args>):
       ...
   ```
4. Remove bespoke try/except, ad-hoc envelope or path logic, and any duplicated start/end logging. Use `ctx.logger` and `env.*`.
5. If you produced files, register them:

   ```python
   env.add_artifact(kind="...", path=some_path, sha256=sha256_file(some_path))
   ```

> Tip: convert one CLI fully, run it, inspect the envelope, then apply the pattern module-by-module.

---

# 7) OpenAPI & Docs alignment (optional but recommended)

If you have a Typer→OpenAPI generator, consume the new fields:

* Use `operation` as your canonical `operationId`.
* Add an `x-cli` vendor extension with `command_path` so docs can link the exact envelope directory:

  ```yaml
  x-cli:
    operation: download.harvest.openalex.run
    command_path: ["download","harvest","openalex","run"]
  ```

In docs, you can show the latest envelope for an operation by reading `docs/_data/cli/<route...>/` and sorting by filename.

---

# 8) Testing

Add `tests/test_cli_runtime.py`:

```python
from __future__ import annotations
from pathlib import Path
from tools._shared.cli_runtime import CliRunConfig, cli_run

def test_success(tmp_path: Path, monkeypatch):
    from tools._shared.paths import Paths
    monkeypatch.setattr(Paths, "discover", staticmethod(lambda: Paths(repo_root=tmp_path, docs_data=tmp_path/"docs/_data", cli_out_root=tmp_path/"docs/_data/cli")))
    cfg = CliRunConfig.from_route("demo", "ok", write_envelope_on="always", exit_on_error=False)
    with cli_run(cfg) as (ctx, env):
        env.set_result(summary="ok")

    # assert envelope exists
    out_dir = tmp_path / "docs/_data/cli/demo/ok"
    paths = sorted(out_dir.glob("*.json"))
    assert len(paths) == 1
    data = paths[0].read_text()
    assert '"status": "success"' in data
    assert '"operation": "demo.ok"' in data

def test_error(tmp_path: Path, monkeypatch):
    from tools._shared.paths import Paths
    monkeypatch.setattr(Paths, "discover", staticmethod(lambda: Paths(repo_root=tmp_path, docs_data=tmp_path/"docs/_data", cli_out_root=tmp_path/"docs/_data/cli")))
    cfg = CliRunConfig.from_route("demo", "fail", write_envelope_on="always", exit_on_error=False)
    try:
        with cli_run(cfg) as (ctx, env):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    out_dir = tmp_path / "docs/_data/cli/demo/fail"
    assert any('"status": "error"' in p.read_text() for p in out_dir.glob("*.json"))
```

---

# 9) CI hooks

* **pre-commit**: nothing special; this is pure Python.
* **Tests**: ensure `pytest -q` runs the new tests.
* **Schema check** (optional): add a tiny script to validate yesterday’s envelopes under `docs/_data/cli/**` against the schema.

---

# 10) Error codes (optional polishing)

If you have a canonical error code table (`tools/error_codes.py`) map common exceptions:

```python
# somewhere near app bootstrap
from tools._shared.cli_runtime import CliRunConfig
from tools._shared.errors import ConfigurationError, ExternalRateLimited

ERROR_CODE_MAP = {
    ConfigurationError: "KGF-1001",
    ExternalRateLimited: "KGF-1203",
}
# pass via CliRunConfig(..., error_code_map=ERROR_CODE_MAP)
```

If using `@cli_operation`, you can make a repo-level default by editing `cli_integration.cli_operation` to include `error_code_map=ERROR_CODE_MAP`.

---

# 11) Rollout plan (safe & incremental)

1. Add the four shared modules + schema update.
2. Convert one CLI (e.g., `download.harvest`) and verify:

   * stdout line appears,
   * JSON envelope written in the expected path,
   * content shows `operation`, `command_path`, `status`, `duration_ms`.
3. Convert other CLIs gradually (search + repeat).
4. Update any docs/scripts that expected the old `download/harvest/` 2-level path—symlink or adjust to the multi-level route.
5. (Optional) update OpenAPI generator to use `operation`.

---

# 12) Common pitfalls (and fixes)

* **“command_path too deep”**: You can lift the `max 6 segments` guard if you really need more; keep it to manage metrics cardinality.
* **Including the binary name**: By default we drop it; set `include_root=True` in `current_route` if you want it.
* **Schema import errors**: If `jsonschema` isn’t installed, validation logs a warning but continues. Add `jsonschema` to your dev extras to enable.
* **Forgetting to add `ctx, env`**: The decorator expects your command to accept `(ctx, env, ...)`; Typer’s error will make this obvious.

---

# 13) One-file “How to add a new CLI” (put this in your docs)

1. Write your Typer command function.
2. Add `@cli_operation()` decorator.
3. Accept `ctx: CliContext, env: EnvelopeBuilder` as the first two parameters.
4. Log via `ctx.logger`; register artifacts with `env.add_artifact`; set results; let the façade do the rest.

---

If you want, I can prep a PR patch that adds these files, converts one real CLI in `src/download/cli.py`, and bumps the schema to v1.1 so you can see the exact diffs.


