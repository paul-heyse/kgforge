## 0. Orientation (AI agent submission pre-flight)
- [ ] 0.1 Confirm runner image aligns with repo toolchain (`scripts/bootstrap.sh`).
- [ ] 0.2 Cache `openspec/changes/tools-hardening/` docs and `openspec/AGENTS.md` locally for reference.
- [ ] 0.3 Review canonical contracts: `tools/_shared/logging.py`, `tools/_shared/proc.py`, `tools/docstring_builder/models.py`, schemas under `schema/tools/`.
- [ ] 0.4 Capture baseline diagnostics (attach to execution report):
  - `uv run ruff check tools`
  - `uv run pyrefly check tools`
  - `uv run mypy tools`
- [ ] 0.5 Generate a four-item design note (Summary, API sketch, Data contracts, Test plan) reflecting this change.

## 1. Shared Infrastructure
- [ ] 1.1 Refactor `_shared/logging.py` to delegate to `kgfoundry_common.logging` (JSON formatter + correlation IDs), fix pyrefly override issues, and document the Problem Details logging contract.
- [ ] 1.2 Harden `_shared/proc.py`: define `JsonValue` type alias, enforce absolute executables, capture stdout/stderr as `tuple[str, str]`, emit Problem Details on failure.
- [ ] 1.3 Introduce `_shared/problem_details.py` with builders/helpers plus example at `docs/examples/tools_problem_details.json`.

  API (implement exactly to ease adoption):
  - `JsonPrimitive = str | int | float | bool | None`
  - `JsonValue = JsonPrimitive | Sequence[JsonValue] | Mapping[str, JsonValue]`
  - `ProblemDetailsDict = Mapping[str, JsonValue]`
  - `def build_problem_details(*, type: str, title: str, status: int, detail: str, instance: str, extensions: Mapping[str, JsonValue] | None = None) -> ProblemDetailsDict`
  - `def problem_from_exception(exc: Exception, *, type: str, title: str, status: int, instance: str, extensions: Mapping[str, JsonValue] | None = None) -> ProblemDetailsDict`
  - `def render_problem(problem: ProblemDetailsDict) -> str`  # JSON string for stdout
- [ ] 1.4 Add pytest coverage for logging/proc/problem_details behavior (`tests/tools/shared/test_logging.py`, `test_proc.py`).

- [ ] 1.5 Add versioned base CLI envelope schema at `schema/tools/cli_envelope.json` and helpers to emit it from CLIs.

## 2. Docstring Builder Integration
- [ ] 2.1 Update `normalizer.py`, `normalizer_signature.py`, `normalizer_annotations.py`, `policy.py`, `render.py`, and `docfacts.py` to consume typed models and eliminate `Any` usage.
- [ ] 2.2 Fix pyrefly/mypy issues in observability module: ensure stub metrics implement `.labels()` and return `self`, guard imports.
- [ ] 2.3 Align CLI (`cli.py`) with typed adapters, Problem Details emission, schema validation, and metrics usage.
- [ ] 2.4 Finalize plugin Protocols (`plugins/base.py`, `plugins/__init__.py`), compatibility shim, and update bundled plugins.
- [ ] 2.5 Add regression tests: schema validation (`tests/tools/docstring_builder/test_schemas.py`), plugin behavior (`test_plugins.py`), CLI integration (`test_cli.py`).

## 3. Documentation Pipelines
- [ ] 3.1 Introduce typed models for analytics, graphs, and test maps; refactor `build_agent_catalog.py`, `build_graphs.py`, `build_test_map.py`, `export_schemas.py`, `render_agent_portal.py` to use them.
- [ ] 3.2 Replace blind exceptions with typed `DocumentationBuildError` hierarchy; migrate all ad-hoc `subprocess.run`/shell helpers to `_shared.proc.run_tool` with enforced timeouts and sanitized environment.
- [ ] 3.3 Add JSON Schemas (`schema/tools/doc_analytics.json`, `doc_graph_manifest.json`, `doc_test_map.json`), fixtures, and validation tests.
- [ ] 3.4 Update docs generator tests (`tests/tools/docs/test_*`) with table-driven cases covering success, validation failure, and subprocess errors.

- [ ] 3.5 Add typed CLI settings via `pydantic_settings` (env-only configuration, fail-fast) and keep libraries free of config.

## 4. Navmap & Ancillary CLIs
- [ ] 4.1 Create `tools/navmap/models.py` and adapters for navmap documents; refactor builders/checkers/repair scripts to use typed models.
- [ ] 4.2 Replace prints/import-at-runtime with structured logging and top-level imports; integrate `run_tool` wrapper.
- [ ] 4.3 Define `schema/tools/navmap_document.json` plus pytest coverage for migration/repair outputs.
- [ ] 4.4 Harden other CLIs (`detect_pkg.py`, `generate_docstrings.py`, `hooks/docformatter.py`, lint helpers) with typed `main()` functions, structured logging, safe subprocess via `run_tool`, and add `--json` output using the base CLI envelope schema.

## 5. Observability & Testing
- [ ] 5.1 Ensure all modules expose `get_logger(__name__)` and register Prometheus metrics via typed providers; add OpenTelemetry span hooks where applicable.
- [ ] 5.2 Add doctests/xdoctests referencing Problem Details samples and schema usage; ensure examples execute in CI.
- [ ] 5.3 Expand pytest suite (`tests/tools/`) with parametrized cases for edge inputs, failure modes, and retry/idempotency checks; name/mark tests to map to spec scenarios.
- [ ] 5.4 Update performance tests (`tests/tools/docstring_builder/test_perf.py`) if data model changes affect baselines.

## 6. Documentation & Rollout
- [ ] 6.1 Document new schemas, logging conventions, and CLI envelopes in `docs/contributing/quality.md` and relevant READMEs.
- [ ] 6.2 Add changelog entry detailing CLI JSON envelope versioning and feature flag defaults.
- [ ] 6.3 Capture telemetry plan: dashboards/alerts on Prometheus metrics (docbuilder runs, plugin failures, navmap errors).
- [ ] 6.4 Prepare rollout note outlining feature flag flip timeline and compatibility shim removal plan.

## 7. Validation & Sign-off
- [ ] 7.1 Run quality gates:
  - `uv run ruff format && uv run ruff check --fix`
  - `uv run pyrefly check && uv run mypy --config-file mypy.ini`
  - `uv run pytest -q tests/tools`
  - `python tools/check_imports.py`
  - `uv run pip-audit --strict`
  - `make artifacts && git diff --exit-code`
  - `openspec validate tools-hardening --strict`
- [ ] 7.2 Attach command outputs + schema validation logs to execution report.
- [ ] 7.3 Flip `DOCSTRINGS_TYPED_IR` default once metrics show stability; document completion in rollout note.


## Appendix A — File-by-file migration guide (junior-ready)

This appendix provides concrete, unambiguous steps per file. Follow the same pattern in any similar modules not listed here.

### Conventions and defaults
- Use `tools._shared.proc.run_tool()` for all subprocess calls, with defaults:
  - **git**: `timeout=10.0`
  - **graphviz (dot)**: `timeout=30.0`
  - **docstring-builder**: `timeout=20.0`
  - **doctoc/formatters**: `timeout=20.0`
- Structured logging via:
  - `from tools._shared.logging import get_logger, with_fields`
  - `logger = get_logger(__name__)`
  - Use `with_fields(logger, correlation_id=..., operation=..., command=...)`
  - Prefer `logger.info/warning/error` with `extra={...}` fields rather than string interpolation.
- Errors:
  - Wrap failures in typed errors (e.g., `CatalogBuildError`, `DocumentationBuildError`, `ToolExecutionError`).
  - When `--json` is provided: emit Problem Details payload inside the base CLI envelope.
- Config:
  - For CLIs, add a `Settings` (pydantic_settings) only if env is used; otherwise accept CLI options explicitly.
  - Libraries must not read environment variables directly.

### tools/generate_docstrings.py
Required changes:
- Replace `subprocess.run` with `run_tool`, enforce `timeout=20.0` and structured logs.
- On failure, convert `ToolExecutionError` to exit code 1 and include Problem Details when `--json` is added (optional CLI enhancement).

Reference before:
```16:22:/home/paul/kgfoundry/tools/generate_docstrings.py
def run_builder(extra_args: list[str] | None = None) -> None:
    """Execute the docstring builder CLI with optional arguments."""
    args = extra_args or []
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "update", *args]
    LOGGER.info("[docstrings] Running docstring builder: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

Implementation checklist:
- Import `run_tool` and swap call.
- Add `timeout=20.0`.
- Log `command`, `status`, and `duration_ms` when available.

### tools/docs/build_agent_catalog.py
Required changes:
- Replace all `subprocess.run` git calls with `run_tool(..., timeout=10.0)`.
- Replace `print(...)` error paths with `logger.error(...)` and, when adding `--json`, emit CLI envelope with Problem Details.
- Add `--json` flag to argparse; include minimal envelope fields: `schemaVersion`, `schemaId`, `generatedAt`, `status`, `command`, `subcommand`, `durationSeconds`, `files` (may be empty), `errors` (with Problem Details when applicable).

References:
```650:661:/home/paul/kgfoundry/tools/docs/build_agent_catalog.py
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )
        except subprocess.CalledProcessError as exc:
            message = "Unable to resolve repository SHA"
            raise CatalogBuildError(message) from exc
```
```1566:1575:/home/paul/kgfoundry/tools/docs/build_agent_catalog.py
        except CatalogBuildError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps([dataclasses.asdict(result) for result in results], indent=2))
        return 0
    try:
        catalog = builder.build()
        builder.write(catalog, args.output, args.schema)
    except CatalogBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

Implementation checklist:
- Import `get_logger`, `with_fields`, and `run_tool`.
- Swap git calls to `run_tool(["git", ...], timeout=10.0)` and check `returncode`.
- Replace all prints with logger; when `--json` add envelope with Problem Details on non-zero exit.

### tools/docs/build_graphs.py
Required changes:
- Delete or refactor `sh()` wrapper to call `run_tool(cmd, timeout=30.0)`.
- Replace `ensure_bin` print warnings with structured logs and early `DocumentationBuildError`.
- Add `--json` option to CLI if present; otherwise just structured logs.

References:
```244:273:/home/paul/kgfoundry/tools/docs/build_graphs.py
def sh(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Compute sh.
    ...
    """
    return subprocess.run(
        cmd, check=check, cwd=str(cwd) if cwd else None, text=True, capture_output=False
    )
```

### tools/hooks/docformatter.py
Required changes:
- Replace all `subprocess.run` with `run_tool` (timeout 20s) and check return code.
- Replace prints with structured logs; return non-zero on failure.

References:
```31:40:/home/paul/kgfoundry/tools/hooks/docformatter.py
    result = subprocess.run(
```
```56:74:/home/paul/kgfoundry/tools/hooks/docformatter.py
    repo = subprocess.run(
```

### tools/gen_readmes.py
Required changes:
- Replace `subprocess.run` (doctoc, etc.) with `run_tool` (`timeout=20.0`).
- Replace prints with structured logs.

Reference:
```700:714:/home/paul/kgfoundry/tools/gen_readmes.py
            print(result.stdout.strip())
            print(result.stderr.strip(), file=sys.stderr)
```

### tools/check_docstrings.py
Required changes:
- Replace single `subprocess.run(cmd, check=True)` with `run_tool(cmd, timeout=20.0, check=True)`; translate exceptions into exit code 1.

Reference:
```96:104:/home/paul/kgfoundry/tools/check_docstrings.py
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "lint", "--no-docfacts", *paths]
    subprocess.run(cmd, check=True)
```

### tools/docs/build_test_map.py
Required changes:
- Replace `print` paths with structured logs; add `--json` option to emit CLI envelope with Problem Details for failures.

Reference:
```932:955:/home/paul/kgfoundry/tools/docs/build_test_map.py
        print(f"[testmap] ERROR: {exc}", file=sys.stderr)
        print(f"[testmap] FAIL: {sum(1 for x in lints if x['severity'] == 'error')} error(s)")
```

### tools/navmap/*.py (check_navmap.py, repair_navmaps.py, migrate_navmaps.py, strip_navmap_sections.py)
-### tools/docs/build_artifacts.py
Required changes:
- Replace `_run_step` internals to use `run_tool(command, cwd=REPO_ROOT, timeout=20.0)`.
- Replace direct stdout/stderr writes with structured logs and propagate exit codes; add `--json` if you want machine output.

Reference:
```73:82:/home/paul/kgfoundry/tools/docs/build_artifacts.py
def _run_step(name: str, command: list[str], message: str) -> int:
    """Execute a single artefact regeneration step."""
    result = subprocess.run(
        command, cwd=REPO_ROOT, check=False
    )
    if result.returncode != 0:
        sys.stderr.write(f"[artifacts] {name} failed (exit {result.returncode})\n")
        return result.returncode
    sys.stdout.write(f"{message}\n")
    return 0
```

Required changes:
- Replace all `print` calls with structured logging; exit codes reflect success/failure.
- Where external tools are invoked (rare), use `run_tool` with appropriate timeouts.
- For `repair_navmaps.py`, add `--json` option to output base CLI envelope on failure with Problem Details.

### Base CLI envelope tasks (for CLIs gaining --json)
For each CLI (catalog builder, graphs, test map, navmap repair):
1. Add `--json` to argparse and collect `start = time.monotonic()`; `end` at exit.
2. Build envelope:
   - `schemaVersion` (e.g., "1.0.0"), `schemaId` (e.g., `https://kgfoundry.dev/schema/cli-envelope.json`),
   - `status` ("success"|"error"|"violation"|"config"), `generatedAt` (UTC ISO 8601),
   - `command` (tool name), `subcommand` (string), `durationSeconds` (float),
   - `files` (empty or per-file entries), `errors` (array of `{file, status, message, problem?}`), `problem` (Problem Details on failure).
3. Validate against `schema/tools/cli_envelope.json` using `jsonschema.Draft202012Validator`.
4. On failure, write envelope JSON to stdout and return non-zero.

Applies to: `tools/docs/build_agent_catalog.py`, `tools/docs/build_graphs.py`, `tools/docs/build_test_map.py`, `tools/navmap/repair_navmaps.py`, and (as needed) `tools/docs/build_agent_api.py`, `tools/docs/build_agent_analytics.py`.

### Structured logging fields (minimum set)
- `correlation_id` (from `kgfoundry_common.logging` context, or generate once per run)
- `operation` (e.g., "git", "dot", "build_catalog")
- `status` (e.g., "started", "success", "error")
- `command` (array or string)
- `duration_ms` (integer)
- Optional: `path_count`, `file`, `reason`, `returncode`

### Default timeouts and env allowlist
- Use `_shared.proc._sanitise_env` defaults; extend only if strictly required.
- Git: 10s; Graphviz: 30s; Doctoc/formatters: 20s; Python module invocations: 20s.

### Tests to add (create under `tests/tools/`)
- `tests/tools/shared/test_proc.py`: table-driven tests for `run_tool` happy path, timeout, missing executable; assert Problem Details shape.
- `tests/tools/shared/test_logging.py`: adapter merges `extra`, preserves correlation_id, no duplicate `NullHandler`.
- `tests/tools/docs/test_catalog_cli.py`: `--json` envelope success/failure, schema validation, Problem Details presence.
- `tests/tools/docs/test_graphs_cli.py`: ensure `run_tool` used for `dot`, envelope on error.
- `tests/tools/navmap/test_repair_cli.py`: envelope output with counterexamples.

