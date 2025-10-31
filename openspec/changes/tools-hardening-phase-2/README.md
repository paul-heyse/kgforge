# tools-hardening-phase-2

- [proposal](./proposal.md)
- [tasks](./tasks.md)
- [spec delta](./specs/tools-suite/spec.md)

### Validation Commands
- `uv run ruff format && uv run ruff check --fix`
- `uv run pyrefly check`
- `uv run mypy --config-file mypy.ini`
- `uv run pip-audit --strict`
- `python tools/check_imports.py`
- `spectral lint path/to/openapi.yaml` *(when HTTP surfaces exist)*
- `uv run pytest -q`
- `make artifacts`


### Metrics & Settings Toggles
- `TOOLS_METRICS_ENABLED=0` disables Prometheus counters/histograms emitted by `tools._shared.metrics` – useful if dashboards regress or scrape load must be rolled back quickly.
- `TOOLS_TRACING_ENABLED=0` stops OpenTelemetry span emission while leaving metrics untouched.
- `TOOLS_EXEC_ALLOWLIST` (comma-separated globs) controls the subprocess allow list enforced by `tools._shared.proc`; updating the variable provides an immediate rollback lever for newly blocked binaries.
- All toggles are read via `tools._shared.settings.ToolRuntimeSettings` so changes only require restarting the invoking CLI.

