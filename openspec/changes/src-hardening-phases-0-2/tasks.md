## 0. Quality Gates & Baseline Diagnostics
- [ ] 0.1 Run `scripts/bootstrap.sh`; confirm Python 3.13.9 + `uv` parity.
- [ ] 0.2 Capture baseline outputs (attach logs and artefacts):
  - `uv run ruff format && uv run ruff check src`
  - `uv run pyrefly check src --show-suppressed`
  - `uv run mypy --config-file mypy.ini src`
  - `python tools/check_imports.py`
  - `uv run pip-audit --strict`
  - `uv run pytest -q --maxfail=1`
  - `uv run python tools/docstr_coverage.py --json` (or successor) for current docstring coverage
- [ ] 0.3 Draft and circulate the four-item design note (summary, typed API sketch, data/schema contracts, test plan) for reviewer sign-off and map scenarios to forthcoming tests.
- [ ] 0.4 Register telemetry dashboards (structured logs, Prometheus metrics, OpenTelemetry traces) covering search latency, error rate, schema validation failures, and correlation IDs.
- [ ] 0.5 Update engineering handbook to require the Phase 0 checklist before approving new `src/` refactors; document rollback and verification expectations.

## 2. Typed Refactors, Contracts, & Architecture Hardening
- [ ] 2.1 **Clarity & API design**: eliminate `D2xx/DOC5xx` lint findings by adding NumPy-style docstrings with one-line summaries, fully annotated signatures (PEP 695 where useful), and explicit `raise ... from err` chains; publish Problem Details JSON samples per module.
- [ ] 2.2 **Data contracts & schemas**: create/refresh JSON Schema 2020-12 + OpenAPI 3.2 artefacts (`schema/search/*.json`, FastAPI spec), including examples and versioning notes; add round-trip validation tests; lint via `python -m kgfoundry_common.schema_helpers validate` and Spectral.
- [ ] 2.3 **Testing strategy**: expand pytest suites with `@pytest.mark.parametrize` tables covering happy, edge, failure, retry, and injection scenarios; ensure doctest/xdoctest runs; record scenario↔test mappings in the execution note.
- [ ] 2.4 **Type safety**: replace `Any` with `Protocol`/`TypedDict` facades across `agent_catalog`, `search_api`, `registry`, `vectorstore_faiss`, and `kgfoundry_common.parquet_io`; justify remaining `# type: ignore[...]` entries and keep pyrefly/mypy clean without suppressions.
- [ ] 2.5 **Logging, observability, configuration**: standardize on module loggers with `NullHandler`, structured logging with correlation IDs, Prometheus metrics, and traces; migrate settings to typed env-driven configuration (`pydantic_settings` or equivalent); document timeout/cancellation rules and propagate `contextvars` or thread pools for async boundaries.
- [ ] 2.6 **Security & supply-chain**: remove unsafe `pickle`/`eval`, validate/sanitize all untrusted inputs, enforce `Path.resolve` whitelists, and keep `pip-audit --strict` and secret scanning clean; pin optional dependency extras sensibly.
- [ ] 2.7 **Modularity & structure**: reinforce layering via updated import-linter contracts and refactor high-complexity functions into single-responsibility units; ensure domain logic remains pure and I/O wrapped in adapters.
- [ ] 2.8 **Performance & scalability**: benchmark FAISS/BM25/SPLADE flows, set p95 latency/memory budgets, use `time.monotonic()` for durations, and document any exceptions; vectorize or stream heavy operations.
- [ ] 2.9 **Idempotency & retries**: verify CLI/HTTP/queue entry points are idempotent or document retry semantics; add tests proving repeated calls converge and Problem Details responses guide clients.
- [ ] 2.10 **File/time/number hygiene**: adopt `pathlib.Path`, timezone-aware datetimes, and `decimal.Decimal` for currency-like values; ban new `os.path` and naive datetime usage.
- [ ] 2.11 Flip feature flags (`AGENT_SEARCH_TYPED`, `SEARCH_API_TYPED`) after telemetry and tests meet acceptance, while maintaining compatibility shims until rollout sign-off.

## 3. Validation, Packaging, & Rollout
- [ ] 3.1 Run full quality gates with recorded outputs: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`, `uv run pytest -q`, doctest/xdoctest suite, `python tools/check_imports.py`, `uv run pip-audit --strict`, `make artifacts && git diff --exit-code`, `openspec validate src-hardening-phases-0-2`.
- [ ] 3.2 Verify packaging & distribution: build wheels via `pip wheel .`, install with `pip install .[faiss,duckdb,splade,gpu]` in a clean venv, and confirm entry points/metadata align with PEP 621.
- [ ] 3.3 Update CHANGELOG with SemVer language, document deprecations and migration paths, and ensure docs/Agent Portal examples are runnable and cross-link schemas.
- [ ] 3.4 Confirm telemetry dashboards capture error logs, metrics, and traces for failing scenarios; ensure signals respect budgets and correlation IDs.
- [ ] 3.5 Flip feature flags to their permanent state post-staging burn-in, document rollback procedures, and archive the execution note with command outputs, telemetry snapshots, benchmarks, and follow-up items.

