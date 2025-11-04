## 1. Implementation
- [ ] 1.1 Establish shared fixtures and utilities.
  - Create `tests/conftest.py` helpers for `CollectorRegistry`, in-memory OTEL span exporter, and search option factories.
  - Provide fixtures `search_options_factory`, `problem_details_loader`, `cli_runner` (using `click.testing.CliRunner` or equivalent).
- [ ] 1.2 Expand search options test coverage.
  - Add `tests/agent_catalog/test_search_options_table.py` with parametrized cases covering facets, candidate pools, alpha overrides, missing dependencies.
  - Verify each case asserts Problem Details exceptions and structured log contents.
- [ ] 1.3 Add schema validation round-trip tests.
  - Create `tests/agent_catalog/test_catalog_schema_roundtrip.py` using fixtures to serialize/deserialize search documents and compare against schema examples.
  - Include failure cases confirming validation errors produce Problem Details JSON.
- [ ] 1.4 Extend orchestration CLI tests.
  - Implement `tests/orchestration/test_index_cli.py` verifying idempotent runs of `index_bm25`/`index_faiss` using temporary directories.
  - Parametrize success/failure cases; assert structured log/metric emissions and no duplicate side effects on retries.
- [ ] 1.5 Add Prometheus wiring tests.
  - Introduce `tests/kgfoundry_common/test_prometheus_metrics.py` capturing counters/histograms via isolated registries.
  - Test both success and failure paths (Problem Details raising increments error counter).
- [ ] 1.6 Capture logs and traces in tests.
  - Use `caplog` to assert structured log fields for failure scenarios.
  - Integrate in-memory OTEL span exporter fixture; assert spans include error status and attributes (operation, request_id).
- [ ] 1.7 Add doctest/xdoctest snippets.
  - Update docstrings in `kgfoundry.agent_catalog.search`, `orchestration.cli`, `vectorstore.faiss_adapter` to include runnable search/FAISS examples.
  - Enable doctest collection by updating `pytest.ini` or `pyproject.toml` (`addopts = --doctest-modules`).
  - Ensure any environment-sensitive examples guard against missing dependencies.
- [ ] 1.8 Validate CLI/HTTP idempotency and retries.
  - Add tests in `tests/search_api/test_client.py` verifying repeated `get`/`post` calls with same payload produce identical results and no duplicate operations (mock backend).
  - Confirm Problem Details errors follow RFC 9457 format and include correlation IDs.
- [ ] 1.9 Update documentation and artifacts.
  - Document new testing strategy in `docs/contributing/testing.md` and observability expectations in `docs/reference/observability.md`.
  - Ensure docstring examples render correctly; run `make artifacts` to sync docs.
- [ ] 1.10 Execute iterative quality gates.
  - After tests and docstrings updated, run `uv run pytest -q tests/agent_catalog tests/orchestration tests/kgfoundry_common tests/search_api`.
  - Run `uv run pytest --doctest-modules kgfoundry` to confirm doctest coverage.
  - Check telemetry packages with `uv run pyrefly check` and `uv run pyright --warnings --pythonversion=3.13`.

## 2. Final Verification
- [ ] 2.1 Capture coverage metrics (if feasible) focusing on new tests.
- [ ] 2.2 Execute full suite: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pytest -q`, `make artifacts`.
- [ ] 2.3 Assemble release notes summarizing new test coverage, telemetry expectations, and Problem Details validation.

