## 1. Implementation

- [ ] 1.1 Create lifecycle module.
  - Add `docs/toolchain/_shared/lifecycle.py` containing `DocToolSettings`, `DocToolContext`, `DocMetrics`, `DocLifecycle`, and Problem Details helpers.
  - Implement structured logging, correlation IDs, and Prometheus counters/histograms (`kgfoundry_docs_operation_total`, `kgfoundry_docs_operation_duration_seconds`).
- [ ] 1.2 Refactor `build_symbol_index.py`.
  - Use `DocToolSettings.parse_args` for CLI options, construct context, and wrap execution with `DocLifecycle.run`.
  - Replace print statements with structured logging and Problem Details for errors.
- [ ] 1.3 Refactor `symbol_delta.py`.
  - Apply the same context/lifecycle pattern; ensure delta generation and output remain unchanged apart from logging.
- [ ] 1.4 Refactor `validate_artifacts.py`.
  - Adopt shared settings/context/lifecycle; ensure validation failures emit consistent Problem Details and metrics.
- [ ] 1.5 Logging & observability alignment.
  - Ensure all scripts emit `operation`, `status`, `artifact`, and correlation ID fields in logs.
  - Confirm metrics are registered once and include appropriate labels.
- [ ] 1.6 Quality gates.
  - Run `uv run ruff format && uv run ruff check --fix docs/toolchain`, `uv run pyright --warnings --pythonversion=3.13 docs/toolchain`, and `uv run pyrefly check docs/toolchain` to maintain lint/type cleanliness.

## 2. Testing

*(Deferred)* — unit tests and documentation updates will be handled in a follow-up change per scope direction.
