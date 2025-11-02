## 1. Implementation
- [ ] 1.1 Capture current behaviour
  - Snapshot CLI JSON payloads, manifest output, docfacts diff handling, and metrics usage for baseline tests.
  - Document existing Ruff failures (C901, PLR091x, PLR1702) on `tools/docstring_builder/orchestrator.py`.
- [ ] 1.2 Introduce orchestration primitives
  - Add `PipelineContext`, `CommandContext`, and `ErrorEnvelope` dataclasses with full typing and docstrings.
  - Implement `MetricsRecorder` wrapper for Prometheus counters/histograms used by the builder.
- [ ] 1.3 Extract file processing
  - Implement `FileProcessor` module encapsulating harvest, plugin transforms, edit application, cache updates, preview generation, and docfacts building.
  - Add focused unit tests covering cache hits, harvest errors, check/update behaviour, and preview generation.
- [ ] 1.4 Extract docfacts coordination
  - Implement `DocfactsCoordinator` managing provenance merging, drift detection, schema validation, and file writes.
  - Add tests for check/update modes, provenance overrides, diff emission, and failure scenarios.
- [ ] 1.5 Extract diff & manifest management
  - Implement `DiffManager` to handle docstring/docfacts/schema diff files and baseline comparisons.
  - Implement `ManifestBuilder` to assemble manifests using typed inputs and safe JSON writes.
- [ ] 1.6 Build pipeline runner
  - Implement `PipelineRunner` orchestrating context setup, plugin/policy loading, file processing loop, docfacts reconciliation, metrics updates, manifest/diff writing, and CLI payload synthesis.
  - Ensure methods remain within complexity thresholds; add integration-style tests with fakes/mocks.
- [ ] 1.7 Refresh failure summary rendering
  - Replace `_print_failure_summary` with `FailureSummaryRenderer` consuming typed summaries; add tests covering empty and populated error lists.
- [ ] 1.8 Wire orchestrator entry point
  - Update `run_docstring_builder` to instantiate helpers, run pipeline, and return typed results.
  - Ensure `render_cli_result` and `render_failure_summary` consume new structures without behaviour drift.

## 2. Testing
- [ ] 2.1 Unit tests for new helpers
  - Add test modules for file processor, docfacts coordinator, diff manager, manifest builder, metrics recorder, and failure summary renderer.
- [ ] 2.2 Pipeline integration tests
  - Construct fixture pipelines with stub plugin/policy/docfacts data ensuring exit statuses, error aggregation, and manifest outputs match baseline JSON snapshots.
- [ ] 2.3 Regression comparisons
  - Compare CLI JSON payloads, manifest files, and docfacts outputs pre/post refactor using golden fixtures.
- [ ] 2.4 Observability verification
  - Assert metrics counters/histograms update via `MetricsRecorder`; validate structured logging fields emitted by helpers using `caplog`.
- [ ] 2.5 Quality gates
  - Run `uv run ruff format && uv run ruff check --fix`.
  - Run `uv run pyrefly check` and `uv run mypy --config-file mypy.ini`.
  - Run `uv run pytest -q` for entire suite.
  - Run `make artifacts && git diff --exit-code`.
  - Run `python tools/check_new_suppressions.py src` and `python tools/check_imports.py`.
  - Run `uv run pip-audit`.

## 3. Docs & Artifacts
- [ ] 3.1 Update module docstrings and ownership markers for new helpers.
- [ ] 3.2 Ensure Agent Catalog/nav artifacts remain consistent; regenerate via `make artifacts` if content changes.
- [ ] 3.3 Document observability and refactored structure in developer notes (if applicable).

## 4. Rollout
- [ ] 4.1 Communicate refactor overview to tooling maintainers; document new entry points.
- [ ] 4.2 Monitor CI for lint/type/test regressions; prepare rollback instructions (revert commit) if behavioural drift detected.

