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
- [ ] 1.1 Refactor `_shared/logging.py` to fix pyrefly override issues, add structured adapter, document Problem Details logging contract.
- [ ] 1.2 Harden `_shared/proc.py`: define `JsonValue` type alias, enforce absolute executables, capture stdout/stderr as `tuple[str, str]`, emit Problem Details on failure.
- [ ] 1.3 Introduce `_shared/problem_details.py` with builders/helpers plus example at `docs/examples/tools_problem_details.json`.
- [ ] 1.4 Add pytest coverage for logging/proc/problem_details behavior (`tests/tools/shared/test_logging.py`, `test_proc.py`).

## 2. Docstring Builder Integration
- [ ] 2.1 Update `normalizer.py`, `normalizer_signature.py`, `normalizer_annotations.py`, `policy.py`, `render.py`, and `docfacts.py` to consume typed models and eliminate `Any` usage.
- [ ] 2.2 Fix pyrefly/mypy issues in observability module: ensure stub metrics implement `.labels()` and return `self`, guard imports.
- [ ] 2.3 Align CLI (`cli.py`) with typed adapters, Problem Details emission, schema validation, and metrics usage.
- [ ] 2.4 Finalize plugin Protocols (`plugins/base.py`, `plugins/__init__.py`), compatibility shim, and update bundled plugins.
- [ ] 2.5 Add regression tests: schema validation (`tests/tools/docstring_builder/test_schemas.py`), plugin behavior (`test_plugins.py`), CLI integration (`test_cli.py`).

## 3. Documentation Pipelines
- [ ] 3.1 Introduce typed models for analytics, graphs, and test maps; refactor `build_agent_catalog.py`, `build_graphs.py`, `build_test_map.py`, `export_schemas.py`, `render_agent_portal.py` to use them.
- [ ] 3.2 Replace blind exceptions with typed `DocumentationBuildError` hierarchy; ensure `run_tool` wrapper is used for subprocess calls.
- [ ] 3.3 Add JSON Schemas (`schema/tools/doc_analytics.json`, `doc_graph_manifest.json`, `doc_test_map.json`), fixtures, and validation tests.
- [ ] 3.4 Update docs generator tests (`tests/tools/docs/test_*`) with table-driven cases covering success, validation failure, and subprocess errors.

## 4. Navmap & Ancillary CLIs
- [ ] 4.1 Create `tools/navmap/models.py` and adapters for navmap documents; refactor builders/checkers/repair scripts to use typed models.
- [ ] 4.2 Replace prints/import-at-runtime with structured logging and top-level imports; integrate `run_tool` wrapper.
- [ ] 4.3 Define `schema/tools/navmap_document.json` plus pytest coverage for migration/repair outputs.
- [ ] 4.4 Harden other CLIs (`detect_pkg.py`, `generate_docstrings.py`, `hooks/docformatter.py`, lint helpers) with typed `main()` functions, structured logging, and safe subprocess usage.

## 5. Observability & Testing
- [ ] 5.1 Ensure all modules expose `get_logger(__name__)` and register Prometheus metrics via typed providers; add OpenTelemetry span hooks where applicable.
- [ ] 5.2 Add doctests/xdoctests referencing Problem Details samples and schema usage.
- [ ] 5.3 Expand pytest suite (`tests/tools/`) with parametrized cases for edge inputs, failure modes, and retry/idempotency checks.
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
  - `make artifacts && git diff --exit-code`
  - `openspec validate tools-hardening --strict`
- [ ] 7.2 Attach command outputs + schema validation logs to execution report.
- [ ] 7.3 Flip `DOCSTRINGS_TYPED_IR` default once metrics show stability; document completion in rollout note.

