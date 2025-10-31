## 1. Implementation
- [ ] 1.1 Audit every subprocess invocation under `tools/`; wrap via `tools._shared.proc.run_tool`, enforce allow-listed executables, explicit working directories, and timeout/error contracts. Replace blind `except Exception` with typed taxonomy (`ToolExecutionError`, etc.), reduce Ruff complexity (split oversized visitors/functions), add `tools/_shared/metrics.py` with Prometheus counters/histograms and OTEL span helpers wired into `run_tool`, introduce `tools/_shared/settings.py` (typed `pydantic_settings`) that raises `SettingsError` with Problem Details, and initialize module loggers with `NullHandler` (no prints) in `tools/__init__.py`.
- [ ] 1.2 Update import-linter integration: adopt current `importlinter` report API, expose structured violation summaries, add Problem Details envelope and tests. Align Agent Catalog tooling with the latest `kgfoundry.agent_catalog.search` signatures by introducing typed request builders and error handling.
- [ ] 1.3 Introduce TypedDict/dataclass models for docstring caches, docs analytics, navmap data, and CLI envelopes; generate/extend JSON Schemas (2020-12) with version fields and normative examples, add `msgspec` (or `pydantic`) validators, and refactor call sites to remove `Any` usage.
- [ ] 1.4 Provide LibCST typing coverage: add local `stubs/libcst/` declarations or upgrade dependency with `py.typed`, refactor codemods to use typed helpers, and add unit tests ensuring transformations compile and behave as expected.
- [ ] 1.5 Enforce tooling-layer import contracts (domain → adapters → io/cli) via updated `importlinter.cfg` rules and ensure `python tools/check_imports.py` passes.
- [ ] 1.6 Ban unsafe `eval/exec`, replace `yaml.load`/pickle with safe alternatives, add input validation helpers, standardize on `pathlib`, timezone-aware datetimes, and `time.monotonic()` timing, and ensure all public helpers follow PEP 8 naming while private helpers are underscored and `raise ... from e` preserves causes.
- [ ] 1.7 (If applicable) Generate/lint OpenAPI 3.2 specs for tooling HTTP surfaces, referencing Problem Details schemas and automated via `spectral lint`.
- [ ] 1.8 Centralize Problem Details builders for tooling in a shared helper, add `schema/examples/problem_details/tool-exec-timeout.json`, and document metrics/settings toggles plus rollback levers.

## 2. Testing
- [ ] 2.1 Expand pytest suites for subprocess wrappers, import-linter integration, Agent Catalog search, typed payload models, observability (log/metric/span assertions on failure), settings fail-fast behaviour, performance budgets, and codemods. Use `@pytest.mark.parametrize` to cover happy path, edge, and failure/retry cases.
- [ ] 2.2 Add doctest/xdoctest examples for Problem Details payloads, typed model usage, and configuration helpers so published examples execute.
- [ ] 2.3 Verify CLI idempotency by running each updated tool twice (legacy + JSON modes) and asserting convergence with documented retry semantics.
- [ ] 2.4 Add micro-bench tests (or pytest markers) asserting the agreed latency/memory budgets and capturing results for review.

## 3. Docs & Artifacts
- [ ] 3.1 Update developer docs/readmes illustrating secure subprocess usage, typed request builders, observability/metrics configuration, schema validation workflows, and cross-link spec ↔ schema ↔ code anchors (including Agent Portal deep links); regenerate artifacts via `make artifacts`.
- [ ] 3.2 Publish sample Problem Details JSON for subprocess timeouts and Agent Catalog search errors under `schema/examples/`, add `schema/tools/tool_observability.json` (with SemVer notes) alongside refreshed `schema/tools/doc_analytics.json`, and document SemVer notes in schema files.
- [ ] 3.3 Ensure Agent Portal/DocFacts navigation references remain consistent after typed model refactors and link to the updated schemas/examples.
- [ ] 3.4 Document packaging/install instructions (`pip install .[tools]`) and reference new OpenAPI/Schema artifacts.

## 4. Rollout
- [ ] 4.1 Sequence deployment: (a) security/lint/observability fixes, (b) integration parity, (c) typed models & schemas, (d) LibCST stubs & codemod coverage; gate each phase on green quality checks.
- [ ] 4.2 Monitor Prometheus counters/logs/traces for subprocess failures and settings validation post-deploy; document rollback for metrics/settings toggles.
- [ ] 4.3 Run `uv run pip-audit --strict`, perform secret scanning, and capture artifact diffs before release.
- [ ] 4.4 Build wheels (`uv run pip wheel .`) and verify `pip install .[tools]` + smoke tests inside a clean venv; record results for release notes.

