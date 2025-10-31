## Why
The `tools/` suite powers DocFacts generation, docs/navmap builds, gallery validation, and lint automation. Today these scripts rely on dynamic dictionaries, unstructured subprocess calls, blind exception handling, and ad-hoc logging. Ruff, mypy, and pyrefly collectively surface >1.8k violations, preventing strict quality gates and risking silent corruption or insecure execution. We need typed contracts, schema validation, structured observability, and secure subprocess orchestration to keep tooling outputs trustworthy as more automation agents (and AI executors) rely on them.

## What Changes
- [x] **ADDED**: Shared typed infrastructure under `tools/_shared/` for logging, subprocess management, Problem Details envelopes, and schema helpers.
- [x] **ADDED**: JSON Schemas for CLI/doc outputs (`schema/tools/docstring_builder_cli.json`, `schema/tools/doc_analytics.json`, `schema/tools/navmap_document.json`, etc.) with version constants and validation utilities.
- [x] **MODIFIED**: Docstring builder pipeline to adopt `tools/docstring_builder/models.py` dataclasses/TypedDicts, refactored modules (`normalizer_*`, `policy`, `render`), plugin Protocols, CLI JSON emission, and Prometheus-compatible observability.
- [x] **MODIFIED**: Documentation pipelines (`tools/docs/build_agent_catalog.py`, `build_graphs.py`, `build_test_map.py`, `export_schemas.py`, `render_agent_portal.py`) to use typed adapters, secure subprocess wrappers, structured logging, and targeted exception taxonomy.
- [x] **MODIFIED**: Navmap utilities and CLI scripts to replace prints with structured logs, adopt typed models, validate outputs against schemas, and expose Problem Details on failure.
- [ ] **REMOVED**: Legacy compatibility shims and blind exception handlers once typed pipeline is default (tracked in rollout plan).
- [ ] **RENAMED**: _None._
- [ ] **BREAKING**: Public CLI surface remains compatible; machine-readable JSON now wrapped in a versioned envelope (flagged in docs / changelog).

## Impact
- **Affected specs:** `docstring-tooling`, `docs-builders`, `navmap-suite`, `tools-observability` (see `specs/tools-suite/spec.md`).
- **Affected code paths:**
  - `tools/_shared/logging.py`, `tools/_shared/proc.py`, `tools/_shared/problem_details.py`
  - `tools/docstring_builder/` modules (normalizer, policy, render, cli, plugins, observability)
  - `tools/docs/` generators (`build_agent_catalog.py`, `build_graphs.py`, `build_test_map.py`, `render_agent_portal.py`, `export_schemas.py`, etc.)
  - `tools/navmap/` scripts and supporting CLIs
  - JSON Schema additions under `schema/tools/`
- **Rollout:** Feature flag `DOCSTRINGS_TYPED_IR` governs docstring pipeline migration; CLI wrappers gain `--legacy-json` for one release cycle. Observability metrics deployed alongside to monitor failures. Compatibility shims logged and scheduled for removal after metrics show stabilization.
- **Risks:** Widespread refactor across tooling; mitigated via phased tasks, schema validation, regression suites, and Prometheus counters/structured logs capturing failures. Potential third-party plugin breakage addressed via compatibility shim and documentation updates.

## Acceptance
- All quality gates pass (`ruff`, `pyrefly`, `mypy`, `pytest`, `make artifacts`, `openspec validate`).
- CLI/JSON outputs validate against new schemas with round-trip tests and published examples.
- Structured logging replaces every `print`; subprocess invocations go through `_shared.proc.run_tool` or documented safe wrappers.
- Docstring builder, docs generators, and navmap utilities emit Problem Details envelopes on failure and expose Prometheus metrics.
- Compatibility flag telemetry shows zero regressions before flipping the typed pipeline to default.

