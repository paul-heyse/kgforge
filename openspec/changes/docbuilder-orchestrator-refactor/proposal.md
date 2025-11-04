## Why
The docstring builder orchestrator has grown into a 1,300+ line module that violates Ruff complexity thresholds across `_run_pipeline`, `_process_file`, `_handle_docfacts`, and `_print_failure_summary`. The single function orchestrates caching, plugin wiring, policy evaluation, manifest writing, diff generation, metrics, and CLI payload assembly, yielding brittle branching, duplicated state maps, and unclear responsibilities. Without structural refactoring, lint gates stay red, readability suffers, and future enhancements (typed pipelines, policy enforcement, observability) remain risky.

## What Changes
- [ ] **ADDED**: Capability spec for modular docstring builder orchestration covering pipeline separation, docfacts coordination, observability, and CLI reporting.
- [ ] **MODIFIED**: `tools/docstring_builder/orchestrator.py` split into composable units (`PipelineRunner`, `FileProcessor`, `DocfactsCoordinator`, `DiffManager`, `FailureSummaryRenderer`, etc.) with typed value objects.
- [ ] **ADDED**: Supporting modules (e.g. `pipeline.py`, `file_processor.py`, `docfacts.py`) to encapsulate responsibilities, each with PEP 257 docstrings and typed APIs.
- [ ] **ADDED**: Regression and unit tests exercising new helpers (policy reporting, docfacts reconciliation, CLI payload assembly) including table‑driven coverage for edge cases.
- [ ] **ADDED**: Observability alignment (structured logs, metrics hooks) captured via focused tests and documented examples.
- [ ] **MODIFIED**: Existing manifests/diff writers to integrate with new helpers without behavioural drift.

## Impact
- **Specs**: New `docbuilder/orchestrator` capability describing modular orchestration, docfacts reconciliation, observability, and CLI reporting guarantees.
- **Code**: `tools/docstring_builder/orchestrator.py`, new helper modules under `tools/docstring_builder/` for pipeline, docfacts, diff, failure summary, metrics, manifest, and policy reporting.
- **Tests**: New suites under `tests/tools/docstring_builder/` for pipeline runner, file processor, docfacts coordinator, failure summary renderer, and diff manager.
- **Docs**: Updated module docstrings/ownership markers; potential Agent Catalog/Nav artifacts regenerated.
- **Quality Gates**:
  - Ruff: `uv run ruff format && uv run ruff check --fix`
  - Types: `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`
  - Tests: `uv run pytest -q`
  - Docs/Artifacts: `make artifacts && git diff --exit-code`
  - Suppression guard: `python tools/check_new_suppressions.py src`
  - Architecture: `python tools/check_imports.py`
  - Security: `uv run pip-audit`

## Out of Scope
- Altering CLI flags or changing builder output formats.
- Modifying docfacts schema contents beyond wiring.
- Refactoring unrelated tooling modules or search/vectorstore code.
- Introducing async execution or significant behaviour changes to plugin interfaces.

## Risks / Mitigations
- **Risk**: Behavioural drift in docfacts/manifest outputs.  
  **Mitigation**: Regression tests comparing old vs new payloads; use fixtures capturing representative runs.
- **Risk**: Import boundary violations when adding modules.  
  **Mitigation**: Validate with `python tools/check_imports.py`; keep helpers in tooling package.
- **Risk**: Increased complexity in dependency injection.  
  **Mitigation**: Provide typed value objects and builders; document usage in design.
- **Risk**: Test fragility around filesystem artifacts.  
  **Mitigation**: Use `tmp_path` fixtures and safe JSON helpers consistent with Agent Operating Protocol.

## Alternatives Considered
- Suppressing Ruff complexity errors — rejected because it ignores maintainability goals.
- Partial refactor limited to `_run_pipeline` — rejected; systemic improvements require modular boundaries across cache/docfacts/diff/reporting.
- Rewriting orchestration as CLI-only script — rejected; tooling library must remain importable and testable.

