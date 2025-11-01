## Context
Despite the tools suite passing Ruff and partial pyrefly checks, structural gaps remain that block full type cleanliness, observability guarantees, and junior-friendly maintenance.

1. **Exports & documentation drift**: `tools/__init__.py` recursively imports subpackages, masking real dependencies and making the public API unclear. Many exported helpers lack PEP 257 docstrings linking back to the exception taxonomy and Problem Details examples, leaving new contributors guessing how to use them.
2. **Monolithic adapters**: Modules such as `tools/docstring_builder/cli.py` (~1.7k LOC) and `tools/docs/build_agent_catalog.py` blend CLI parsing, filesystem IO, and domain orchestration. This creates long functions that violate Ruff complexity rules and frustrate testing; failures in navmap/docstring flows are hard to isolate.
3. **Typed domain models**: Phase 3 introduced msgspec structs, yet significant surfaces (CLI envelopes, navmap repair documents, analytics payloads) still expose `dict[str, Any]`. Time handling often relies on naive datetimes or string formatting; path handling uses `os.path`. Schemas are not always regenerated from code, creating drift.
4. **Observability & idempotency**: `tools._shared.proc.run_tool` uses observability helpers but lacks correlation IDs, retry semantics, or enforced structured payloads for failure cases. Metrics and traces are not thoroughly tested, so regressions go unnoticed.
5. **Configuration & supply chain**: Although a `ToolRuntimeSettings` class exists, documentation of required env vars is sparse. Some tooling commands read arbitrary paths/URLs without allowlists. Security tooling (pip-audit, secret scan) is not embedded in the workflow documentation, making regressions likely.

## Goals
- Publish an explicit, documented public API for the tools package with curated exports and Problem Details examples.
- Restructure oversized adapters into layered modules separating CLI concerns from domain orchestration, enforced by import-linter.
- Ensure all tooling payloads (CLI envelopes, navmap docs, analytics, docstring edits) use typed domain models with schema round-trips, timezone-aware datetimes, and pathlib usage.
- Harden subprocess execution with correlation IDs, tested metrics/traces, retry helpers, and consistent Problem Details.
- Strengthen configuration hygiene, path/url validation, and dependency scanning, and document the required workflow for contributors.

## Non-Goals
- Changing business behaviour of tooling commands (outputs should remain semantically equivalent aside from structured metadata).
- Replacing msgspec or Prometheus/OpenTelemetry libraries.
- Refactoring external services (Agent Catalog, Observability dashboards) beyond adjusting imports.
- Introducing a new orchestration backend; the focus is on structure and safety.

## Detailed Plan

### 1. Govern public exports

#### Implementation steps
1. **Collapse recursive imports inside `tools/__init__.py`**: replace `from tools import codemods as codemods`-style statements with explicit imports (`from . import codemods`, `from tools._shared import cli as cli_shared`). Build a `PUBLIC_EXPORTS` dictionary that maps exported names to their canonical definitions and derive `__all__` from that mapping.
2. **Document the public contract**: add a concise module docstring describing the tooling surface, referencing the shared exception taxonomy (`kgfoundry_common.errors`) and the Problem Details example (`schema/examples/tools/problem_details/tool-execution-error.json`). For aliases that improve readability (e.g., exposing `_shared.proc.ToolExecutionError`), provide inline docstrings that explain intent.
3. **Curate `tools/docs/__init__.py`**: import explicit entry points (`build_agent_catalog`, `render_agent_portal`, `gen_readmes`) and rebuild `__all__` to match runtime exports. Document the module so downstream users understand which functions are stable and where Problem Details are emitted on failure.
4. **Synchronise stubs**: regenerate `stubs/tools/__init__.pyi` and `stubs/tools/docs/__init__.pyi` with the same export sets and accurate signatures (no `Any`). Ensure each stub module includes `py.typed`.
5. **Verification**: run `uv run pyrefly check tools/__init__.py stubs/tools` and `uv run mypy --config-file mypy.ini tools stubs/tools` to confirm imports resolve without `Any` leakage.

### 2. Layer docstring builder & docs tooling

#### Implementation steps
1. **Create orchestrator modules**: add `tools/docstring_builder/orchestrator.py` containing pipeline coordination (`_execute_pipeline`, `_run`, `_process_file`, docfacts reconciliation). Introduce typed dataclasses like `DocstringBuildRequest`, `FileOutcome`, and `DocfactsOutcome` so orchestrators return structured results.
2. **Refactor CLI adapters**: rewrite `tools/docstring_builder/cli.py` to perform only argument parsing, command dispatch, and CLI envelope rendering. Move filesystem constants into `tools/docstring_builder/paths.py` and path helpers into `tools/docstring_builder/io.py` so orchestrators and adapters share them without circular imports.
3. **Apply the pattern to docs tooling**: extract pure computation from `tools/docs/build_agent_catalog.py` into `tools/docs/catalog_orchestrator.py`, covering tasks like symbol harvesting, analytics aggregation, and diff generation. Keep the original module as a thin adapter handling CLI arguments and persistence.
4. **Regenerate layering contracts**: update `tools/make_importlinter.py` to output contracts for `tools.docstring_builder`, `tools.docs`, and `tools.navmap` (e.g., `cli -> adapters -> orchestrator -> domain -> _shared`). Ensure the script supports `--check` mode and add it to the verification loop.
5. **Adjust module initialisers**: review `__all__` exports in the newly split modules so orchestrators expose typed entry points while adapters stay lightweight. Update import sites accordingly.

### 3. Typed domain models & schemas

#### Implementation steps
1. **Consolidate model definitions**: expand `tools/docstring_builder/models.py` with msgspec structs covering CLI envelopes, docfacts documents, cache entries, and Problem Details wrappers. Introduce `tools/docs/catalog_models.py` for catalog packages/modules/symbols and ensure navmap code relies on `tools/navmap/document_models.py` for typed documents.
2. **Adopt models at integration points**: replace dictionary payloads in adapters (`tools/docstring_builder/cli.py`, `tools/docs/build_agent_catalog.py`, `tools/navmap/repair_navmaps.py`, `tools/docs/render_agent_portal.py`) with the new structs. Where legacy JSON must be supported, add conversion helpers (`from_legacy_docfacts`, `navmap_document_from_payload`) that validate and upgrade inputs.
3. **Enhance schema generation**: update `tools/_shared/schema.py` with utilities like `render_schema(model)` and `write_schema(model, destination)` that leverage msgspec's schema generation. Ensure schema IDs and versions are kept in sync via a shared metadata table.
4. **Centralise validation**: implement `validate_struct_payload(payload, model)` that converts payloads to typed structs and runs `kgfoundry_common.serialization.validate_payload`. Replace bespoke validation logic across docs/navmap/docstring modules with this helper.
5. **Normalise temporal/path precision**: require all new models to use timezone-aware `datetime` (UTC) and `Path` objects; convert to strings only at serialization boundaries. For ratios or monetary values, adopt `Decimal` to avoid float drift and update schema definitions accordingly.

### 4. Harden subprocess execution & observability

#### Implementation steps
1. **Propagate operation context**: add a `ContextVar` (e.g., `TOOL_OPERATION_ID`) in `tools._shared.context` with helpers (`operation_context`, `set_operation_id`). Wrap orchestrator entry points (docbuilder generate/fix, catalog build, navmap repair) in `operation_context` to ensure consistent correlation IDs.
2. **Refine `run_tool`**: update `tools._shared.proc.run_tool` to pull the correlation ID from the context, enrich logs/Problem Details, and raise typed exceptions using `raise ... from exc`. Allow optional streaming/callback hooks while maintaining backwards compatibility with existing return values.
3. **Upgrade telemetry**: modify `tools._shared.metrics.observe_tool_run` to accept injectable counter/histogram/span factories, emit operation/correlation/retry labels, and set OpenTelemetry span status on failures. Provide a shared helper (`record_tool_result`) for orchestrators to log outcome, record metrics, and finish spans consistently.
4. **Introduce retry helpers**: implement `run_tool_with_retry` supporting configurable attempts and backoff. Document which commands are safe to retry and update orchestrators to use the helper where idempotent operations occur (e.g., navmap diff generation).
5. **Standardise exceptions**: move core exception classes into `tools/_shared/exceptions.py` (e.g., `ToolingError`, `ToolExecutionError`, `CatalogBuildError`, `NavmapError`). Update modules to inherit from these classes and build Problem Details via shared factories so every failure path emits RFC 9457-compliant payloads.

### 5. Secure configuration & supply chain

#### Implementation steps
1. **Namespace-aware settings**: extend `tools._shared.settings` with dedicated classes (`DocbuilderSettings`, `DocsSettings`, `NavmapSettings`) inheriting from `BaseSettings`. Each class should define required environment variables (exec allowlists, cache directories, metrics/tracing endpoints) and expose a `get_settings(namespace: str)` helper that caches validated instances.
2. **Centralise path/url validation**: enrich `tools._shared.validation` with helpers (`require_workspace_file`, `require_workspace_directory`, `validate_allowed_url`) and replace ad-hoc path checks in docstring builder, docs builders, navmap repair, and lint utilities. Reject traversal (`..`), `file://` schemes, and off-repo paths with structured `ValidationError` Problem Details.
3. **Secure parsing & dependency gating**: introduce `tools._shared.serialization.safe_load_json/yaml` wrappers that validate inputs against typed models. Add `tools._shared.security` exposing `run_pip_audit()`/`run_secret_scan()` wrappers around the hardened subprocess helper and integrate them into packaging/build flows.
4. **Pin optional dependencies**: revisit `pyproject.toml` extras to ensure observability/docs/navmap options install the necessary libraries. Update orchestrators to surface clear errors when optional dependencies are missing and ensure the new settings layer controls fallbacks.

## Data Contracts & Validation Strategy
- Maintain JSON Schemas under `schema/tools/**`, generated or synchronized via the enhanced schema helper. Each schema must specify `$id`, `$schema` (2020-12), and version notes in comments/docs.
- Provide normative examples in `schema/examples/tools/**` matching the new models. For Problem Details, include at least one sample enumerating extensions (correlation ID, retry hint).
- Regression tests must validate current and legacy payloads, using parametrized pytest cases for invalid inputs (missing field, wrong type, naive datetime, disallowed path).

## Testing Strategy
1. **Unit tests**: new orchestrators, schema converters, and retry helpers must have direct unit coverage.
2. **Property/round-trip tests**: For each typed model, implement parametrized tests verifying encode/decode and schema validation.
3. **Integration tests**: CLI adapters tested via `pytest` using temporary directories; verify logs/Problem Details.
4. **Type checks**: Run `uv run pyrefly check` and `uv run mypy --config-file mypy.ini` against `tools`, `stubs/tools`, and `tests/tools`.
5. **Import-linter**: Run `python tools/make_importlinter.py --check` as part of the verification loop.

## Rollout & Verification
- Implement subsections sequentially (exports → layering → typing → observability → security) to keep PR size manageable.
- After each subsection, run the verification loop: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`, `uv run pytest -q`, `make artifacts && git diff --exit-code`, `python tools/check_new_suppressions.py src`, `python tools/check_imports.py`, `uv run pip-audit`.
- Record before/after metrics (Prometheus counters/histograms) and type-check error counts in the PR summary.
- Communicate any breaking API changes via SemVer bumps and deprecation notes in README/docs.

