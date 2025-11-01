## 1. Governance & Baseline

- [x] 1.1 Refactor `tools/__init__.py` to import concrete modules directly, define an explicit `__all__`, and add PEP 257-compliant docstrings that reference the exception taxonomy and Problem Details sample payload.
  - [x] 1.1.a Replace statements such as `from tools import codemods as codemods` with explicit absolute imports (`from tools import codemods` or `from . import codemods`) to avoid recursive package initialisation; ensure imports are grouped stdlib/third-party/first-party before populating `__all__`.
  - [x] 1.1.b Assemble a `PUBLIC_EXPORTS` mapping that enumerates re-exported helpers from `tools._shared`, `tools.docs`, `tools.docstring_builder`, and `tools.navmap`; rebuild `__all__` from this mapping so only intentional symbols are exported.
  - [x] 1.1.c Add a module docstring summarising the contract, linking to `schema/examples/tools/problem_details/tool-execution-error.json` (create the example if it does not exist) and `kgfoundry_common.errors` for taxonomy references.
  - [x] 1.1.d Ensure every newly re-exported symbol has an inline `typing.Annotated` or descriptive alias when necessary (e.g., `ToolExecutionError = _proc.ToolExecutionError`) so Sphinx/autoapi renders accurate documentation.
  - [x] 1.1.e Update `stubs/tools/__init__.pyi` to mirror the curated `__all__`, using precise types from the source modules instead of `Any`.
- [x] 1.2 Apply the same restructuring to `tools/docs/__init__.py`, ensuring only supported entry points are exported and documented.
  - [x] 1.2.a Switch to explicit imports (`from tools.docs import build_graphs`) and build a sorted `__all__` list that matches runtime exports exactly.
  - [x] 1.2.b Add PEP 257 docstrings that describe each exported builder (catalog, portal renderer, analytics) and reference the Problem Details example for failure envelopes.
  - [x] 1.2.c Update `stubs/tools/docs/__init__.pyi` to expose the same symbol set with concrete return types (e.g., `def build_agent_catalog(...) -> CatalogBuildResult`).
  - [x] 1.2.d Validate that consumers under `tests/tools` and `tools/navmap` import only documented names; adjust call sites if new module boundaries require relocation.
- [x] 1.3 Extract domain-layer orchestrators from oversized adapters such as `tools/docstring_builder/cli.py`, keeping CLI modules thin shims that delegate to pure orchestration modules.
  - [x] 1.3.a Create a new `tools/docstring_builder/orchestrator.py` module that owns `_execute_pipeline`, `_run`, `_process_file`, docfacts reconciliation, and baseline/git helpers; define typed request/response dataclasses (`DocstringBuildRequest`, `DocstringBuildResult`).
  - [x] 1.3.b Move input normalisation helpers (`_normalize_input_path`, `_resolve_ignore_patterns`, `_select_files`) into a focused utility module (e.g., `tools/docstring_builder/io.py`) so they can be reused by automation without importing argparse.
  - [x] 1.3.c Rewrite `tools/docstring_builder/cli.py` to build the argument parser, map subcommands to orchestrator entry points, and format CLI envelopes only—no direct filesystem or subprocess calls.
  - [x] 1.3.d Update module-level constants (e.g., `CACHE_PATH`, `DOCFACTS_PATH`) to live in a dedicated `tools/docstring_builder/paths.py` so orchestrator tests can substitute paths without patching globals.
  - [x] 1.3.e Ensure CLI adapters return `ExitStatus` values produced by orchestrator results; refactor JSON output/Problem Details rendering into a helper (`render_cli_result`) housed alongside the orchestrator.
- [x] 1.4 Update `tools/make_importlinter.py` contracts so adapter layers cannot import back into domain or shared internals, enforcing the new layering.
  - [x] 1.4.a Extend `_build_template` to emit explicit layered contracts for `tools.docstring_builder` (`adapters > orchestrator > domain > shared`), `tools.docs`, and `tools.navmap` packages.
  - [x] 1.4.b Modify `main()` so it can write multiple contracts (one per package) into `.importlinter`, including sections that forbid adapter modules from importing `_shared` internals directly.
  - [x] 1.4.c Add a thin wrapper under `tools/lint/` or `Makefile` to run `python tools/make_importlinter.py --check`, ensuring CI enforces the new contracts.
  - [x] 1.4.d Update `stubs/tools/make_importlinter.pyi` (if present) or create one so mypy recognises the new function signatures.

## 2. Type Safety & Data Contracts

- [ ] 2.1 Introduce or extend typed domain models (msgspec structs, dataclasses, or PEP 695 generics) for CLI envelopes, docstring edits, navmap documents, and analytics payloads, replacing `dict[str, Any]` surfaces.
  - [ ] 2.1.a Expand `tools/docstring_builder/models.py` with value objects for harvested symbols, IR edits, CLI results, and cache summaries; ensure each struct exposes `model_dump()` helpers for JSON schema generation.
  - [ ] 2.1.b Promote the dataclasses in `tools/docs/build_agent_catalog.py` (`SymbolRecord`, `ModuleRecord`, etc.) into a new `tools/docs/catalog_models.py` module built on msgspec structs or frozen dataclasses with explicit field types.
  - [ ] 2.1.c Replace ad-hoc dict usage in `tools/navmap/repair_navmaps.py`, `tools/navmap/build_navmap.py`, and `tools/navmap/strip_navmap_sections.py` with typed models from `tools/navmap/document_models.py` and newly introduced patch/edit structs.
  - [ ] 2.1.d Ensure analytics payloads use `AgentAnalyticsDocument` exclusively; remove `dict[str, Any]` handling in `tools/docs/build_agent_analytics.py` by returning typed objects.
- [ ] 2.2 Replace legacy payload construction sites to use the new models, ensuring conversions handle legacy inputs while eliminating redundant `# type: ignore` usage.
  - [ ] 2.2.a Introduce migration helpers (`from_legacy_docfacts`, `from_legacy_navmap`) that accept existing dict payloads, validate them, and return typed structs; call these helpers wherever legacy input is expected.
  - [ ] 2.2.b Update CLI envelopes (`tools/docstring_builder/cli.py`, `tools/navmap/repair_navmaps.py`) to construct responses via `CliEnvelopeBuilder` and new typed models, removing casts and inlined JSON dumps.
  - [ ] 2.2.c Delete unused `# type: ignore` pragmas uncovered during refactors and ensure msgspec struct instantiation is fully typed.
  - [ ] 2.2.d Update manifest writers (`tools/docstring_builder/cache`, `tools/docs/build_agent_catalog`) to serialise typed models through a shared helper instead of `json.dumps(payload)`, guaranteeing schema compliance.
- [ ] 2.3 Enhance `tools/_shared/schema.py` so models can emit JSON Schema 2020‑12 definitions, wiring generation hooks where schemas need to stay in sync with code.
  - [ ] 2.3.a Add functions such as `render_schema(model: type[Struct], name: str) -> dict[str, object]` and `write_schema(model: type[Struct], destination: Path)` leveraging msgspec or pydantic schema generation.
  - [ ] 2.3.b Update schema-producing commands (`tools/docstring_builder/cli`, `tools/docs/export_schemas.py`) to delegate schema dumps to the shared helper, keeping IDs/versions consistent.
  - [ ] 2.3.c Store generated schema metadata (version, checksum) alongside the files so future changes can detect drift automatically.
  - [ ] 2.3.d Add convenience validators (`validate_struct_payload(payload: Mapping[str, object], model: type[Struct])`) to reduce repeated casting logic across modules.
- [ ] 2.4 Provide typed facades or local stubs under `stubs/tools/**` for third-party integrations (`pydot`, `click`, `libcst`, etc.), removing `Any` from decorators and visitors.
  - [ ] 2.4.a Author `stubs/pydot/__init__.pyi` that covers the subset used by `tools/docs/build_graphs.py` (Graph, Node, Edge, write functions) and update the module to annotate return types accordingly.
  - [ ] 2.4.b Extend existing `stubs/libcst` to include builder utilities referenced by `tools/codemods` and `tools/docstring_builder/apply.py`, eliminating casts in visitors.
  - [ ] 2.4.c Add thin wrappers in runtime code (e.g., `tools/docs/graph_io.py`) where third-party APIs are particularly dynamic, exposing typed Protocols for mypy/pyrefly.
  - [ ] 2.4.d Ensure the stub packages ship with `py.typed` markers and are referenced from `pyproject.toml` so downstream installs remain type-safe.
- [ ] 2.5 Normalize time and path handling across tooling modules to use `pathlib.Path`, timezone-aware `datetime`, and `decimal.Decimal` where precision matters.
  - [ ] 2.5.a Replace naive timestamp generation in `tools/docs/build_agent_catalog.py` and `tools/docstring_builder/cli.py` with helpers that call `datetime.now(tz=UTC)`.
  - [ ] 2.5.b Audit all path manipulations (especially `_normalize_input_path`, `build_agent_catalog` file walkers, navmap repair utilities) to use `Path` joins rather than string concatenation, adding allowlist enforcement where necessary.
  - [ ] 2.5.c Where analytics or metrics handle ratios/percentages, adopt `Decimal` to avoid floating-point drift; extend typed models accordingly.
  - [ ] 2.5.d Update schema and manifest writers to serialise `Path` and `Decimal` types via canonical string formats.

## 3. Error Handling, Logging, Observability, Idempotency

- [ ] 3.1 Harden `tools._shared.proc.run_tool` to route all subprocess execution through a single wrapper that enforces timeouts, sanitizes environments, and raises typed exceptions.
  - [ ] 3.1.a Introduce a `contextvars.ContextVar[str]` (e.g., `TOOL_OPERATION_ID`) in a new `tools._shared.context` module; initialise it in `run_tool` when missing.
  - [ ] 3.1.b Refactor environment sanitisation to accept allowlists/denylists injected via `ToolRuntimeSettings`, ensuring secrets are excluded by default.
  - [ ] 3.1.c Guarantee all exceptions raised within `run_tool` use `raise ... from exc` and attach Problem Details instances including operation ID, cwd, and command.
  - [ ] 3.1.d Expose a streaming interface (generator or callback) for stdout/stderr if required by adapters, while keeping the synchronous API backward compatible.
- [ ] 3.2 Thread correlation data via `contextvars` so `StructuredLoggerAdapter` instances, metrics, and traces include operation identifiers automatically.
  - [ ] 3.2.a Update `tools._shared.logging.with_fields` to automatically inject the operation ID from the context var when present.
  - [ ] 3.2.b Extend `Tools._shared.metrics.observe_tool_run` to accept an optional correlation ID argument and default to the context variable value; propagate it into metric labels and OpenTelemetry span attributes.
  - [ ] 3.2.c Add helpers (`push_operation_context`, `pop_operation_context`) and use them in orchestrators (docstring builder, navmap, docs) before invoking subprocesses.
- [ ] 3.3 Extend `tools._shared.metrics` to expose Prometheus counters/histograms and OpenTelemetry spans through injectable factories, ensuring all tool run outcomes emit consistent telemetry.
  - [ ] 3.3.a Parameterise metric factories so tests can supply stub implementations; default to lazy-loading `prometheus_client`/OpenTelemetry at runtime.
  - [ ] 3.3.b Emit additional labels (operation, correlation ID, retry count) and ensure span status is set correctly on failure.
  - [ ] 3.3.c Record observations in orchestrators (docstring builder, navmap repair, docs builders) through a shared helper (`record_tool_result`) to reduce copy/paste logic.
- [ ] 3.4 Standardize exception taxonomy across tooling (e.g., `ToolExecutionError`, `CatalogBuildError`) so every failure emits RFC 9457 Problem Details payloads and retry guidance.
  - [ ] 3.4.a Create `tools/_shared/exceptions.py` housing base classes (`ToolingError`, `ToolExecutionError`, `CatalogBuildError`, `NavmapError`) with constructors that accept Problem Details and cause metadata.
  - [ ] 3.4.b Update modules currently defining bespoke exceptions (`tools/docs/errors.py`, docstring builder models) to inherit from the shared base and populate Problem Details consistently.
  - [ ] 3.4.c Provide serializer helpers so CLI envelopes can embed Problem Details without duplicating typing logic.
  - [ ] 3.4.d Replace direct `ProblemDetailsDict` assembly scattered across modules with calls to `tools._shared.problem_details.build_tool_problem_details` or new specialised builders.
- [ ] 3.5 Implement retry-aware wrappers for idempotent tooling operations, guaranteeing repeated invocations converge without unintended side effects.
  - [ ] 3.5.a Add `run_tool_with_retry` to `tools._shared.proc`, supporting configurable retry counts/backoff and raising the same typed exceptions upon exhaustion.
  - [ ] 3.5.b Audit docstring builder, navmap repair, and docs generation entry points to identify idempotent operations; wrap those subprocess invocations with the retry helper.
  - [ ] 3.5.c Record retry attempts in structured logs/metrics to aid SRE dashboards.
  - [ ] 3.5.d Document retry eligibility within each orchestrator to avoid accidental retries of non-idempotent commands (e.g., `git commit`).

## 4. Security, Configuration, & Supply Chain

- [ ] 4.1 Promote `tools._shared.settings` to a canonical `pydantic_settings` hub that hydrates environment variables, validates eagerly, and surfaces structured errors for missing or malformed configuration.
  - [ ] 4.1.a Extend `ToolRuntimeSettings` with fields for metrics/tracing endpoints, default working directories, and subprocess allowlists; ensure validators coerce comma-separated lists into tuples.
  - [ ] 4.1.b Provide a `get_settings(namespace: str)` helper that caches per-namespace settings (e.g., `docbuilder`, `docs`, `navmap`) derived from environment variable prefixes.
  - [ ] 4.1.c Update orchestrators and adapters to request settings through the helper instead of reading environment variables directly.
  - [ ] 4.1.d Emit structured `SettingsError` Problem Details that include the namespace, missing fields, and remediation guidance.
- [ ] 4.2 Audit all file/URL inputs handled by tooling, enforcing allowlists and `resolve_path` guards to prevent traversal or injection vectors.
  - [ ] 4.2.a Harden `_normalize_input_path` in `tools/docstring_builder/cli.py` (or its relocated utility module) to require paths within the repository root and reject `..` segments.
  - [ ] 4.2.b Apply similar guards to navmap repair inputs, docs builders (which read configuration files), and `tools/lint/add_gpu_header.py` (which edits files under `tests/`).
  - [ ] 4.2.c Introduce central helpers (`require_workspace_file`, `require_workspace_directory`) in `tools._shared.validation` and replace local checks with these helpers.
  - [ ] 4.2.d Sanitize any URL inputs (e.g., link policies in `build_agent_catalog.py`) by parsing via `urllib.parse` and validating allowed schemes (`https`, `vscode-remote`).
- [ ] 4.3 Replace any unsafe YAML/JSON loading or dependency handling with safe alternatives, and keep dependency gating (`uv run pip-audit`, secret scanning) wired into the tooling workflow.
  - [ ] 4.3.a Ensure all YAML loads use `yaml.safe_load` (already true in most modules) and centralise the helper in `tools._shared.serialization` with schema validation hooks.
  - [ ] 4.3.b Add a `tools._shared.security` module exposing `run_pip_audit()` and `scan_for_secrets()` wrappers that call `run_tool` with hardened settings; invoke these wrappers from packaging or release scripts.
  - [ ] 4.3.c Update build scripts (`tools/docs/build_artifacts.py`, `tools/validate_gallery.py`) to validate external JSON against schemas using the enhanced `validate_struct_payload` helper.
  - [ ] 4.3.d Document required security commands in module docstrings (e.g., `tools/lint/__init__.py`) so automation can rely on them without reading ancillary docs.

## 5. Verification Loop

- [ ] 5.1 After completing each tranche of refactors above, run `uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini` and ensure that all code blocks you made edits too are error free.

