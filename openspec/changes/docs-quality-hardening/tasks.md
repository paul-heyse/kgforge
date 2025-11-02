## 1. Baseline & Governance

- [ ] 1.1 Expand `docs/_scripts/shared.py` so repository discovery, settings, and logging are single-sourced.
  - [ ] 1.1.a Introduce `@lru_cache` wrappers for `detect_environment()` and `load_settings()` to avoid repeated filesystem/env scans during Sphinx builds.
  - [ ] 1.1.b Refine `DocsSettings` to include explicit `docs_build_dir: Path` and `navmap_candidates: tuple[Path, ...]`, mirroring the constants currently embedded in `build_symbol_index.py`.
  - [ ] 1.1.c Add helper factories `make_logger(operation: str, *, artifact: str | None = None) -> logging.LoggerAdapter` that call `with_fields` internally; update scripts to use these adapters instead of constructing `LOG = with_fields(...)` locally.
  - [ ] 1.1.d Move `NAVMAP_CANDIDATES` and similar constants from `build_symbol_index.py` into shared settings and update call sites accordingly.
- [ ] 1.2 Harden optional dependency imports in `docs/conf.py` with typed shims and explicit fallbacks.
  - [ ] 1.2.a Define protocols (e.g., `_AutoDocstringsModule`, `_AstroidBuilderProto`, `_AutoapiParserProto`) in a local `TYPE_CHECKING` block and wrap the existing `importlib.import_module(...)` calls with safe attribute accessors.
  - [ ] 1.2.b Ensure `_apply_auto_docstring_overrides`, `_autoapi_parse_file`, and `_lookup` accept protocol-typed arguments rather than `ModuleType`/`object`, eliminating the current casts in those helpers.
  - [ ] 1.2.c Expand the module docstring to document the new dependency handling and cite the Problem Details example at `schema/examples/problem_details/search-missing-index.json` as the canonical failure envelope.
- [ ] 1.3 Normalize subprocess and git usage across docs scripts using the shared proc utilities.
  - [ ] 1.3.a Replace `_git_sha` filesystem reads in `docs/conf.py` with a call to `shared.resolve_git_sha`, and ensure the function accepts a `WarningLogger` adapter created via section 1.1 helpers.
  - [ ] 1.3.b Update `_symbols_from_git_blob` and `_git_rev_parse` in `symbol_delta.py` to pass `check=True`, `timeout`, and structured metadata to `run_tool`; propagate `ToolExecutionError` problem details to callers.
  - [ ] 1.3.c Ensure the CLI `main()` functions in `build_symbol_index.py` and `symbol_delta.py` exit using structured problem details when validation fails, removing direct `print()`/`sys.exit` patterns.

## 2. Type Safety & Modularity

- [ ] 2.1 Refactor `docs/_scripts/build_symbol_index.py` to eliminate `Any` and centralize transformation logic.
  - [ ] 2.1.a Replace the global `defaultdict` accumulators with dataclasses such as `NavLookup` and `SymbolIndexArtifacts`, ensuring `_index_navmap`, `_record_module_defaults`, and `_record_symbol_meta` return typed instances instead of mutating shared state.
  - [ ] 2.1.b Introduce a `GriffeNode` protocol capturing the attributes accessed across `_canonical_path`, `_doc_first_paragraph`, `_relative_file`, and `_iter_members`; update helper signatures accordingly.
  - [ ] 2.1.c Move GitHub permalink construction into a pure helper (`build_github_permalink(file: Path, span: LineSpan, settings: DocsSettings) -> str | None`) that relies on the shared settings from section 1.1.
  - [ ] 2.1.d Ensure JSON serialization paths (`_write_json`, `_load_navmap`) operate on `Mapping[str, JSONValue]` without casting by routing through a `_coerce_json_value` helper aligned with `symbol_delta.py`.
- [ ] 2.2 Update `docs/_scripts/mkdocs_gen_api.py` so loader interactions and MkDocs writes are explicit and typed.
  - [ ] 2.2.a Define a lightweight `DocumentableNode` protocol exposing `members: Mapping[str, GriffeNode]`, `is_package`, `is_module`, `name`, and `canonical_path`; apply it to `_documentable_members`, `_write_node`, and `_write_index`.
  - [ ] 2.2.b Introduce a `RenderedPage` dataclass capturing destination path and content so `generate_api_reference` can return a list of pages; make MkDocs file writes a thin adapter around this pure result (aid testing).
  - [ ] 2.2.c Replace the implicit `mkdocs_gen_files.open` usage with a context manager helper `write_api_page(page: RenderedPage) -> None`, enabling type-checked IO flows.
- [ ] 2.3 Promote `docs/_scripts/symbol_delta.py` to typed builders and schema-aware coercion.
  - [ ] 2.3.a Replace the `TypedDict` definitions with dataclasses (`SymbolRow`, `ChangeEntry`, `SymbolDeltaPayload`) plus `to_payload()` helpers that return schema-compatible `Mapping[str, JSONValue]`.
  - [ ] 2.3.b Extend `_coerce_symbol_rows` and `_diff_rows` to operate on these dataclasses, removing the residual casts and ensuring list comprehensions produce typed results.
  - [ ] 2.3.c Add a `DeltaResult` dataclass bundling `base_snapshot`, `current_snapshot`, and `changes`, enabling the CLI to pipe structured data into schema validation before writing JSON.

## 3. Data Contracts & Validation

- [ ] 3.1 Define canonical JSON Schemas and examples for docs artifacts.
  - [ ] 3.1.a Reverse-engineer the current `docs/_build/symbols.json` structure by inspecting `SymbolRow` usage in `build_symbol_index.py`; distill required/optional fields, enums (`kind`, `stability`), and nested objects (`source_link`).
  - [ ] 3.1.b Author `schema/docs/symbol-index.schema.json` and `schema/docs/symbol-delta.schema.json`, ensuring they import shared definitions (Problem Details, line spans) from existing `schema/common/**` if available.
  - [ ] 3.1.c Generate sample payloads (`schema/examples/docs/symbol-index.sample.json`, `symbol-delta.sample.json`) via the refactored scripts to serve as golden fixtures.
- [ ] 3.2 Integrate schema validation into writers.
  - [ ] 3.2.a Create a helper `validate_against_schema(payload: Mapping[str, object], schema_path: Path, *, artifact: str) -> None` that leverages `tools._shared.schema.validate_struct_payload`.
  - [ ] 3.2.b Invoke validation in `build_symbol_index.py` immediately before writing `symbols.json` and its secondary indices, surfacing structured errors when validation fails.
  - [ ] 3.2.c Apply the same helper in `symbol_delta.py` for both loaded snapshots and outgoing deltas; ensure CI will fail if the schema drifts.
- [ ] 3.3 Add automation around validation.
  - [ ] 3.3.a Create `docs/_scripts/validate_artifacts.py` that loads `docs/_build/symbols.json`, `docs/_build/symbols.delta.json`, etc., validating them against the schemas and emitting Problem Details on failure.
  - [ ] 3.3.b Update `make artifacts` (or introduce a new phony target) to call the validation script after docs generation, ensuring artifacts cannot drift silently.
  - [ ] 3.3.c Document the validation command in `docs/README` or contributor docs so human operators can run it locally.

## 4. Logging, Errors, and Observability

- [ ] 4.1 Standardize error handling and Problem Details emission.
  - [ ] 4.1.a Audit `docs/conf.py` for remaining `except Exception` blocks (e.g., in directive overrides, `_lookup`) and replace them with focused exception handling that constructs `ToolProblemDetails` entries via `tools._shared.problem_details`.
  - [ ] 4.1.b Ensure CLI entry points catch `ToolExecutionError` (and related subclasses) and render Problem Details JSON to stderr before exiting with non-zero status.
  - [ ] 4.1.c Add regression coverage by invoking the scripts with intentionally malformed inputs (e.g., missing navmap) in a dry-run mode, asserting the logged details contain `instance`, `status`, and remediation hints.
- [ ] 4.2 Enrich logging with correlation metadata and metrics hooks.
  - [ ] 4.2.a Replace direct `LOGGER.warning(...)` calls with `with_fields(LOGGER, operation="symbol_index", artifact="symbols.json")` style adapters (centralized via section 1.1 helpers) so logs always include operation, artifact, and package context.
  - [ ] 4.2.b Integrate `tools._shared.metrics.observe_tool_run` (or equivalent) into long-running scripts, emitting durations and status labels for symbol indexing and delta computation.
  - [ ] 4.2.c Add optional tracing hooks (contextvars-driven correlation IDs) so downstream observability stacks can link docs builds with other automation runs; document how to enable these via environment variables.



