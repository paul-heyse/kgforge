# Migration Scope: `kgfoundry codeintel indexer` Typer CLI (`codeintel/indexer/cli.py`)

This document enumerates the concrete tasks required to migrate the Tree-sitter code-intel CLI onto the shared tooling stack. The outcome is a CLI whose configuration, metadata, envelopes, Problem Details, and downstream documentation hooks are aligned with the standards captured in:

- `openspec/changes/tools-cli-openapi-alignment/`
- `openspec/changes/tools-cli-augment-registry/`
- `openspec/changes/tools-cli-metadata-contracts/`
- `openspec/changes/docs-toolchain-lifecycle/`
- `docs/toolchain/_shared/lifecycle.py`

## Current State Overview

- `codeintel/indexer/cli.py` exposes two Typer commands (`query`, `symbols`) with hard-coded option metadata and direct `typer.echo` JSON dumps.
- No shared CLI configuration: there is no `CLIToolSettings`, `OperationContext`, or augment/registry ingestion; metadata lives inline (docstrings, help strings) and is not consumable by other tooling.
- No standardized CLI envelope or Problem Details emission; failures rely on Typer exceptions (`typer.BadParameter`) and unstructured output.
- Downstream tooling (OpenAPI generator, D2 diagram builder, docstring builder) does not currently surface this CLI because the metadata contracts are absent.
- Logging, metrics, and observability are minimal; there is no structured logging or Prometheus integration to plug into the doc lifecycle helpers.

## Migration Goals

1. **Adopt shared CLI configuration + operation context.**
2. **Drive command metadata through the augment + registry facade with Pydantic contracts.**
3. **Emit CLI envelopes, Problem Details, and structured logging consistent with the shared standards.**
4. **Integrate with downstream tooling (OpenAPI, diagrams, doc lifecycle) using the new metadata.**

## Detailed Implementation Plan

### 1. CLI Configuration & Operation Context

- **Introduce settings factory** (`codeintel/indexer/cli_context.py` or similar) returning `CLIToolSettings` with:
  - `bin_name="kgf-codeintel"` (confirm finalized name), `title="KGFoundry CodeIntel Indexer"`, `version` from `importlib.metadata.version("kgfoundry-codeintel", fallback="0.0.0")`.
  - `augment_path=Path("tools/_shared/_augment_cli.yaml")`, `registry_path=Path("tools/mkdocs_suite/api_registry.yaml")`.
  - `interface_id="codeintel-indexer"` (ensure registry metadata added below).
- **Load shared context**: at module import, call `load_cli_tooling_context(settings)` and expose `CLI_CONTEXT`, `CLI_CONFIG`, and shortcuts for augment / registry models.
- **Typer app metadata**: update `app = typer.Typer(...)` to pull help text, version, and command names from `CLI_CONFIG`. Ensure `CLI_CONFIG.help_text` (from augment `x-cli`) is used for console help.
- **Operation metadata**:
  - For each command, call `operation = CLI_CONFIG.operation_context.build_operation(command_name="query", func=query)` and use `operation.summary`, `operation.description`, and typed parameter specs to populate Typer option help and constrained values. For `language` parameter, derive allowed choices from augment metadata (e.g., `operation.parameters` enumeration) rather than `LANGUAGE_NAMES` constant; keep the runtime `LANGUAGE_NAMES` for validation but ensure metadata matches.
  - For positional arguments (`path`, `dirpath`), create augment parameter definitions referencing canonical schemas (e.g., `kgfoundry:schema/filesystem/path`).
- **Input validation**: integrate metadata-aware validation by leveraging `OperationContext.validate_parameters` before executing the command. If validation fails, raise a Problem Details response (see section 3).

### 2. Augmentation + Registry Handling

- **Registry entry**:
  - Add a `codeintel-indexer` interface in `api_registry.yaml` with fields: owner team, lifecycle stage, entrypoint `codeintel.indexer.cli:app`, supported operations, documentation links, and dependencies. Include `x-docs` pointing to code-intel documentation.
- **Augment metadata**:
  - Create operation entries `codeintel.indexer.query` and `codeintel.indexer.symbols` in `_augment_cli.yaml` with: `summary`, `description`, `tags` (e.g., `code-analysis`, `tree-sitter`), `x-handler` details, environment requirements (Tree-sitter grammars, query files), and Problem Details references for common errors (unsupported language, I/O errors).
  - Define CLI-level extras under `x-cli` (examples illustrating `kgf-codeintel query ...`) and tag groups for diagram grouping (e.g., `codeintel` vs `symbols`).
- **Runtime consumption**:
  - Replace inline help strings and manual metadata with calls to `CLI_CONTEXT.augment.operation_override(operation_id)` and `CLI_CONTEXT.registry.interface("codeintel-indexer")`.
  - Derive option defaults and enumerations from augment metadata to keep Typer configuration in sync with documentation.
- **Validation & fail-fast**: remove hidden fallbacks; if metadata is missing or inconsistent, raise `CLIConfigError` with Problem Details pointing to spec mismatch.

### 3. Metadata Contracts, Envelopes & Observability

- **CLI envelopes**:
  - Instantiate `CliEnvelopeBuilder` within each command. Capture inputs (`path`, `language`, `query_file`) and results (`hits`, `symbol_count`). If there are multiple outputs, add them via `builder.add_record` with structured fields.
  - Write envelope JSON to `site/_build/cli/codeintel/<command>.json` (configurable via `CLI_CONFIG.artifact_dir`). Provide `--envelope-path` option (wired through settings) for overrides when running locally.
- **Problem Details**:
  - Replace Typer exceptions with metadata-driven Problem Details: when language unsupported, raise a domain exception caught by a shared handler that calls `OperationContext.render_problem` referencing the augment-defined problem type (e.g., `codeintel/unsupported-language`). Include `supported_languages` in the Problem Details extensions.
  - For filesystem errors (missing query file, unreadable source), wrap exceptions in `ProblemDetailsError` with context fields (`path`, `query`).
- **Structured logging**:
  - Use `CLI_CONTEXT.logger.with_fields(operation_id=operation.operation_id, correlation_id=builder.correlation_id)` for all logs. Emit start/success/failure events with counts (e.g., number of captures).
- **Metrics**:
  - Register counters/histograms via `CLI_CONTEXT.metrics.observe_start/success/failure`. Track counts of captures, languages processed, and query durations for observability dashboards.
- **Doc lifecycle coordination**:
  - Ensure envelope artifacts are consumable by `docs/toolchain/_shared/lifecycle.py` once integrated, enabling doc scripts to surface code-intel examples with consistent logs/metrics. Add follow-up tasks to extend lifecycle helpers with code-intel specific hooks (if needed) referencing the `docs-toolchain-lifecycle` spec.

### 4. Downstream Tooling & Tests

- **Typer → OpenAPI**: update fixtures in `tests/tools/test_typer_to_openapi_cli.py` (or add new ones) to import the code-intel CLI and assert the generated OpenAPI `x-cli` block matches augment metadata.
- **CLI diagrams**: extend `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` to cover multi-tag operations from the code-intel CLI, ensuring HTTP verb checks and missing `operationId` logic operate correctly with the new metadata.
- **Docstring builder**: confirm `DocstringBuildResult.tooling_metadata` includes the code-intel operations and add regression tests verifying the docstring builder surfaces code-intel metadata when generating docs.
- **Navmap alignment**: retire `__navmap__` exports once the navmap loader migrates to shared metadata (`cli_scope_navmap_loader.md`); ensure the code-intel entries appear via the shared loader.
- **Lifecycle tests**: once the doc lifecycle helpers are in place, add smoke tests ensuring CLI envelopes are recognized by the lifecycle (e.g., building symbol index docs referencing code-intel commands).

### Acceptance Checklist

- [ ] `codeintel/indexer/cli.py` constructs `CLIToolSettings` and loads `CLIToolingContext` during import; Typer app metadata reads from `CLI_CONFIG`.
- [ ] Each command (`query`, `symbols`) derives metadata via `OperationContext.build_operation` with no direct YAML parsing or hard-coded help strings.
- [ ] `_augment_cli.yaml` and `api_registry.yaml` contain complete entries for the code-intel interface and operations, validated by `ToolingMetadataModel`.
- [ ] CLI emits standardized envelopes, structured logs, metrics, and Problem Details consistent with the shared contracts.
- [ ] Downstream tooling (OpenAPI generator, CLI diagrams, docstring builder) includes regression coverage for the code-intel CLI using the shared metadata.
- [ ] Envelope artifacts integrate with `docs/toolchain/_shared/lifecycle.py` when the doc lifecycle refactor lands, ensuring observability hooks remain consistent with `openspec/changes/docs-toolchain-lifecycle`.
- [ ] Ruff, Pyright, and Pyrefly run clean after changes; legacy navmap exports and bespoke logging paths are deprecated.

With these steps, the code-intel CLI will align fully with the shared CLI tooling standards, enabling immediate consumption across documentation, observability, and downstream applications without bespoke adapters.


