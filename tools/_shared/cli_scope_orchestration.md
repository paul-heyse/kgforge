# Migration Scope: `kgfoundry_orchestration` Typer CLI (`src/orchestration/cli.py`)

This document describes the concrete steps required to migrate the orchestration CLI onto the shared tooling stack (CLI configuration/operation contexts, augment + registry facade, Pydantic metadata contracts, and the new doc toolchain lifecycle helpers). The goal is to eliminate bespoke metadata paths, unify observability, and guarantee downstream documentation tools consume the canonical view of orchestration commands.

## Current State Overview

- `src/orchestration/cli.py` defines multiple Typer commands (`api`, `e2e`, `index_bm25`, `index_faiss`, `run_index_faiss`) with heterogeneous help text and logging.
- Metadata is derived ad hoc: manual docstrings, `load_nav_metadata` sidecar population, direct schema loads, and custom Problem Details builders.
- Error handling relies on bespoke exceptions (`IndexBuildError`, `ConfigurationError`) mapped manually to Problem Details via `kgfoundry_common.problem_details` helpers.
- CLI outputs do not emit standardized CLI envelopes; telemetry is logged via raw `logging` calls.
- Downstream tooling (OpenAPI generator, CLI diagrams, docstring builder) pulls inconsistent metadata; navmap alignment depends on `__navmap__` exported from `load_nav_metadata`.

## Migration Goals

1. **Adopt shared CLI configuration + operation context (`CLIConfig`, `OperationContext`).**
2. **Drive commands exclusively through the augment/registry facade (`tools._shared.augment_registry`).**
3. **Emit standardized CLI envelopes, Problem Details, and structured logging consistent with metadata contracts.**
4. **Align downstream tooling (OpenAPI, MkDocs suites, docstring builder, doc lifecycle helpers) with the new canonical metadata.**

## Detailed Implementation Plan

### 1. CLI Configuration & Operation Context

- **Create orchestration-specific settings factory**: add `src/orchestration/cli_context.py` (or reuse existing config module) exposing `build_cli_settings()` returning `CLIToolSettings`. Populate with:
  - `bin_name="kgf-orchestrate"` (confirm naming), `title="KGFoundry Orchestration"`, `version` from `importlib.metadata.version("kgfoundry-orchestration", fallback="0.0.0")`.
  - `augment_path=Path("tools/_shared/_augment_cli.yaml")` (confirm location) and `registry_path=Path("tools/mkdocs_suite/api_registry.yaml")`.
  - `interface_id="kgfoundry-orchestration"` (ensure registry entry exists / updated below).
- **Load shared context during module import**: in `src/orchestration/cli.py`, call `load_cli_tooling_context(build_cli_settings())` to obtain `CLI_CONTEXT`. Cache `CLI_CONFIG = CLI_CONTEXT.cli_config` plus `ORCHESTRATION_INTERFACE = CLI_CONTEXT.registry.interface(...)`.
- **Wire Typer application details**: update `app = typer.Typer(...)` to use `CLI_CONFIG.title`, `CLI_CONFIG.version`, and `CLI_CONFIG.help_text` (derived from augment `x-cli`). Ensure Typer `Context.command_path` matches `CLI_CONFIG.bin_name`.
- **Operation metadata for each command**:
  - Replace manual docstrings/help text with `operation = CLI_CONFIG.operation_context.build_operation(command_name="index_bm25", func=index_bm25)`.
  - Use `operation.summary`, `operation.description`, and `operation.parameters` to populate Typer option help and default values.
  - For commands that spawn sub-flows (`run_index_faiss`, `e2e`), include `operation.tags` to group outputs in diagrams.
- **Inputs/outputs alignment**: ensure Typer parameters align with `operation.request_body` / `operation.parameters`. Where augment defines JSON schema references, leverage `CLI_CONFIG.operation_context.resolve_parameter_schema` to validate input prior to executing the command.

### 2. Augmentation + Registry Handling

- **Registry metadata**:
  - Update `tools/mkdocs_suite/api_registry.yaml` to include an interface entry for `kgfoundry-orchestration` with fields: owner team, lifecycle, entrypoint (`src.orchestration.cli:app`), supported operations, and canonical documentation URLs.
  - Ensure each operation declares `operation_id` matching the Typer command path (`kgfoundry-orchestration.index_bm25`, etc.).
- **Augment metadata**:
  - Expand `_augment_cli.yaml` to define `operations` entries for the five commands including `summary`, `description`, `tags`, `x-handler`, `x-cli` envelope hints, and Problem Details references.
  - Define tag groups (e.g., `indexing`, `pipelines`, `api`) for consistent grouping in diagrams and docs.
  - Document CLI-level examples (sample invocations, environment requirements) under `x-cli.examples`.
- **Runtime usage**:
  - Replace direct `load_nav_metadata`, schema, or YAML parsing with calls to `CLI_CONTEXT.augment` and `CLI_CONTEXT.registry` helpers (`operation_override`, `interface`, `tag_group`).
  - Ensure CLI logic references `AugmentMetadataModel` for configuration (e.g., retrieving Problem Details template for index failures).
- **Fallback removal**: delete bespoke metadata fallback logic once augment/registry coverage reaches 100%; add guard rails to fail-fast if metadata missing (raise `CLIConfigError` with Problem Details).

### 3. Metadata Contracts, Envelopes & Observability

- **CLI envelope integration**:
  - Introduce a `CliEnvelopeBuilder` invocation at the entry to each command. Include `operation_id`, correlation ID, inputs, and outputs as the command runs.
  - For long-running tasks (index builds), log progress via `builder.add_step(...)` or `builder.add_log(...)`, aligning messages with metadata tags.
  - On completion/failure, call `builder.finish(status="success" | "failure", duration_seconds=elapsed)` and write JSON to `site/_build/cli/orchestration/<command>.json`. Provide a CLI flag or environment variable to control output paths (configured via `CLI_CONFIG.artifact_dir`).
- **Problem Details mapping**:
  - Convert existing exception handling to raise domain-specific exceptions that bubble into `OperationContext.render_problem`, ensuring Problem Details output fields (`type`, `title`, `detail`, `status`, `instance`) match augment definitions.
  - For JSON schema validation (vector ingestion), move the formatting of validation errors into a helper registered with the metadata contract so Problem Details attachments align with CLI docs.
- **Structured logging**:
  - Replace manual `logger.info`/`logger.warning` with `CLI_CONTEXT.logger.with_fields(operation_id=..., correlation_id=...)`. For failures, use `logger.exception` to capture stack traces while referencing Problem Details ID.
- **Metrics & doc lifecycle alignment**:
  - Register CLI-specific Prometheus counters (operation success/failure, duration) via `CLI_CONTEXT.metrics`. This data should feed into doc lifecycle metrics once the doc toolchain consumes the CLI envelopes.
  - Coordinate with the doc lifecycle module by emitting events/envelope artifacts that `docs/toolchain/_shared/lifecycle.py` can ingest (e.g., handing off compilation results or index metadata).

### 4. Downstream Tooling & Lifecycle Integration

- **Typer → OpenAPI generator**: ensure `tools/typer_to_openapi_cli.py` can import `src.orchestration.cli:app` without side effects. Add/adjust fixtures so the new CLI context loads during tests, verifying that generated OpenAPI `x-cli` blocks match augment metadata.
- **CLI diagrams**: update `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` with orchestration-specific fixtures ensuring multi-tag operations and missing operation IDs are handled correctly via the shared metadata.
- **Docstring builder**: confirm `tools/docstring_builder/pipeline.py` (already consuming `ToolingMetadataModel`) picks up orchestration metadata; add targeted regression cases for envelope ingestion.
- **Navmap loader alignment**: replace `__navmap__ = load_nav_metadata(...)` with metadata sourced from the shared facade. If navmap sidecars remain necessary, generate them via `CLI_CONTEXT` to avoid drift.
- **Doc toolchain lifecycle (`docs/toolchain/_shared/lifecycle.py`)**:
  - Ensure envelope JSON from the orchestration CLI is registered as an input for `build_symbol_index.py`, `symbol_delta.py`, and `validate_artifacts.py` once those scripts adopt the lifecycle helpers. Provide a follow-up task to add hooks in `docs/toolchain/_shared/lifecycle.py` that read the CLI envelope artifacts and populate doc context, per the `docs-toolchain-lifecycle` spec/design/tasks.
- **Documentation updates**: update `docs/_sources` or relevant guides to reference the shared metadata, removing screenshots or instructions based on legacy behaviour.

### Acceptance Checklist

- [ ] `src/orchestration/cli.py` constructs `CLIToolSettings` and loads `CLIToolingContext` at import time.
- [ ] Each Typer command derives operation metadata via `OperationContext.build_operation` and no longer loads navmap or YAML directly.
- [ ] `_augment_cli.yaml` and `api_registry.yaml` contain complete, typed metadata for orchestration commands and interface.
- [ ] CLI commands emit standardized envelopes, Problem Details, and structured logs aligned with metadata contracts.
- [ ] Legacy navmap exports (`__navmap__`) and bespoke JSON schema handling are removed or rewritten to use shared helpers.
- [ ] OpenAPI generator, CLI diagrams, and docstring builder tests cover the orchestration CLI using the shared metadata.
- [ ] Doc toolchain lifecycle integration plan documented: envelope artifacts consumed by `docs/toolchain/_shared/lifecycle.py` once the lifecycle refactor merges, ensuring observability and docs stay in sync.
- [ ] Linting and type-checking (Ruff, Pyright, Pyrefly) clean post-migration.

When these items are complete, the orchestration CLI will operate entirely on top of the shared tooling abstractions, eliminating duplicated metadata handling and enabling immediate downstream adoption across documentation and observability tooling.


