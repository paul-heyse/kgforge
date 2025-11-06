# Migration Scope: `kgfoundry_download` Typer CLI

This document captures the concrete steps required to migrate the downloader/harvester CLI (`src/download/cli.py`) onto the shared CLI tooling stack (configuration context, augment/registry facade, Pydantic metadata contracts) and to update downstream consumers accordingly.

## Current State Overview

- The CLI exposes a single Typer command `harvest` with minimal help text and no structured metadata.
- There is no consumption of `_augment_cli.yaml` or `api_registry.yaml`; tags/handlers/problem details are hard-coded (or absent).
- Telemetry/logging relies on raw `typer.echo` output; no CLI envelope or Problem Details schema.
- Documentation tooling (OpenAPI generator, CLI diagrams) currently has no canonical metadata for this command.

## Migration Goals

1. **Adopt shared CLI configuration + operation context.**
2. **Route all metadata through the augment/registry facade.**
3. **Emit standard CLI envelopes and Problem Details, enabling downstream tooling.**

## Detailed Implementation Plan

### 1. CLI Configuration & Operation Context

- **Introduce settings loader**: add a helper (e.g., `src/download/cli_context.py`) that constructs `CLIToolSettings` with:
  - `bin_name="kgf"`, `title="KGFoundry Downloader"`, `version` read from package metadata (`importlib.metadata.version("kgfoundry")` fallback to `0.0.0`).
  - `augment_path=Path("openapi/_augment_cli.yaml")`, `registry_path=Path("tools/mkdocs_suite/api_registry.yaml")`, `interface_id="download-cli"` (ensure this ID exists in the registry; update YAML if needed).
- **Load shared context at module import**: in `src/download/cli.py`, call `load_cli_tooling_context(settings)` once and store the result in a module-level `CLI_CONTEXT`. Expose convenience accessors (`CLI_CONFIG = cast(CLIConfig, CLI_CONTEXT.cli_config)` etc.).
- **Integrate with Typer**: update `app = typer.Typer(...)` to use metadata from `CLI_CONFIG` (e.g., `help=f"{CLI_CONFIG.title} ({CLI_CONFIG.version})"`). Pass `CLI_CONFIG.bin_name` to CLI envelope later.
- **Operation context binding**: modify the `harvest` command to derive operation metadata via `CLI_CONFIG.operation_context.build_operation(tokens=["harvest"], command_obj=harvest)`. Use the returned operation dict to populate summary/description and to drive help text (`typer.Option(..., help=operation_summary)`).
- **Argument normalization**: ensure Typer options align with the operation schema defined in augment (e.g., if augment declares `x-env`, mirror them via Typer options or environment docs).

### 2. Augmentation + Registry Handling

- **Registry alignment**: ensure `api_registry.yaml` contains an interface entry for `download-cli` with fields (`entrypoint: src.download.cli:app`, owner, stability, spec). Add any missing metadata; sync tests.
- **Augment metadata**:
  - Add an `operations` entry in `_augment_cli.yaml` for `download.harvest` (or similar ID) specifying tags, summary, description, `x-handler`, and Problem Details references.
  - Update augment to include any `x-cli` extras (e.g., sample commands) so the CLI diagrams and OpenAPI generator can display them.
- **Use `ToolingMetadataModel` at runtime**: when the command executes, call `CLI_CONTEXT.augment.operation_override(operation_id)` to fetch metadata and embed it in logs/envelopes. Also retrieve interface metadata (`CLI_CONTEXT.registry.interface(interface_id)`) to enrich CLI envelope extras (owner, stability, spec URLs).
- **Remove legacy defaults**: delete hard-coded help strings and environment documentation; derive them only from the facade.

### 3. Metadata Contracts & Downstream Integration

- **Standardized envelope**:
  - Before executing command logic, instantiate `CliEnvelopeBuilder.create(command="download", status="success", subcommand="harvest")`.
  - As the command runs, record file-level actions (e.g., what would be harvested) via `builder.add_file(...)` or `builder.add_error(...)` if validation fails.
  - Log using `tools._shared.logging.with_fields(logger, {"operation": op_id, ...})` to ensure structured telemetry.
  - On completion, call `builder.finish(duration_seconds=elapsed)` and write the JSON to `site/_build/cli/download.json` (mirroring orchestration CLI). Print or log the JSON path (typer echo only when necessary).
- **Problem Details alignment**:
  - Refactor error handling (e.g., invalid topic or network failure) to raise domain-specific exceptions that are converted to Problem Details using `tools._shared.problem_details.build_problem_details` with `OperationContext`. Ensure augment metadata includes references to canonical Problem Details examples.
- **OpenAPI + diagram updates**:
  - Add/expand tests in `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` to include the download CLI; ensure the generated operations include tags/links from the new metadata.
  - Update CLI OpenAPI generator tests (or add new ones) verifying that `tools/typer_to_openapi_cli.py` can import the download CLI and produce consistent `x-cli` metadata (requires augment/registry definitions).
- **Docstring builder / downstream docs**:
  - If docstring builder references CLI metadata (e.g., for docs automation), ensure its pipeline ingests `ToolingMetadataModel` output (already stored in `DocstringBuildResult.tooling_metadata`) to keep docs in sync.
  - Once the doc toolchain lifecycle (`docs/toolchain/_shared/lifecycle.py`, see `openspec/changes/docs-toolchain-lifecycle`) lands, wire the download CLI’s envelope artifacts into the lifecycle-driven docs commands so interface catalogs/navmaps automatically reflect the new metadata.
- **Deprecate old outputs**:
  - Remove any existing JSON/print formats or ad-hoc logging not aligned with the envelope.
  - Add lint/tests ensuring `_augment_cli.yaml` is the only metadata source (e.g., check there’s no direct YAML loads in the CLI module).

### Acceptance Checklist

- [ ] Typer app constructs `CLIToolSettings` and loads `CLIToolingContext` at import time.
- [ ] `harvest` command uses `OperationContext.build_operation` for metadata + Problem Details.
- [ ] CLI emits envelopes via `CliEnvelopeBuilder` and writes JSON artifacts.
- [ ] `_augment_cli.yaml` and `api_registry.yaml` contain complete metadata for `download-cli`.
- [ ] OpenAPI/diagram/doc tests updated to cover the download CLI.
- [ ] Legacy help text / manual metadata removed; there is no direct YAML parsing outside the facade.
- [ ] Structured logging and metrics (once lifecycle helpers land) are wired to the shared helpers.
- [ ] Documentation (AGENTS / developer docs) updated to warn that the legacy CLI path is deprecated and the new metadata contract is authoritative.

Once these items are complete, the download CLI will be fully aligned with the shared tooling and ready for downstream consumers to rely on the standardized metadata without additional glue.


