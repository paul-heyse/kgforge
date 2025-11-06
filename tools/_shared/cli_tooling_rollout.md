# CLI Tooling Rollout

## Purpose

We now have a canonical augment/registry facade (`tools._shared.augment_registry`) and a shared
CLI config loader (`tools._shared.cli_tooling`). The next step is to migrate every CLI surface so
that:

- augment metadata, registry interfaces, and Problem Details envelopes flow through the shared
  helpers;
- Typer/Click apps expose consistent metadata for OpenAPI + MkDocs generators;
- tooling CLIs (doc builders, navmap, docs toolchain) emit the same structured logging/metrics and
  reuse the shared context objects.

This document inventories the current CLI entry points and captures the migration tasks required to
bring each one onto the new standard.

## Inventory of CLI Entry Points

The list below covers every executable Python entry point we ship today (Typer apps, console
scripts, and stand‑alone tooling). Each row includes the implementation module, how the command is
invoked, and its current relationship to the shared augment/registry tooling.

### Runtime Typer Apps (user-facing)

| CLI / Command | Module & Entry Point | Framework | Current Metadata Integration | Notes / Actions |
| --- | --- | --- | --- | --- |
| Orchestration CLI (`kgfoundry orchestration`) | `src/orchestration/cli.py::app` | Typer | Uses `kgfoundry_common.navmap_loader` + hand-rolled configs; does **not** consume `ToolingMetadataModel` | Needs augment/registry-backed context for index commands + Problem Details alignment; map CLI options to shared envelope/logging helpers |
| Downloader CLI | `src/download/cli.py::app` | Typer | No augment/registry usage (simple Typer wrapper) | Decide whether this CLI should surface through shared metadata (at minimum, adopt CLI envelope + Problem Details schema) |
| CodeIntel indexer | `codeintel/indexer/cli.py::app` | Typer | No augment/registry usage; outputs raw JSON | Adopt shared CLI envelope + Problem Details; evaluate if augment/registry metadata applies for future docs |

### Documentation & Spec Tooling

| CLI / Command | Module & Entry Point | Framework | Current State w.r.t Shared Tooling | Follow-up |
| --- | --- | --- | --- | --- |
| OpenAPI generator | `tools/typer_to_openapi_cli.py::main` | Click (Typer import) | ✅ Already uses `ToolingMetadataModel` + `load_cli_tooling_context` | Monitor after metadata refactor to keep schema/examples in sync |
| CLI diagram generator | `tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py::main` | MkDocs hook (imperative) | ✅ Consumes shared context to avoid dead links | No further action beyond keeping tests in sync |
| Interface catalog generator | `tools/mkdocs_suite/docs/_scripts/gen_interface_pages.py::main` | MkDocs hook | ⚠️ Uses shared registry loader; augment still partially manual; logging/metrics inconsistent | Finish migration to new lifecycle helpers + adopt Problem Details + metrics |
| Module page generator | `tools/mkdocs_suite/docs/_scripts/gen_module_pages.py::main` | MkDocs hook | ⚠️ Shared registry loader integrated; rest of metadata pipeline still bespoke | Wire through shared augment context for operation links + unify logging |
| Documentation navmap builder | `tools/navmap/build_navmap.py::main` (and related `check_navmap`, `migrate_navmaps`, `repair_navmaps`, `strip_navmap_sections`) | argparse | ❌ No shared context; bespoke YAML parsing | Introduce helper that loads augment/registry once; refactor logging + Problem Details to new facade |
| Typer CLI OpenAPI metadata | `tools/_shared/cli_tooling.py` consumers | n/a | ✅ Ready (facade now Pydantic) | Propagate to remaining tooling |

### Documentation Toolchain Console Scripts (`pyproject.toml [project.scripts]`)

| Script | Entry Point | Current Status | Rollout Notes |
| --- | --- | --- | --- |
| `docs-build-symbol-index` | `docs.toolchain:build_symbol_index` | Placeholder; not wired into new lifecycle | When implementing Phase 3.2, ensure it uses shared lifecycle helpers + augment facade |
| `docs-symbol-delta` | `docs.toolchain:symbol_delta` | Placeholder | Same as above |
| `docs-validate-artifacts` | `docs.toolchain:validate_artifacts` | Placeholder | Same as above |
| `tools-build-navmap` | `tools.cli:build_navmap` (re-exports `tools/navmap/build_navmap`) | Active CLI, currently bespoke | Align with navmap plan above |

### Documentation Builder Tooling

| CLI / Command | Module & Entry | Framework | Status | Actions |
| --- | --- | --- | --- | --- |
| Docstring Builder CLI | `tools.docstring_builder.cli:main` / `python -m tools.docstring_builder` | argparse | ❌ Manual YAML/JSON handling; CLI envelope partially used downstream (pipeline now records `ToolingMetadataModel`) | Migrate CLI argument parsing to shared settings/context; emit Problem Details + metrics via new lifecycle helpers |
| Docstring builder automation scripts | `tools/generate_docstrings.py`, `tools/generate_pr_summary.py`, `tools/add_module_docstrings.py`, etc. | argparse | ❌ Stand-alone; no shared tooling | Evaluate which scripts should consume the shared CLI envelope vs remain internal helpers |

### Other Utility CLIs

| CLI / Command | Module | Notes |
| --- | --- | --- |
| `tools/update_navmaps.py` | argparse script orchestrating navmap + mkdocs | Needs migration once navmap helpers are refactored |
| `tools/update_docs.sh` (shell) | bash | Out of Python scope; reference only |
| `tools/validate_gallery.py` | argparse | Consider folding into shared CLI envelope post-doc toolchain migration |
| `tools/docs/*` (build_artifacts, build_graphs, export_schemas, build_test_map, scan_observability) | argparse scripts | Evaluate case-by-case; many are batch utilities that could reuse shared logging + Problem Details |

## Rollout Phases

1. **Complete metadata hardening**
   - Finalize `ToolingMetadataModel` adoption path (PR in flight for metadata scope).
   - Ensure `load_cli_tooling_context` emits `CLIConfig` + `ToolingMetadataModel` consistently.

2. **Documentation tooling migration**
   - Refactor `navmap` suite to consume the facade and adopt lifecycle helpers.
   - Implement docs toolchain lifecycle module (per `docs-toolchain-lifecycle` spec) and wire the
     console scripts through it.
   - Finish MkDocs script migration (interface/module generators) with shared logging + Problem Details.

3. **Docstring builder alignment**
   - Update `tools.docstring_builder.cli` to build requests from `ToolingMetadataModel`
     (operation tags, interface metadata) and emit CLI envelopes via `tools._shared.cli`.
   - Extend pipeline outputs/tests once metadata model lands.

4. **Runtime CLI integration**
   - Decide how much augment/registry data runtime Typer CLIs should surface (at minimum, provide
     consistent metadata so OpenAPI + diagrams stay in sync).
   - Hook orchestration/download/codeintel apps into shared context (likely read-only access for
     CLI diagrams + Problem Details registry).

5. **Enforcement & automation**
   - Add CI lint that flags new direct reads of `_augment_cli.yaml` or `api_registry.yaml` outside
     the facade.
   - Add documentation describing the standard (AGENTS + developer docs) and link from new CLIs.
   - Track adoption with a simple metric/log in the facade (e.g., info log per CLI name).

## Immediate Next Steps

- Finish metadata scope (Pydantic models + facade) and land pyright/ruff fixes.
- Draft migration playbook (code snippets + before/after) for engineers touching each CLI
  category.
- Kick off navmap + docstring builder refactors—they touch the most downstream consumers and will
  unblock the rest.
- Update AGENTS.md / tooling docs referencing this rollout document once migrations begin.

## Open Questions

- Should runtime Typer CLIs expose augment metadata for automation (e.g., generating docs from
  their Typer apps), or is the shared context primarily for documentation tooling?
- Do we want a single registry of CLI envelope outputs (e.g., store JSON in `_build` for drift
  detection)? If so, we should specify storage conventions alongside the rollout.
- Which smaller utility scripts should be folded into the shared standard vs remain lightweight wrappers?

Maintain this document as we migrate each CLI—update the inventory tables with ✅ once a command is
fully aligned with the shared tooling.

## CLI Implementation Scopes

To avoid staging or partial rollouts, each CLI listed above will receive a focused implementation
plan. These scopes describe the exact refactors required to align with the new configuration,
augment/registry facade, and metadata contracts.

### `kgfoundry_orchestration` (Typer app in `src/orchestration/cli.py`)

**Goal:** migrate the orchestration CLI so every command (BM25/FAISS index builders, E2E flows,
Prefect helpers) consumes the shared CLI tooling context, emits standard envelopes, and exposes
metadata that downstream documentation tooling can use without bespoke glue.

#### 1. CLI Configuration & Operation Context

- Introduce a thin adapter module (e.g., `src/orchestration/cli_context.py`) that constructs
  `CLIToolSettings` using the authoritative repository defaults (bin name `kgf`, title
  "KGFoundry CLI", version resolved from package metadata, augment/registry paths at repo root).
- At Typer start-up (`app = typer.Typer(...)`), call `load_cli_tooling_context(settings)` once and
  store the resulting `CLIConfig` + `ToolingMetadataModel` on the app state (e.g., `app.state` or a
  module-level `ORCH_CONTEXT`).
- Refactor each command handler (`index_bm25`, `index_faiss`, `e2e`, etc.) to:
  - Accept an injected `OperationContext` (obtain via `CLIConfig.operation_context`), using the
    command tokens Typer already provides to derive `operationId`, tags, summaries, etc.
  - Replace ad-hoc strings for Problem Details `instance`/`type` with values derived from
    `OperationContext.build_operation` so generated OpenAPI + CLI diagram metadata stays in sync.
- Swap manual configuration (multiple dataclasses/constants) for a single `CLIConfig` access point:
  - Map Typer arguments/options to the config’s `settings` block (populate CLI envelope fields).
  - When running an operation (e.g., building FAISS index), record success/error against the
    shared `CliEnvelopeBuilder` from `tools._shared.cli` before returning or raising.

#### 2. Augmentation + Registry Handling

- Remove direct imports of `kgfoundry_common.navmap_loader` for CLI metadata. Instead, enrich the
  Typer commands by reading from the shared `ToolingMetadataModel`:
  - Use `ToolingMetadataModel.operation_override(tokens)` to fetch `summary`, `x-handler`,
    `problem_details`, and environment variables for each subcommand; surface these in help text and
    telemetry logs.
  - Use `ToolingMetadataModel.registry.interface(interface_id)` to populate CLI-wide metadata such
    as binary name, owner, stability, and spec links (for Typer `app.help` and for log summaries).
- Cache the loaded metadata in module scope to avoid repeated YAML parsing; rely on the facade’s
  lru-cache for file change detection.
- Guarantee that generated Problem Details reference the shared registry entries—e.g., when FAISS
  indexing fails, use `interface.to_payload()` to embed interface metadata/tags and point to the
  canonical Problem Details examples declared in the registry extras.
- Delete any legacy YAML parsing or navmap lookups that previously attempted to read
  `_augment_cli.yaml` or `api_registry.yaml` directly.

#### 3. Metadata Contracts & Downstream Consumers

- Wrap all command execution in the shared CLI envelope builders:
  - On command start, create `CliEnvelopeBuilder.create(command="orchestration", subcommand=...)
    and record start via shared logging (`tools._shared.logging.with_fields`).
  - On success/failure, call `.finish(duration_seconds=elapsed)` and write the JSON to disk (e.g.,
    `site/_build/cli/orchestration.json`) so downstream tooling (docs, navmap, dashboards) has a
    deterministic artifact.
  - Emit structured logging/metrics using `DocLifecycle` once the docs toolchain lifecycle lands;
    for now, reuse the existing orchestrator logging but add the envelope payload to log extras.
- Update downstream documentation consumers:
  - Ensure `tools/typer_to_openapi_cli.py` has an integration test that imports the orchestration
    app using the new context so that OpenAPI generation matches the shared metadata.
  - Update CLI diagram tests (`tests/tools/mkdocs_suite/test_gen_cli_diagram.py`) to assert that
    orchestration commands include the new tags/links pulled from `ToolingMetadataModel`.
  - Confirm that docstring builder tests ingest the new `tooling_metadata` emitted by the pipeline so
    any doctest or documentation generation referencing orchestration commands uses the canonical
    metadata.
  - As the doc toolchain lifecycle (`docs/toolchain/_shared/lifecycle.py`) rolls out, ensure the
    orchestration CLI’s envelope JSON is consumed by the lifecycle-powered docs scripts so navmap
    sections, interface catalog entries, and drift reports stay aligned (link the implementation to
    the `docs-toolchain-lifecycle` spec once merged).
- Communicate migration details to downstream users:
  - Document the new envelope/artifact location in developer docs.
  - Advertise breaking changes (e.g., improved Problem Details, new CLI JSON output) to SRE/doc
    teams so dashboards/tests can be updated immediately.
- Remove deprecated code paths: delete any code under `src/orchestration/cli.py` that rehydrates
  navmap metadata or keeps alternate config dataclasses, ensuring the shared facade is the single
  source of truth from this commit forward.

Once these steps are merged, the orchestration CLI will operate entirely on top of the new shared
tooling with no legacy fallback, enabling documentation generators and runtime telemetry to consume
the same metadata contracts.


