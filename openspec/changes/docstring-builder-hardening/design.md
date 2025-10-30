## Context
The docstring builder suite powers DocFacts, README generation, and Agent Catalog metadata. Each run pulls harvested symbols, applies policy rules, renders docstrings, and emits artifacts consumed by navmap, README generator, and the Agent Portal. The current implementation relies on dynamically typed dictionaries, broad exception handling, and `print` statements, which:

- prevent Ruff/Security rules (`S603`, `S607`, `T201`, `BLE001`) from passing,
- hide schema drift because payloads are never validated against `docs/_build/schema_docfacts.json`, and
- make failure triage difficult (no structured logs, no Problem Details).

DocFacts 2.0 introduces richer metadata and stricter validation requirements, therefore typed models, safe CLI contracts, and documented observability are mandatory for quality gates and junior engineer handoffs.

## Goals / Non-Goals
- **Goals:**
  - Provide typed data models and schema validation for DocFacts and CLI machine outputs.
  - Reduce cyclomatic complexity in `normalizer`, `policy`, and `render` hotspots through composable helpers.
  - Formalize plugin interfaces via `Protocol`, ensuring explicit error handling and observability.
  - Replace ad-hoc logging/subprocess calls with shared, secure utilities.
- **Non-Goals:**
  - Overhauling downstream consumers beyond necessary type/schema updates.
  - Shipping new summarization features or LLM integrations.
  - Regenerating all docstrings (handled in separate rollout tasks).
  - Changing DocFacts schema contents beyond validation requirements (no new fields without separate proposal).

## Prerequisites for Developers
- Run `scripts/bootstrap.sh` to align Python/uv versions with CI.
- Skim `tools/docstring_builder/` modules to understand existing flow (harvest → normalize → policy → render → CLI).
- Review `docs/_build/schema_docfacts.json` and current CLI JSON emitted by `docstring_builder --json` to understand required shapes.
- Familiarize yourself with structured logging helpers used elsewhere in the repo (e.g., `src/shared/logging.py`) for consistency.

## Decisions (with rationale)
- **Typed IR via `dataclass` + `TypedDict`:** Maintains runtime flexibility while satisfying mypy/pyrefly and aligns with JSON schema contracts.
- **Plugin `Protocol` + compatibility shim:** Allows incremental migration without breaking third-party extensions; deprecation warnings enforce timeline.
- **Shared logging/subprocess modules:** Centralizes security and observability practices, satisfying Ruff S603/S607 and providing structured logs for Ops teams.
- **Jinja autoescape enabled:** Defaults to safe HTML rendering to prevent latent XSS in documentation outputs; configuration allows opt-out when safe.

## Architecture Sketch
```
Harvest → Typed IR (dataclass) → Policy Engine (pure functions) → Renderer (autoescape)
                                 ↘ Plugin Protocol hooks ↗
CLI (structured logs + schema emits) ← Shared logging/proc utilities
```

## Detailed Implementation Plan
| Step | Artifact | Owner Notes |
| --- | --- | --- |
| 1 | Baseline typed contracts (`tools/docstring_builder/models.py`, `schema/tools/docstring_builder_cli.json`) | Confirm shared understanding of dataclasses/TypedDicts, constants, and Problem Details examples; treat as source of truth |
| 2 | Update `normalizer.py` to use new models, split helpers into `resolve`, `format`, `annotations` sections or modules | Each helper returns typed values; add targeted exceptions |
| 3 | Extract `_apply_mapping` logic in `policy.py` into pure helpers with deterministic ordering; add tests under `tests/tools/docstring_builder/test_policy.py` | Keep branch count ≤ 12 |
| 4 | Update `render.py` to enable `autoescape`, move signature construction into smaller functions, and leverage typed IR for inputs | Add golden-file tests |
| 5 | Introduce `tools/_shared/logging.py` + `tools/_shared/proc.py`, adopt in CLI, and remove all `print` usage in library code | Document default log fields |
| 6 | Define plugin `Protocol`, update bundled plugins, and ship compatibility shim with warnings | Provide migration guide |
| 7 | Integrate CLI schema validators & regression tests for success/failure flows | Validate against `schema/tools/docstring_builder_cli.json`; tests run builder in `--json` mode |
| 8 | Instrument observability (logs, metrics, traces) and capture sample outputs for docs | Verify via unit/integration tests |

Junior developers should work through the table sequentially, opening draft PRs per major step to keep reviews focussed.

## Typed Models Outline
- `tools/docstring_builder/models.py` (new)
  - `DocstringIRParameter`, `DocstringIRReturn`, `DocstringIRRaise`, `DocstringIR` dataclasses mirror runtime IR with exhaustive typing.
  - `DocfactsParameter`, `DocfactsReturn`, `DocfactsRaise`, `DocfactsEntry`, `DocfactsDocumentPayload` TypedDicts align with `schema_docfacts.json`.
  - `ProblemDetails` TypedDict + `PROBLEM_DETAILS_EXAMPLE` provide RFC 9457-compliant envelopes for error surfacing.
  - `FileReport`, `ErrorReport`, `RunSummary`, `PolicyReport`, `PluginReport`, `DocfactsReport`, `CliResult` TypedDicts express the CLI payload without `Any` usage.
  - Constants `CLI_SCHEMA_VERSION` (`1.0.0`) and `CLI_SCHEMA_ID` track schema provenance.
  - Helper `build_cli_result_skeleton(status: RunStatus) -> CliResult` assists tests in constructing valid payloads.
- Validators (future work):
  - `validate_docfacts(record: DocfactsDocumentPayload) -> None`
  - `validate_cli_output(payload: CliResult) -> None`
  - Raising `SchemaViolationError` with nested `ProblemDetails` upon failure.

## CLI Schema Definition
- JSON Schema location: `schema/tools/docstring_builder_cli.json` (Draft 2020-12).
- Captures top-level metadata (`schemaVersion`, `status`, `generatedAt`, `durationSeconds`), file-level outcomes, policy violations, plugin summaries, cache metrics, and optional Problem Details payloads.
- Defines reusable `$defs` for `RunStatus`, `FileReport`, `ErrorReport`, `PolicyViolation`, `RunSummary`, `ProblemDetails`, etc., enabling precise validation and linting.
- Provides success and failure examples (including Problem Details) to guide consumers and documentation authors.
- Future CLI code will load the schema at runtime for validation and expose it via `docstring_builder schema --output ...`.

## Plugin Protocol Outline
```python
class DocstringBuilderPlugin(Protocol):
    name: ClassVar[str]

    def supports(self, symbol: SymbolHarvest) -> bool: ...

    def apply(
        self,
        symbol: SymbolHarvest,
        ir: DocstringIR,
        context: PluginContext,
    ) -> PluginResultEnvelope: ...
```

- `PluginContext` includes logger, feature flags, repo root, and schema version.
- Compatibility shim wraps legacy call signature (`apply(symbol, ir)`), logs a `DeprecationWarning`, and converts return data into the envelope shape.
- Plugin errors raise `PluginExecutionError` with nested `ProblemDetails` for consistency.

## CLI Contracts
- CLI config dataclass: `BuilderConfig(input_paths: list[Path], repo_root: Path, jobs: int, feature_flags: dict[str, bool])`.
- Shared subprocess wrapper returns structured results with `stdout`, `stderr`, `duration_ms`, `returncode`.
- CLI payloads must be instances of `CliResult` constructed via `build_cli_result_skeleton` and validated against `schema/tools/docstring_builder_cli.json` prior to emission.
- Success payload example:
  ```json
  {
    "schemaVersion": "1.0.0",
    "schemaId": "https://kgfoundry.dev/schema/docstring-builder-cli.json",
    "status": "success",
    "generatedAt": "2025-10-30T12:00:00Z",
    "command": "update",
    "subcommand": "generate",
    "durationSeconds": 18.42,
    "files": [
      {"path": "src/docling/canonicalizer.py", "status": "success", "changed": true, "skipped": false, "cacheHit": false}
    ],
    "errors": [],
    "summary": {...},
    "policy": {...}
  }
  ```
- Failure payloads include a top-level `problem` entry conforming to the RFC 9457 `ProblemDetails` TypedDict as well as per-file `problem` fields when individual files fail.

## Data Model & Schemas
- DocFacts: `docs/_build/schema_docfacts.json` (existing) — will be loaded and enforced during builder runs.
- CLI Output: `schema/tools/docstring_builder_cli.json` (new) — covers `--json` summaries and planned `--baseline` comparisons.
- Plugin Metadata: optional `schema/tools/docstring_builder_plugin.json` capturing capability flags and failure envelopes.
- Provide examples for each schema; validate with `jsonschema` in tests.

## Test Strategy & Coverage
- **Unit tests:**
  - `test_models.py`: round-trip typed models ↔ schema payloads.
  - `test_normalizer.py`: positional-only/keyword-only/variadic signatures, annotated/unannotated parameters.
  - `test_policy.py`: `_apply_mapping` branches, deterministic ordering.
  - `test_render.py`: autoescape behavior, signature formatting.
- **Plugin tests:**
  - `test_plugins.py`: Protocol compliance, dataclass variance regression, failure envelopes.
- **CLI tests:**
  - `test_cli.py`: success, schema validation failure, subprocess failure (Problem Details path).
- **Doctests:**
  - Problem Details example, sample logging snippet, feature flag usage.
- **Performance tests:** optional micro-benchmark verifying runtime budget.
- **Shadow validation workflow:** run builder with `--json --validate-only` to compare typed vs legacy outputs (no diff expected).

## Invariants & Edge Cases
- Invariant: DocFacts `docfactsVersion` must match builder `DOCFACTS_VERSION` constant; mismatch triggers `SchemaViolationError` with Problem Details payload.
- Edge cases: positional-only/keyword-only/variadic signatures, dataclasses with `kw_only=True`, union annotations, Annotated/Literal forms, missing runtime objects.
- Failure modes: Module import errors, annotation resolution errors, plugin timeouts; each raises typed exceptions with contextual metadata.

## Observability
- Logger namespace: `tools.docstring_builder` with structured fields (`operation`, `duration_ms`, `symbol_id`, `error_type`).
- Metrics: Prometheus counters (`docbuilder_runs_total`, `docbuilder_plugin_failures_total`) and histograms for runtime; integration with existing metrics exporter.
- Traces: Create spans for `harvest`, `policy`, `render`, and `cli` operations; propagate correlation IDs across subprocess boundaries.

### Sample Problem Details Payload
```json
{
  "type": "https://kgfoundry.dev/problems/docbuilder/schema-mismatch",
  "title": "DocFacts schema validation failed",
  "status": 422,
  "detail": "Field anchors.endLine is missing",
  "instance": "urn:docbuilder:run:2025-10-30T12:00:00Z",
  "extensions": {
    "schemaVersion": "2.0.0",
    "docstringBuilderVersion": "1.6.0",
    "symbol": "kg.module.function"
  }
}
```
Doctests must exercise rendering of this payload and confirm structured logging includes the `instance` identifier.

## Security / Privacy
- All subprocess invocations flow through `tools._shared.proc.run_tool`, enforcing absolute executables and environment sanitization.
- Autoescape prevents untrusted docstring content from injecting HTML; sanitized logs avoid PII.
- Plugin inputs validated against schema; reject path traversal by normalizing via `pathlib` and explicit allowlists.

## Migration / Compatibility
- Phase 0: Ship typed models behind feature flag and compatibility shim for plugins; run in CI shadow mode.
- Phase 1: Flip default to typed/validated pipeline; retain shim until next minor release.
- Rollback: Flag disables typed enforcement, reverting to legacy behavior while logging warnings.

### Rollout Playbook
1. Merge typed models + validators with feature flag default off.
2. Enable feature flag in CI only; collect metrics/logs for one sprint.
3. Address discrepancies discovered via validation (update schema or code as needed).
4. Flip feature flag default on; communicate to plugin authors and ensure compatibility shim still present.
5. After two releases with no regressions, remove shim and deprecated code paths.

## Open Questions
- Q1: Should CLI Problem Details payloads be centralized across all tools? (Decision pending shared error module.)
- Q2: Do we version plugin Protocol separately from DocFacts schema? (Coordinate with tooling stakeholders.)
- Q3: Can we reuse existing metrics exporters for doc tooling, or do we introduce a new Prometheus namespace? (Requires Ops input.)
- Q4: Do we need a migration command to auto-fix third-party plugins, or is documentation sufficient? (Assess after stakeholder review.)
