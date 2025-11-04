## Context

The docs toolchain is the final consumer-facing pipeline that renders API references, symbol indices, and
nav data. It currently skirts type enforcement and schema policies, leaving residual Ruff/Pyrefly/pyright debt
and inconsistent logging. This design note captures the implementation shape before coding begins.

## Goals / Non-Goals

- **Goals:**
  - Deliver a typed, schema-validated docs pipeline with cohesive configuration and observability.
  - Remove `Any` usage and blind exception handling from Sphinx config and helper scripts.
  - Provide deterministic validation commands that gate merges.
- **Non-Goals:**
  - Redesign the documentation site content or theming.
  - Introduce new docs build targets beyond the existing JSON and MkDocs pipelines.

## Four-Item Design Note

1. **Summary**
   - Introduce shared helpers, typed protocols, and schema validation so every docs build script operates
     without `Any` surfaces, while hardening logging/error handling to emit RFC 9457 envelopes and emit
     metrics/tracing metadata for downstream observability.

2. **Public API Sketch**
   - `docs._scripts.shared.DocsSettings` — `@dataclass(slots=True, frozen=True)` with fields:
     `packages: tuple[str, ...]`, `link_mode: Literal["editor", "github", "both"]`, `github_org: str | None`,
     `github_repo: str | None`, `github_sha: str | None`, `docs_build_dir: Path`, `navmap_candidates: tuple[Path, ...]`.
   - `docs._scripts.shared.make_logger(operation: str, *, artifact: str | None = None) -> logging.LoggerAdapter`
     returning a `with_fields` adapter seeded with correlation metadata.
   - `docs._scripts.shared.resolve_git_sha(env: BuildEnvironment, settings: DocsSettings, *, logger: WarningLogger) -> str`.
   - `docs._scripts.build_symbol_index.generate_index(packages: Sequence[str], loader: GriffeLoader, settings: DocsSettings) -> SymbolIndexArtifacts` (dataclass bundling symbol rows, reverse lookups, nav metadata).
   - `docs._scripts.mkdocs_gen_api.generate_api_reference(loader: GriffeLoader, packages: Sequence[str]) -> list[RenderedPage]` where `RenderedPage` is a dataclass (`output_path: Path`, `content: str`).
   - `docs._scripts.symbol_delta.compute_delta(base: SymbolIndexSnapshot, current: SymbolIndexSnapshot) -> DeltaResult` producing dataclass-backed `SymbolDeltaPayload` via `.to_payload()`.
   - Schema helper: `docs._scripts.validation.validate_against_schema(payload: Mapping[str, object], schema: Path, *, artifact: str) -> None` reused across writers and the new CLI (`docs/_scripts/validate_artifacts.py`).
   - JSON Schema files: `schema/docs/symbol-index.schema.json`, `schema/docs/symbol-delta.schema.json`, plus generated examples under `schema/examples/docs/`.

3. **Data/Schema Contracts**
   - `symbols.json` SHALL comply with `schema/docs/symbol-index.schema.json`, including `source_link`,
     `tested_by`, and `stability` metadata.
   - `symbols.delta.json` SHALL comply with `schema/docs/symbol-delta.schema.json`, documenting change entries
     with explicit `change_type` enumerations and optional problem detail references.
   - Validation SHALL run during artifact generation through the shared helper; failures MUST surface
     `ProblemDetails` envelopes via `tools._shared.problem_details` and block artifact writes as well as
     `make artifacts` completion.

4. **Test Plan**
   - Static analysis: `uv run ruff check docs`, `uv run pyrefly check docs`, `uv run pyright --warnings --pythonversion=3.13 docs`.
   - Schema validation: new `docs/_scripts/validate_artifacts.py` invoked directly and through `make artifacts`.


## Decisions

- Leverage existing `tools._shared.proc.run_tool` instead of reinventing subprocess wrappers, ensuring consistent logging.
- Use dataclasses/TypedDicts instead of msgspec initially, deferring more complex runtime dependencies.
- Reuse `tools._shared.metrics.observe_tool_run` and contextvars-based correlation IDs for observability so tooling stays
  aligned with the broader runtime stack.





