## Context
- CLI (`src/kgfoundry/agent_catalog/cli.py`) builds search payloads inline, passing only facets and `k`, leaving alpha, embedding model, and batch size unset.
- HTTP client (`src/kgfoundry/agent_catalog/client.py`) mirrors the CLI pattern, forwarding user facets and defaults but never wires required loader factories.
- SQLite adapter (`src/kgfoundry/agent_catalog/sqlite.py`) pulls catalog rows directly into `SearchDocument` without schema validation.
- Docs builders (`tools/docs/build_agent_catalog.py`, `tools/docs/observability.py`, `tools/docstring_builder/observability.py`, `tools/navmap/observability.py`) duplicate construction logic and suppress type errors.
- Type stubs (`stubs/kgfoundry/agent_catalog/search.pyi`) expose `Any` signatures, blocking Pyrefly/Mypy and letting schema drift go unnoticed.

## Goals / Non-Goals
- **Goals**
  - Provide a typed public surface for constructing search options/documents with consistent defaults and validation hooks.
  - Align JSON Schema/OpenAPI definitions with the runtime structures and ensure builder/test parity.
  - Preserve backward compatibility for existing CLI/client consumers by routing through the new helpers.
- **Non-Goals**
  - Overhauling FAISS/vectorstore adapters (Phase 2).
  - Reducing algorithmic complexity in the search pipeline beyond helper extraction.
  - Reworking broader agent-catalog RBAC or session flows.

## Decisions
- Introduce helper factories (`build_default_search_options`, `build_faceted_search_options`, `build_embedding_aware_search_options`) inside `kgfoundry.agent_catalog.search` with full type signatures and docstrings referencing the JSON Schema IDs.
- Define payload structures (`type SearchOptionsPayload = TypedDict(...)`, `type SearchDocumentPayload = TypedDict(...)`) to enforce field completeness and drive schema generation.
- Add validation routines that ensure facets are recognized, candidate pools satisfy minimum thresholds, alpha ∈ [0.0, 1.0], and embedding/model loader factories are present; violations raise `CatalogSearchConfigurationError` carrying Problem Details metadata.
- Update all core call sites (CLI, client, SQLite, docs builders) to call the helpers and pass explicit dependency providers (embedding model loader, candidate pool calculators) to avoid global state.
- Expand type stubs to mirror the public helpers and payload aliases, removing `Any` and documenting return types.
- Document helper usage in module-level docstrings and intro examples for Agent Portal consumers.

## Alternatives
- **Inline validation in each caller** — rejected; would perpetuate duplication and drift.
- **Adopt Pydantic models for runtime validation** — deferred to avoid runtime overhead in tight search loops and to keep validation explicit.

## Risks / Trade-offs
- Helper defaults might conceal missing dependency wiring.
  - Mitigation: Require explicit factory arguments in helper signatures; add failing tests when omitted; log structured warnings when defaults kick in.
- Schema adjustments could break downstream tooling.
  - Mitigation: Regenerate artifacts and run targeted smoke tests (CLI search, docs build) before merge; coordinate with docs owners.
- Potential performance regression from additional validation.
  - Mitigation: Keep validation lightweight (pure Python checks), add micro-bench to confirm negligible overhead.

## Migration
- Refactor existing call sites to use helpers in one pull request.
- Add deprecation warnings to legacy constructors (if any) with clear removal timeline.
- Publish internal doc snippet demonstrating helper usage for CLI/client teams.
- Monitor telemetry (logs/metrics) post-deploy to ensure no spike in Problem Details responses.

