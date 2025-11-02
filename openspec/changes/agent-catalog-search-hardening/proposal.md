## Why
Search callers and tooling currently construct `SearchOptions` and `SearchDocument` inconsistently, leaving required dependencies unset, leaking `Any` types, and breaking schema parity across the CLI, client, and catalog builders. This undermines type safety, validation, and downstream documentation consumers.

## What Changes
- [x] **ADDED**: Central helper factories in `kgfoundry.agent_catalog.search` that accept explicit embedding/model loaders, candidate pool policies, alpha tuning, and return prevalidated `SearchOptions` objects with structured Problem Details on failure.
- [x] **ADDED**: Typed payload aliases (`SearchOptionsPayload`, `SearchDocumentPayload`) and validation routines that guarantee parity with JSON Schema definitions.
- [x] **MODIFIED**: `SearchDocument` / `SearchOptions` exports now surface only the documented helpers, shed `Any` annotations, and include docstrings with runnable examples.
- [x] **MODIFIED**: CLI, HTTP client, SQLite adapter, and docs builders rewritten to call the helpers, inject dependencies, and log structured errors.
- [x] **MODIFIED**: Search request/response JSON Schemas and OpenAPI fragments updated to mirror helper-produced payloads, with versioned examples.
- [ ] **REMOVED**: Legacy implicit defaults that silently injected `None` or untyped factories for model loaders.
- [ ] **RENAMED**: _None_
- [ ] **BREAKING**: Backward-compatible — existing callers receive equivalent defaults via the shared helper; no signature removals.

## Impact
- **Affected specs (capabilities):** `agent-catalog/search` (new requirements for helper coverage and schema parity)
- **Affected code paths:** `src/kgfoundry/agent_catalog/search.py`, `src/kgfoundry/agent_catalog/{cli,client,sqlite}.py`, `tools/docs/build_agent_catalog.py`, `tools/docs/observability.py`, `tools/docstring_builder/observability.py`, `stubs/kgfoundry/agent_catalog/search.pyi`
- **Data contracts:** `schema/agent_catalog/search-request.json`, `schema/agent_catalog/search-response.json`, OpenAPI components referenced by search endpoints
- **Rollout plan:** Ship alongside updated docs and migration note; provide sample helper usage to downstream teams; monitor catalog build pipeline before removing legacy defaults.

## Acceptance
- [ ] Helper factories deliver fully populated `SearchOptions` for CLI, HTTP client, SQLite, and docs builders with no missing dependencies.
- [ ] Typed payload aliases align with JSON Schema and eliminate `Any` usage in `kgfoundry.agent_catalog.search` and associated stubs.
- [ ] Search request/response schemas and OpenAPI components reflect the new payload structure and include updated examples.
- [ ] Agent Portal/doc artifacts regenerate without manual suppressions and remain schema-valid.

## Out of Scope
- FAISS/vectorstore interface alignment and GPU orchestration fixes (Phase 2)
- Complexity reductions or error taxonomy changes outside the search option/document surface

## Risks / Mitigations
- **Risk:** Helper defaults diverge from historical behavior and break hidden consumers.
  - **Mitigation:** Capture current defaults during inventory, document them in helper docstrings, and notify downstream teams of parameter overrides.
- **Risk:** Schema updates ripple through downstream docs builds.
  - **Mitigation:** Preview regenerated artifacts, validate schema round-trips, and coordinate with docs maintainers before merge.

## Alternatives Considered
- Consolidate logic inside the CLI/client without public helper exposure — rejected because tooling (docs builders, SQLite adapter) also needs consistent construction and a private helper would reintroduce duplication.
- Wrap all option/document typing inside Pydantic models — rejected for now to avoid coupling runtime validation into the hot search path; proposal keeps lightweight dataclasses with explicit schema validation hooks.

