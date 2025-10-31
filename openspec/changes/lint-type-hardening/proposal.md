## Why
Ruff identifies 709 violations (pathlib misuse, blind exceptions, security flags), and mypy reports 706 errors caused by pervasive `Any` types, unfollowed imports, and protocol mismatches. These issues block dependable releases, hide real defects, and prevent downstream agents from trusting the platform.

## What Changes
- [ ] **ADDED**: `quality-hardening` capability defining seventeen requirements (R1–R17) covering pathlib standardization, typed exception taxonomy with Problem Details, secure serialization/SQL, schema-driven models, vector search protocols, typed Parquet IO, typed configuration, structured observability, docstring/public API hygiene, strict tooling enforcement, concurrency/context propagation, performance budgets, documentation/discoverability, packaging integrity, security/supply-chain hardening, idempotency/retries, and file/time/number hygiene.
- [ ] **BREAKING**: None anticipated; configuration tightening may surface misconfigured environments early but will ship with mitigation guidance.

## Impact
- **Affected specs (capabilities):** `quality-hardening` (new)
- **Affected code paths:** `kgfoundry_common.{fs,errors,logging,config,parquet_io}`, `kgfoundry/agent_catalog/*`, `search_api/*`, `search_client`, `vectorstore_faiss`, `embeddings_sparse/*`, `registry/*`, orchestration/download flows.
- **Data contracts:** New/updated JSON Schemas for agent sessions, MCP envelopes, registry records, observability metrics, Problem Details, plus refreshed OpenAPI 3.2 for `search_api`.
- **Rollout plan:** Execute Phase 1 (foundation) first: land shared helpers/codemods (pathlib, taxonomy + registry, serialization helpers, schema/model scaffold, protocols, RuntimeSettings, observability adapter, tooling automation, namespace consolidation, GPU extras). Once the clean baseline is achieved and exception codes are frozen, proceed to Phase 2 (adoption) migrating module clusters with the codemods, running requirement acceptance gates after each cluster and attaching logs (codemod, import-linter, suppression checks) to PR summaries.

## Acceptance
- [ ] Scenarios under `specs/quality-hardening/spec.md` pass as written (tests + validation scripts).
- [ ] Quality gates all green (Ruff, pyrefly, mypy, pytest, artifacts) with zero outstanding suppressions in `src/**`.
- [ ] Problem Details example validated and referenced in docs/tests; structured observability verified via automated checks.

## Out of Scope
- Docstring reformatting / indentation fixes (covered by `docstring-builder-hardening`).
- Governance/process documentation updates beyond existing AGENTS.md guidance.
- New product surface area beyond the hardening work described.

## Risks / Mitigations
- **Risk:** Tightened configuration validation may break existing deployments lacking required env vars.
  - **Mitigation:** Provide migration guide + defaults, roll out behind staged branch with smoke tests.
- **Risk:** Replacing serialization/SQL pathways could introduce regression in data compatibility.
  - **Mitigation:** Add regression fixtures, checksum validation, and fallback loaders for legacy artifacts during migration.
- **Risk:** Increased schema strictness could slow development.
  - **Mitigation:** Supply helper generators/tests and document workflows in the PR checklist.

## Alternatives Considered
- **Status quo + selective suppressions:** Rejected; perpetuates technical debt and security gaps.
- **Adopt third-party scaffolding tools for migrations:** Rejected for now; bespoke helpers keep scope contained and align with existing infrastructure.

