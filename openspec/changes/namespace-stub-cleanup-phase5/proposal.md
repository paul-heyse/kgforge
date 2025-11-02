## Why
The namespace proxy and stub packages still rely on `Any` sentinels, stale exports, and redundant ignores that block strict Pyrefly/Mypy runs. Aligning runtime modules with their stubs is essential to maintain type safety and developer ergonomics.

## What Changes
- [x] **MODIFIED**: `_namespace_proxy.py` rewritten to use typed registries or lazy loaders without `Any`, ensuring exported symbols are tracked explicitly.
- [x] **MODIFIED**: Stub packages (e.g., `stubs/kgfoundry/agent_catalog/search.pyi`) updated to mirror runtime exports with `type[...]` aliases, eliminating `Any` annotations and redundant `# type: ignore`.
- [x] **MODIFIED**: Search modules cleaned of redundant casts and ignores, with helper functions typed precisely to avoid fallback to `Any`.
- [ ] **REMOVED**: Obsolete stub entries or namespace proxies no longer necessary post-alignment.

## Impact
- **Affected specs (capabilities):** `namespace-alignment/core`
- **Affected code paths:** `src/kgfoundry/_namespace_proxy.py`, `stubs/kgfoundry/**`, `src/kgfoundry/agent_catalog/search.py`, `src/kgfoundry/agent_catalog/{cli,client}.py`
- **Data contracts:** None directly, but consistent exports are required for documentation generation and tooling.
- **Rollout plan:** Update runtime modules first, then stubs; run Pyrefly/Mypy to confirm alignment before merging.

## Acceptance
- [ ] `_namespace_proxy.py` exposes only explicitly registered symbols with typed registries; no `Any` usage remains.
- [ ] Stub files match runtime exports exactly, using `type[...]` or `Protocol` definitions with zero redundant ignores.
- [ ] Search modules compile without redundant casts or `# type: ignore`; Pyrefly and Mypy pass cleanly with strict settings.
- [ ] Ruff lint (`ANN401`, `RUF100`) reports no issues in stubs or namespace files.

## Out of Scope
- Broader refactors of search logic (handled in other phases).
- Introducing new API surface beyond aligning existing exports.
- Automated stub generation (manual curation only for now).

## Risks / Mitigations
- **Risk:** Removing `Any` may reveal hidden typing issues.
  - **Mitigation:** Incrementally add precise types and update call sites/tests to accommodate stricter typing.
- **Risk:** Misaligned stubs could break downstream tooling if export names change.
  - **Mitigation:** Double-check export lists, run Pyrefly/Mypy against stubs, and communicate any renamed symbols.
- **Risk:** Lazy loader changes may alter import timing.
  - **Mitigation:** Maintain compatibility by preserving import semantics, add tests verifying lazy loader behavior.

## Alternatives Considered
- Generating stubs automatically via `stubgen` — rejected; existing stubs require curated typing for complex generics.
- Leaving `_namespace_proxy.py` as-is with ignores — rejected because it blocks strict typing goals.

