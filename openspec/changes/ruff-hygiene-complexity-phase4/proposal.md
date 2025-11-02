## Why
Large swaths of the codebase exceed Ruff’s complexity thresholds, rely on `except Exception`, defer imports for no documented reason, and still reference deprecated typing aliases and `os.path`. These issues hide bugs, make the system harder to reason about, and keep lint/type gates from going green.

## What Changes
- [x] **MODIFIED**: High-complexity functions (`load_catalog_payload`, `_attach_symbols_to_modules`, `index_bm25`, `index_faiss`, catalog search hot paths) refactored into composable helpers and typed dataclasses/Protocols.
- [x] **MODIFIED**: Exception handling updated across these modules to use a scoped taxonomy (`CatalogLoadError`, `IndexBuildError`, etc.) with documented `raise ... from e` semantics.
- [x] **MODIFIED**: Deferred/inline imports replaced with top-level imports; where late imports are unavoidable, annotated with rationale and tests.
- [x] **MODIFIED**: Pickle usage wrapped with defensive allow-list checks or replaced with safer serialization APIs.
- [x] **MODIFIED**: Deprecated typing aliases removed, private import violations eliminated, and filesystem interactions migrated to `pathlib`.
- [ ] **REMOVED**: Legacy helpers superseded by the new pure functions and any unused imports uncovered during refactor.

## Impact
- **Affected specs (capabilities):** `ruff-hygiene/core`
- **Affected code paths:** `src/kgfoundry/agent_catalog/models.py`, `src/kgfoundry/agent_catalog/sqlite.py`, `src/orchestration/cli.py`, `src/embeddings_sparse/{bm25,splade}.py`, `src/search_api/{bm25_index,fixture_index,splade_index}.py`, plus related utilities/tests
- **Data contracts:** None directly, but refactors must preserve schema outputs for catalog/index artifacts
- **Rollout plan:** Implement module by module, running Ruff complexity checks after each; communicate exception taxonomy changes to downstream teams; monitor CLI/indexing jobs post-deploy.

## Acceptance
- [ ] Complexity metrics (Ruff `C901`, `PLR091x`) eliminated across targeted modules.
- [ ] New exception classes documented and used consistently, with regression tests proving correct handling.
- [ ] No remaining `except Exception` without justification; pickle usage audited with allow-list tests.
- [ ] Ruff, Pyrefly, Mypy, and pytest pass with no new ignores; docs/build artifacts remain unchanged apart from intentional refactors.

## Out of Scope
- Performance optimizations beyond what the refactors naturally deliver.
- Rewriting algorithms outside the targeted complexity hotspots.
- Broader security reviews (handled in other phases).

## Risks / Mitigations
- **Risk:** Refactoring complex functions may introduce regressions.
  - **Mitigation:** Add characterization tests before refactoring; use parameterized fixtures to ensure feature parity.
- **Risk:** Changing exception types may confuse downstream error handlers.
  - **Mitigation:** Maintain subclassing hierarchy, document taxonomy, and update documentation with migration notes.
- **Risk:** Switching pickle behavior may break legacy artifacts.
  - **Mitigation:** Provide allow-listed serializer wrapper with compatibility mode; add tests loading existing fixtures.

## Alternatives Considered
- Incremental lint suppressions — rejected; fails to address root causes.
- Introducing third-party serialization formats — deferred; focus is on safety wrappers first.
- Leaving imports deferred — rejected to keep module boundaries explicit and measurable.

