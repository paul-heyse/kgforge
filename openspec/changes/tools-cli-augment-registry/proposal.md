## Why
Even with shared CLI operation contexts in place, augment metadata and interface registries are still parsed through multiple bespoke paths (OpenAPI generator, MkDocs scripts, docstring builder tooling, pipeline utilities). Each implementation performs its own JSON loading, falls back differently on missing keys, and surfaces inconsistent error messages. This duplication risks divergent behaviour, makes unit testing cumbersome, and leaves junior developers uncertain which helper to rely on.

## What Changes
- [ ] **ADDED**: capability spec defining a canonical augment/registry facade that normalizes payloads, enforces types, and exposes structured errors.
- [ ] **MODIFIED**: shared CLI tooling module to depend on the facade instead of raw JSON readers.
- [ ] **MODIFIED**: OpenAPI generator, MkDocs CLI scripts, and docstring builder tooling to import the facade for a unified view of augment overrides, tag groups, and interface metadata.
- [ ] **MODIFIED**: Tests across tooling suites to assert shared behaviour and Problem Details responses for missing or malformed inputs.

## Impact
- **Capability surface**: adds `tooling/augment_registry` spec documenting guarantees for augment and registry access.
- **Code scope**: new module under `tools/_shared/augment_registry.py` plus refactors in `tools/typer_to_openapi_cli.py`, `tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py`, `tools/docstring_builder` helpers, and associated tests.
- **Operational**: no production runtime changes; documentation tooling becomes easier to extend and reason about.

- [ ] Ruff / Pyright / Pyrefly clean for all touched modules.
- [ ] All CLI tooling paths load augment & registry data exclusively through the facade.
- [ ] Problem Details emitted by the facade are consistent across consumers.

## Out of Scope
- Expanding manifest schemas, FAISS factories, or other observability work covered by separate specs.
- Migrating unrelated tooling (e.g., orchestration pipelines) that do not rely on augment metadata.

## Risks / Mitigations
- **Risk:** Tooling that needs bespoke augment behaviour may resist the shared API.  
  **Mitigation:** expose optional hooks (callbacks, adapter interfaces) while keeping defaults consistent.
- **Risk:** Introducing the facade could mask existing assumptions around mutable dictionaries.  
  **Mitigation:** return immutable dataclasses and provide conversion helpers so consumers adapt gradually.

## Alternatives Considered
- Leaving each tool to manage augment data independently — rejected due to ongoing drift and duplicated validation logic.
- Generating Python modules from augment/registry YAML during docs build — rejected for now to keep the workflow lightweight.
