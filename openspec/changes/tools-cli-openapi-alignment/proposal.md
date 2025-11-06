## Why
The CLI OpenAPI generator recently gained structured configuration and context helpers, but the rest of the documentation/tooling surface (MkDocs diagram writers, registry loaders, docstring builders) still relies on ad-hoc dictionaries and loosely typed access. The drift forces each tool to reinvent metadata parsing, leads to mismatched `x-cli` extensions, and makes it difficult to guarantee that Problem Details, tag groups, and augment overrides stay consistent. Without a shared foundation junior contributors must reverse-engineer the generator before touching any related tooling.

## What Changes
- [ ] **ADDED**: capability spec describing a shared CLI tooling context module (`tools/_shared/cli_tooling.py`) that centralises augmentation loading, registry hydration, and operation context creation.
- [ ] **MODIFIED**: `tools/typer_to_openapi_cli.py` to consume the shared helpers, reduce bespoke loaders, and expose a stable API for other scripts.
- [ ] **MODIFIED**: MkDocs CLI diagram generator (`tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py`) and its façade to rely on the shared context rather than private imports.
- [ ] **MODIFIED**: handful of tests under `tests/tools/mkdocs_suite/` to exercise the new entrypoints and ensure diagrams track OpenAPI metadata.

## Impact
- **Capability surface**: introduces `tooling/cli_openapi` spec governing the shared configuration/context helpers for CLI-related documentation tooling.
- **Code paths**: touches `tools/_shared` (new module), `tools/typer_to_openapi_cli.py`, MkDocs CLI diagram scripts and facades, plus lightweight updates to associated tests.
- **Operational**: no runtime behaviour change; tooling becomes deterministic and easier to extend. Future phases can onboard additional scripts (docstring builder, registry exporters) onto the same primitives.

- [ ] Ruff / Pyright / Pyrefly remain clean across touched modules.
- [ ] CLI generator and MkDocs diagram share the same augment/tag/metadata handling.
- [ ] Regression tests cover both the shared helper module and the diagram script using the common context.

## Out of Scope
- Expanding observability, manifest schemas, or FAISS-specific enhancements (handled by separate specs).
- Refactoring other tooling (docstring builder, pipeline orchestration) until the shared foundation stabilises.

## Risks / Mitigations
- **Risk:** Centralising helpers could introduce circular imports.  
  **Mitigation:** house shared code under `tools/_shared/cli_tooling.py` with no reverse imports.
- **Risk:** Diagram script expectations may diverge.  
  **Mitigation:** add integration tests comparing diagram output before/after refactor.

## Alternatives Considered
- Leaving helpers embedded in `typer_to_openapi_cli.py` and duplicating logic in other scripts — rejected due to maintenance and type-safety burden.
- Generating diagrams directly from the OpenAPI file without shared context — rejected because we still need augment metadata resolution and registry enrichment for other tooling.
