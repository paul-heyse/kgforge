# Design

## Context

`tools/typer_to_openapi_cli.py` now encapsulates CLI metadata in `CLIConfig` and `OperationContext`, yet other documentation scripts (MkDocs CLI diagrams, augment registry tools) copy fragments of logic, bypass the new dataclasses, and operate on loose dictionaries. This makes the tooling brittle: augment overrides can fall out of sync, structured extensions drift, and tests must build bespoke fixtures. We need a shared foundation so every CLI-adjacent tool consumes the same typed primitives and converges on the OpenAPI generator’s contract.

## Goals

- Provide a reusable module that loads augment metadata, registry interfaces, and operation contexts once, returning typed objects for all CLI tooling.
- Refactor the CLI generator and MkDocs diagram script to depend exclusively on the shared module, eliminating bespoke loaders.
- Deliver comprehensive documentation and tests so future tooling can adopt the helpers without rediscovering implementation details.

## Non-Goals

- Changing the OpenAPI schema or introducing new CLI behaviour.
- Refactoring unrelated tooling such as docstring builders, FAISS orchestration, or schema validators (reserved for subsequent proposals).

## Decisions

1. **Shared module location** — create `tools/_shared/cli_tooling.py` (importable by both generator and MkDocs scripts) exporting:
   - `AugmentConfig` dataclass (path, payload, tag groups).
   - `RegistryContext` dataclass (interface metadata resolved from `api_registry.yaml`).
   - `load_cli_tooling_context()` returning a composite `CLIToolingContext` with validated augment + registry data and a ready-to-use `CLIConfig`.
2. **File system strategy** — adopt the safe-IO wrappers already used in docs tooling (`safe_json_deserialize`) for reading augment/registry files, returning helpful Problem Details on failure.
3. **Deterministic caching** — memoise augmentation parsing within the module using an LRU keyed by path to avoid redundant IO when multiple scripts run in the same process (MkDocs builds).
4. **Typed API** — reuse the existing `CLIConfig`/`OperationContext` dataclasses, but expose factory functions in the shared module so consumers need not import the generator directly.
5. **Testing approach** — enhance `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` to use the shared context and add unit tests for `cli_tooling.load_cli_tooling_context` covering augment overrides, missing files, and registry lookups.

## Detailed Plan

### 1. Shared tooling module

1. Add `tools/_shared/cli_tooling.py` containing:
   - `AugmentConfig` (`path: Path`, `payload: AugmentPayload`, `tag_groups: list[dict[str, object]]`).
   - `RegistryContext` (`interfaces: dict[str, JSONMapping]`).
   - `CLIToolingContext` (`augment: AugmentConfig`, `registry: RegistryContext`, `cli_config: CLIConfig`).
   - Helper functions `load_augment_config(path: Path)`, `load_registry_context(path: Path)`, and `build_cli_config(augment, registry, interface_id, bin_name, title, version)` returning `CLIConfig` seeded with OperationContext.
   - `load_cli_tooling_context(settings: CLIToolSettings)` where `CLIToolSettings` captures CLI args (paths, defaults) used by both generator and diagram scripts.
2. Implement robust validation:
   - Ensure augment payloads are dictionaries; convert non-string keys via `str`. Raise `CLIConfigError` with Problem Details when the file is missing or malformed.
   - For registry lookups, default to an empty mapping but log warnings via `LOGGER.warning` when the requested interface is unknown.
3. Provide optional dependency injection for file readers/writers to support testing (pass alternate `read_json` callable).

### 2. Refactor CLI generator

1. Update `tools/typer_to_openapi_cli.py` to:
   - Import `load_cli_tooling_context` and `CLIToolSettings`.
   - Replace inline augment/registry parsing with the shared loader (remove `_load_registry`).
   - Use the returned `CLIConfig` directly in `make_openapi` call.
   - Ensure error propagation remains consistent (Problem Details for load failures).
2. Adjust CLI argument parsing to populate `CLIToolSettings` (bin, title, version, augment path, registry path, interface ID).
3. Update unit tests (if any) to account for the new abstraction.

### 3. Align MkDocs CLI diagram generator

1. Modify `tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py` and its façade to:
   - Import the shared loader and call `load_cli_tooling_context` to obtain operations.
   - Replace private augment parsing with `context.cli_config.operation_context` when collecting operations.
   - Ensure diagram generation uses consistent tag groups and `x-cli` metadata.
2. Update `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` to:
   - Patch the shared loader rather than private module internals.
   - Exercise the new path when augment metadata or tag groups are absent.

### 4. Documentation & discoverability

1. Document the shared helpers in module docstrings (NumPy style) with runnable examples showing how tooling scripts should create contexts.
2. Update `tools/README.md` (if present) or relevant developer docs to highlight the new module and migration path for other tools.
3. Capture Problem Details expectations when files are missing or malformed.

## Risks & Mitigations

- **Circular dependencies** — ensure the shared module depends only on neutral helpers; avoid importing MkDocs or Typer modules directly. Use string annotations to defer typing imports.
- **Performance regressions** — memoise context loaders so repeated invocations during MkDocs builds don’t re-read files.
- **Test brittleness** — offer dependency injection (overriding file readers) to keep tests deterministic.

## Migration

1. Implement the shared module and migrate the CLI generator; confirm existing tests and linting pass.
2. Migrate MkDocs diagram tooling and adjust tests.
3. Update documentation and finalise acceptance criteria.
