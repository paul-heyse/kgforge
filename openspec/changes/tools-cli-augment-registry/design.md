# Design

## Context

Augment metadata (`openapi/_augment_cli.yaml`) and interface registries (`mkdocs_suite/api_registry.yaml`) are consumed by multiple tooling entrypoints. Each script reads the files directly, coerces keys to strings, and handles missing data differently. This duplication leads to inconsistent defaults, scattered error handling, and frequent type-checking suppressions. A dedicated facade can centralize IO, validation, caching, and Problem Details emission so every tool interacts with the same normalized structures.

## Goals

- Provide a single module that loads augment and registry data, validates structure, and returns immutable context objects.
- Ensure all CLI tooling (OpenAPI generator, MkDocs scripts, docstring builder utilities) depend on the facade to obtain augment overrides, tag groups, and interface metadata.
- Expose consistent, testable behaviour for error cases (missing files, malformed YAML) via RFC 9457 Problem Details.

## Non-Goals

- Changing the content of augment or registry files beyond normalization.
- Refactoring unrelated tooling that does not use augment metadata.

## Decisions

1. **Facade module** — add `tools/_shared/augment_registry.py` housing dataclasses (`AugmentData`, `RegistryData`, `ToolingMetadata`), custom exceptions (`AugmentRegistryError`), and loader functions (`load_augment`, `load_registry`, `load_tooling_metadata`).
2. **Immutable outputs** — dataclasses expose read-only views (tuples, frozen mappings) to prevent accidental mutation and to make caching deterministic.
3. **Problem Details** — on IO or validation failure, the facade raises `AugmentRegistryError` carrying a Problem Details dict suitable for printing/logging.
4. **Caching** — implement `functools.lru_cache` keyed by `(augment_path, registry_path)` to avoid redundant deserialization during long-running MkDocs builds.
5. **Hook support** — allow optional adapters (e.g., transform functions) so future tools can enrich metadata without forking the loader.

## Detailed Plan

### 1. Facade implementation

1. Create `tools/_shared/augment_registry.py` with:
   - `AugmentData`: frozen dataclass containing `path`, `raw_payload`, `operations`, `tag_groups`.
   - `RegistryData`: frozen dataclass containing `path`, `interfaces` (mapping of interface IDs to metadata dicts).
   - `ToolingMetadata`: frozen dataclass bundling `augment` and `registry` plus helper methods (`get_operation_override(op_id, tokens)`, `get_interface_meta(interface_id)`).
   - Helper functions `load_augment(path: Path, *, reader=_DEFAULT_JSON_READER)`, `load_registry(path: Path, *, reader=...)`, and `load_tooling_metadata(augment_path: Path, registry_path: Path)`.
2. Validation rules:
   - Augment payload must be a dict; convert keys to strings; ensure `operations` is a dict mapping string keys to dict overrides.
   - Tag groups must resolve to a `list[dict[str, object]]`; drop invalid entries with warning logs.
   - Registry file must contain an `interfaces` mapping; unknown structures raise `AugmentRegistryError` with Problem Details (`type`, `title`, `status`, `detail`, `instance`).
3. Logging & Problem Details:
   - Use `LOGGER` from module root; attach `extra` context (`paths`, `error`).
   - Provide `render_problem_details(error)` helper returning JSON string for CLI printing.
4. Caching:
   - Decorate `load_tooling_metadata` with `functools.lru_cache(maxsize=16)`; provide `clear_cache()` for tests.
5. Testing hooks:
   - Accept optional `reader` callables for augment/registry loads, enabling in-memory fixtures in unit tests.

### 2. Integrate with existing tooling

1. Update shared CLI tooling module to import `load_tooling_metadata` and wrap it in `load_cli_tooling_context`, removing duplicate file logic.
2. Refactor `tools/typer_to_openapi_cli.py` to:
   - Replace direct augment/registry parsing with `load_tooling_metadata`.
   - Derive `CLIConfig` from the returned metadata (reusing `OperationContext`).
3. Modify MkDocs CLI diagram and docstring builder utilities to:
   - Call the shared loader and derive required information (tags, operations, interface metadata) from `ToolingMetadata`.
   - Remove bespoke YAML loading helpers.

### 3. Documentation & errors

1. Document the facade in module docstrings with examples showing success and failure cases (including Problem Details output).
2. Update relevant developer docs or README entries to instruct contributors to use `load_tooling_metadata` instead of raw JSON loads.

## Risks & Mitigations

- **Cache staleness** — long-lived processes may need to reload files when augment/registry changes.  
  *Mitigation:* expose `clear_cache()` and allow passing `force_reload=True` in loaders.
- **Consumer migration** — tooling may depend on mutable dictionaries.  
  *Mitigation:* provide `.to_mutable()` helpers or document copying patterns.

## Migration

1. Implement the facade module and swap the shared CLI tooling to use it; ensure unit tests cover success/failure paths.
2. Update the OpenAPI generator and MkDocs scripts to rely on the new API.
3. Adjust docstring builder (or other tooling) to adopt the facade.
4. Refresh tests, docs, and run lint/type/test gates.
