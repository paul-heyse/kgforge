# Design

## Context

Augment and registry data now flow through a shared facade, but the payloads are still dictionaries. Downstream tooling needs to know which keys are guaranteed (`operations`, `tags`, `interfaces`, etc.), their types, and default behaviours. Without explicit models, contributors add manual coercions (`Mapping[str, object]`, `dict`, type ignores) and unit tests must re-validate shapes. We need structured data models with declarative validation to enforce shape, provide docs, and satisfy static analyzers.

## Goals

- Define **Pydantic** models (`BaseModel`, `RootModel`) that capture augment metadata (tag groups, operation overrides, CLI info) and registry interfaces.
- Update the facade to instantiate these models, performing validation/normalization (string key conversion, default values, Problem Details on failure).
- Ensure all downstream consumers depend on the typed models, eliminating raw dict access.

## Non-Goals

- Changing file formats beyond clarifying optional fields.
- Refactoring unrelated tooling that does not use augment metadata.

## Decisions

1. **Model technology** — use Pydantic v2 `BaseModel` subclasses with `model_config = ConfigDict(frozen=True)` for immutability, leveraging field validators/serializers for coercion.
2. **Model hierarchy**:
   - `OperationOverrideModel`: describes entries in `augment['operations']` (fields like `summary`, `description`, `tags`, `handler`, `env`, `code_samples`, `problem_details`, `x_extras`).
   - `TagGroupModel`: name, description, and ordered tags.
   - `AugmentMetadataModel`: contains `operations: dict[str, OperationOverrideModel]`, `tag_groups: tuple[TagGroupModel, ...]`, `raw_payload` (for debugging), and helper methods.
   - `RegistryInterfaceModel`: typed representation of interface metadata (id, module, owner, stability, binary, protocol, extras).
   - `RegistryMetadataModel`: mapping of interface IDs to `RegistryInterfaceModel`, with safe lookup helpers.
   - `ToolingMetadataModel`: aggregates augment + registry models and exposes convenience accessors.
3. **Validation strategy** — implement root validators to coerce keys to strings, normalize collections to tuples, and emit `AugmentRegistryValidationError` with Problem Details when input is invalid.
4. **Backward compatibility** — apply default values for optional fields, log warnings for deprecated keys, and expose `.model_dump()` for legacy consumers needing dicts.
5. **Helper API** — extend facade helpers (`load_tooling_metadata`) to return the new Pydantic models and provide convenience methods (e.g., `operation_override(operation_id, tokens)`, `interface(id)`).

## Detailed Plan

### 1. Define models

1. Introduce new Pydantic models in `tools/_shared/augment_registry.py`:
   - `OperationOverrideModel`: typed fields with validators ensuring sequences become tuples, optional fields default to `None`, and unknown `x-` attributes captured in `x_extras`.
   - `TagGroupModel`: validates names and tag lists, ensuring uniqueness and order preservation.
   - `AugmentMetadataModel`: top-level model containing path info, operations, tag groups, and raw payload snapshot for diagnostics.
   - `RegistryInterfaceModel`: typed interface metadata with optional fields, default descriptions, and extras dict.
   - `RegistryMetadataModel`: custom root model mapping interface IDs to `RegistryInterfaceModel`, exposing `.get(id)` with helpful errors.
   - `ToolingMetadataModel`: embeds augment + registry models and offers helper methods for consumers.
2. Implement custom exceptions `AugmentRegistryValidationError` (subclassing existing facade error) including `problem_details: ProblemDetailsPayload` property for consistent error reporting.

### 2. Update facade logic

1. Adjust `load_augment` to parse raw payloads, pass them into `AugmentMetadataModel.model_validate`, and capture validation errors to raise `AugmentRegistryValidationError` with structured Problem Details.
2. Adjust `load_registry` and `load_tooling_metadata` similarly, returning instances of the new Pydantic models.
3. Update caching (`functools.lru_cache`) to store the model instances and expose `clear_cache()` for tests.
4. Provide bridging helpers (`to_mutable_*`) only if necessary for legacy code, documenting their temporary nature.

### 3. Refactor consumers

1. Shared CLI tooling:
   - Update `load_cli_tooling_context` to depend on `ToolingMetadataModel`, replacing dict access with typed methods.
   - Expose typed properties (e.g., `context.metadata.augment.operations[...]`).
2. OpenAPI generator:
   - Replace dictionary indexing with calls to Pydantic model attributes (e.g., `override.tags`, `augment.tag_groups`).
   - Ensure docstrings/tests reflect the new typed API.
3. MkDocs scripts:
   - Use typed metadata when collecting operations and rendering diagrams (no raw `dict` operations).
4. Docstring builder tooling:
   - Update registry usage to call `ToolingMetadataModel.registry.interface(interface_id)`.

### 4. Testing & docs

1. Add Pydantic-focused unit tests verifying validation, default handling, serialization, and Problem Details output.
2. Regression tests for CLI generator and MkDocs diagrams ensuring outputs stay identical post-migration.
3. Document model fields and usage in module docstrings; include examples demonstrating error handling via `.model_validate` exceptions.

## Risks & Mitigations

- **Strict validation rejects real-world data** — run models against current augment/registry files; log warnings for deprecated shapes and document migration path.
- **Pydantic dependency weight** — already in project for other tooling; ensure optional features (like JSON schema generation) remain disabled unless needed.

## Migration

1. Implement Pydantic models and update facade; ensure validation errors convert to Problem Details.
2. Update all consumers (shared CLI tooling, OpenAPI generator, MkDocs scripts, docstring tooling) to use typed models; remove bool/dict casts.
3. Refresh tests/docs and run quality gates to confirm lint/type/test suites remain green.
