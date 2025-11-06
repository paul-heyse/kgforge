## Why
The CLI tooling recently gained a shared facade for augment/registry data, but payloads still flow through loose dictionaries. Callers routinely coerce values with `Mapping[str, object]`, annotate everything as `dict`, and rely on runtime shape assumptions. This ambiguity frustrates static analyzers, obscures missing fields at runtime, and makes it difficult for contributors to determine which keys are required. We need explicit metadata contracts—backed by **Pydantic** models—so every tool receives canonical, validated structures.

## What Changes
- [ ] **ADDED**: capability spec defining Pydantic models (`AugmentMetadataModel`, `RegistryMetadataModel`, etc.) covering augment payloads and interface metadata.
- [ ] **MODIFIED**: augment/registry facade to instantiate and return the Pydantic models, performing field-level validation and normalization.
- [ ] **MODIFIED**: shared CLI tooling, OpenAPI generator, MkDocs scripts, and docstring tooling to depend on the Pydantic models rather than raw dicts.
- [ ] **MODIFIED**: tests to assert validation behaviour and confirm static typing satisfaction.

## Impact
- **Capability surface**: adds `tooling/cli_metadata_contracts` spec clarifying required fields, optional overrides, and error semantics.
- **Code**: updates `tools/_shared/augment_registry.py`, `tools/_shared/cli_tooling.py`, and all consumers to use the new dataclasses/pydantic models.
- **Testing**: new unit tests ensuring invalid augment data is rejected with precise Problem Details.

- [ ] Ruff / Pyright / Pyrefly clean across modified modules.
- [ ] All tooling retrieves instances of typed metadata objects (no raw dictionary access).
- [ ] Validation errors surface with structured Problem Details; successful loads provide canonical shapes.

## Out of Scope
- Altering existing augment/registry file semantics beyond ensuring optional fields are documented.
- Introducing runtime schema migrations (future work if file formats evolve).

## Risks / Mitigations
- **Risk:** Rigid models might reject currently tolerated loose data.  
  **Mitigation:** provide backward-compatible defaults, conversions, and deprecations documented in the spec.
- **Risk:** Choosing between dataclasses and pydantic may add dependency weight.  
  **Mitigation:** prefer stdlib dataclasses with manual validation unless pydantic features are essential; evaluate performance impact.

## Alternatives Considered
- Continuing to rely on `TypedDict` or `Mapping[str, object]` annotations — rejected because they do not enforce runtime invariants and still require bespoke casting.
- Serializing augment/registry into generated Python modules — deferred until we understand real-world scale and mutation frequency.
