## ADDED Requirements
### Requirement: Typed CLI Metadata Contracts
CLI tooling SHALL rely on Pydantic data models for augment metadata and interface registry entries so all consumers receive validated, canonical structures with documented fields.

#### Scenario: Facade returns typed augment metadata
- **GIVEN** tooling loads augment metadata via the shared facade
- **WHEN** `load_augment` executes
- **THEN** it returns an `AugmentMetadataModel` instance derived from `pydantic.BaseModel` whose operations and tag groups expose typed accessors and validated defaults

#### Scenario: Registry metadata exposed through models
- **GIVEN** a tooling script requests interface metadata
- **WHEN** it uses `RegistryMetadataModel`
- **THEN** it receives `RegistryInterfaceModel` instances with documented fields (`module`, `owner`, `stability`, etc.) and no longer manipulates untyped dicts

#### Scenario: Validation errors surface Problem Details
- **GIVEN** the augment payload misses required keys or uses invalid types
- **WHEN** the facade attempts to build models
- **THEN** it raises `AugmentRegistryValidationError` (backed by Pydantic validation errors) with an RFC 9457 Problem Details payload that scripts can render consistently

#### Scenario: Downstream tools pass static analysis
- **GIVEN** Ruff, Pyright, and Pyrefly runs across CLI tooling modules
- **WHEN** Pydantic metadata contracts are in place
- **THEN** analyzers succeed without `Mapping[str, object]` suppressions or cast-heavy code paths
