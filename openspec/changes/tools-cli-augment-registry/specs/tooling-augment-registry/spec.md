## ADDED Requirements
### Requirement: Canonical Augment and Registry Facade
Tooling SHALL consume augment metadata and registry interfaces through a shared facade that validates structure, normalizes payloads, and emits RFC 9457 Problem Details on error.

#### Scenario: Shared loader populates CLI generator
- **GIVEN** the OpenAPI generator runs with default augment and registry paths
- **WHEN** it builds the tool context
- **THEN** it calls `tools._shared.augment_registry.load_tooling_metadata(...)` and no longer performs bespoke JSON parsing

#### Scenario: MkDocs scripts reuse facade
- **GIVEN** the MkDocs CLI diagram script collects operations
- **WHEN** it requests augment overrides and tag groups
- **THEN** they originate from the shared facade and match the generator’s view of the metadata

#### Scenario: Docstring builder accesses registry via facade
- **GIVEN** docstring builder tooling requires interface metadata
- **WHEN** it loads the registry
- **THEN** it invokes the facade and receives normalized interface definitions without duplicating IO logic

#### Scenario: Errors emit Problem Details
- **GIVEN** the augment file is missing or malformed
- **WHEN** tooling calls the facade
- **THEN** it raises `AugmentRegistryError` carrying a Problem Details payload with `type`, `title`, `status`, `detail`, and `instance`, allowing scripts to render consistent error messages
