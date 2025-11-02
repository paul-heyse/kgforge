## ADDED Requirements
### Requirement: Typed Search Options Helpers
The system SHALL expose documented helper functions for constructing `SearchOptions` that always populate required dependencies (alpha, embedding model, model loader, candidate pool, batch size, facets), emit structured Problem Details errors on invalid input, and provide runnable examples demonstrating typical usage.

#### Scenario: CLI search uses helper
- **GIVEN** the CLI requests a faceted search with user-provided limits
- **WHEN** the CLI builds options via the public helper
- **THEN** the resulting `SearchOptions` instance includes validated facets, candidate pool defaults, and injected model loader references without emitting type checker errors

#### Scenario: HTTP client wires helpers with custom loaders
- **GIVEN** the HTTP client receives a request overriding `alpha` and supplying a bespoke embedding loader
- **WHEN** the client builds options via the helper
- **THEN** the helper returns a fully populated `SearchOptions` object whose repr documents the injected loader and alpha, and Pyrefly/Mypy report zero missing-argument errors

#### Scenario: Missing dependency emits problem details
- **GIVEN** a caller omits the embedding model factory when building options
- **WHEN** the helper validates dependencies
- **THEN** it raises a domain-specific exception that surfaces an RFC 9457 Problem Details payload including `type`, `title`, `status`, and `detail`

#### Scenario: Invalid facet name rejected early
- **GIVEN** a caller supplies a facet key not present in the schema allow-list
- **WHEN** the helper validates the facets mapping
- **THEN** it raises a `CatalogSearchConfigurationError` identifying the bad key and no request is issued downstream

### Requirement: Search Document Schema Parity
The system SHALL ensure `SearchDocument` instances, schemas, type stubs, and generated artifacts remain in lockstep, providing explicit JSON Schema definitions, deterministic serialization order, and round-trip validation coverage.

#### Scenario: Schema round-trips through helper
- **GIVEN** a catalog result with package, module, qname, docstring, and metadata facets
- **WHEN** it is converted to a `SearchDocument` and serialized via the schema codec
- **THEN** the JSON output validates against the 2020-12 schema and deserializes back to an identical document

#### Scenario: Docs builder validates schema alignment
- **GIVEN** the docs tooling generates catalog artifacts using the helper
- **WHEN** `make artifacts` runs
- **THEN** the generated Agent Portal index validates against the updated schema without suppressions or manual ignores

#### Scenario: Stubs mirror public payloads
- **GIVEN** the type stubs for `kgfoundry.agent_catalog.search`
- **WHEN** Pyrefly and Mypy ingest the stubs
- **THEN** the payload aliases and helper signatures match the implementation exactly, exposing no `Any` types and allowing downstream tooling to type-check successfully

