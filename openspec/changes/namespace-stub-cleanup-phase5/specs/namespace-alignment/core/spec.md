## ADDED Requirements
### Requirement: Typed Namespace Registry
The system SHALL expose a typed namespace registry in `_namespace_proxy.py`, eliminating `Any` usage and ensuring `__getattr__` resolves only registered symbols.

#### Scenario: Registered symbol resolves with correct type
- **GIVEN** a symbol registered via the namespace registry with loader returning `SearchOptions`
- **WHEN** the namespace proxy resolves the symbol
- **THEN** the returned object is typed as `SearchOptions` and Mypy/Pyrefly report no `Any` leakage

#### Scenario: Missing symbol raises AttributeError
- **GIVEN** an unregistered symbol access
- **WHEN** `getattr` executes
- **THEN** it raises `AttributeError` with a helpful message listing available symbols

### Requirement: Stub Alignment with Runtime Exports
The system SHALL ensure stub packages mirror runtime exports with precise typing (`type[...]`, `Protocol`), without redundant ignores or `Any`.

#### Scenario: Stub exports match runtime symbols
- **GIVEN** `kgfoundry.agent_catalog.search` and its stub
- **WHEN** the parity check script runs
- **THEN** it confirms every public symbol is represented in the stub with matching type aliases and no `# type: ignore`

#### Scenario: Type checkers pass using stubs
- **GIVEN** Pyrefly and Mypy type checking sessions
- **WHEN** they analyze code depending on the stubs
- **THEN** no errors arise from missing exports or `Any` types in the stub files

### Requirement: Search Module Type Hygiene
The system SHALL remove redundant casts and `# type: ignore` directives from the search modules by tightening signatures and return types.

#### Scenario: Redundant cast eliminated
- **GIVEN** the `resolve_search_parameters` helper
- **WHEN** the refactored code executes
- **THEN** it returns typed structures without needing `cast(...)`, and unit tests confirm unchanged behavior

#### Scenario: Clean type checker output
- **GIVEN** the updated search modules
- **WHEN** Pyrefly and Mypy run
- **THEN** no `unused-ignore` or `redundant-cast` errors remain

