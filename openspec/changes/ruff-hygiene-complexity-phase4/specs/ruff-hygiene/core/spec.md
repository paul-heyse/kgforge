## ADDED Requirements
### Requirement: Targeted Exception Handling
The system SHALL replace all `except Exception` blocks within the targeted modules with taxonomy-aligned exceptions, each documented and chained via `raise ... from e`.

#### Scenario: Catalog load surfaces typed error
- **GIVEN** a malformed catalog payload encountered in `load_catalog_payload`
- **WHEN** the loader refactor executes
- **THEN** it raises `CatalogLoadError` chained from the underlying parsing exception, and the structured log records `error_type="CatalogLoadError"`

#### Scenario: Index builder reports specific failure
- **GIVEN** an IO failure during `index_bm25`
- **WHEN** the exception is caught
- **THEN** `IndexBuildError` is raised with cause attached, and no bare `except Exception` blocks exist in the module

### Requirement: Reduced Cyclomatic Complexity
The system SHALL break down high-complexity functions flagged by Ruff into smaller helpers, ensuring each helper stays within Ruff thresholds and exposes typed inputs/outputs.

#### Scenario: Catalog payload parsing helpers
- **GIVEN** the refactored `agent_catalog.models`
- **WHEN** Ruff checks run
- **THEN** no `C901`/`PLR091x` violations remain, and helper functions have explicit return types and docstrings

#### Scenario: Orchestration index helpers
- **GIVEN** the updated `orchestration.cli`
- **WHEN** `index_bm25` and `index_faiss` execute
- **THEN** they orchestrate newly extracted helpers (`build_bm25_index`, `train_faiss_index`) and maintain behavioral parity verified by tests

### Requirement: Safe Serialization & Path Hygiene
The system SHALL wrap pickle usage with allow-list guards (or replace it) and ensure filesystem code uses `pathlib.Path` with documented behavior.

#### Scenario: Pickle allow-list rejects unknown payload
- **GIVEN** a malicious pickle file presented to the index loader
- **WHEN** `safe_pickle.load` executes
- **THEN** it raises `UnsafeSerializationError` without executing payload code, and tests assert this behavior

#### Scenario: Pathlib throughout orchestrators
- **GIVEN** the refactored indexing modules
- **WHEN** tests run across platforms
- **THEN** all path operations use `Path` objects, and no `os.path` calls remain in the touched code

### Requirement: Import Discipline & Typing Modernization
The system SHALL move imports to module top level (with justified exceptions), eliminate deprecated typing aliases, and remove private import usage within the refactored scope.

#### Scenario: Late import justification documented
- **GIVEN** a conditional GPU import in `orchestration.cli`
- **WHEN** import fails
- **THEN** the module emits an informative error message; Ruff `PLC0415` no longer triggers because imports are top-level or annotated with comments explaining conditional loading.

#### Scenario: Type hints modernized
- **GIVEN** the refactored modules
- **WHEN** mypy and Ruff analyze them
- **THEN** no `typing.Dict`/`typing.List` usage remains; generics and Protocols are used where appropriate, and stubs align with implementation.

