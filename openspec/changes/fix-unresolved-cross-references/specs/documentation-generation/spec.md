## MODIFIED Requirements

### Requirement: Type Cross-Reference Resolution
The system SHALL resolve all type references in function signatures and docstrings to their canonical definitions, enabling hyperlinked navigation in generated API documentation without warnings.

#### Scenario: Numpy types resolve to external documentation
- **WHEN** a function signature includes `numpy.typing.NDArray`, `numpy.float32`, or other numpy types
- **THEN** Sphinx resolves the type reference and creates a hyperlink to numpy's official documentation

#### Scenario: Numpy short aliases resolve correctly
- **WHEN** a function signature uses `np.int32`, `np.float64`, or other `np.` prefixed types
- **THEN** Sphinx maps the short alias to canonical numpy type and resolves reference

#### Scenario: PyArrow types resolve to external documentation
- **WHEN** a function signature includes `pyarrow.Table`, `pyarrow.Schema`, `pyarrow.Field`, or other pyarrow types
- **THEN** Sphinx resolves the type reference and creates a hyperlink to PyArrow's official documentation

#### Scenario: Pydantic types resolve to external documentation
- **WHEN** a function signature includes `pydantic.Field`, `pydantic.ValidationError`, or other pydantic types
- **THEN** Sphinx resolves the type reference and creates a hyperlink to Pydantic's official documentation

#### Scenario: Custom type aliases resolve to internal definitions
- **WHEN** a function signature includes custom aliases like `VecArray`, `StrArray`, `Doc`, `Chunk`, or `Concept`
- **THEN** Sphinx resolves the type reference and creates a hyperlink to the internal definition in the codebase

#### Scenario: Standard library types resolve correctly
- **WHEN** a function signature includes `pathlib.Path`, `collections.defaultdict`, `datetime.datetime`, or other stdlib types
- **THEN** Sphinx resolves the type reference via intersphinx to Python's official documentation

#### Scenario: Documentation build produces zero unresolved reference warnings
- **WHEN** `make html` builds Sphinx documentation
- **THEN** build completes with zero unresolved reference warnings for type names

## ADDED Requirements

### Requirement: Canonical Exception Documentation
The system SHALL designate a single canonical module for each exception type to eliminate duplicate-target warnings and ensure consistent documentation links.

#### Scenario: Exceptions have single canonical source module
- **WHEN** an exception like `DownloadError` or `UnsupportedMIMEError` is defined in `errors.py` and re-exported from `exceptions.py`
- **THEN** AutoAPI indexes only the canonical source (`errors.py`) and excludes the legacy alias module (`exceptions.py`) from documentation

#### Scenario: Exception references link to canonical module
- **WHEN** documentation references `DownloadError` or `UnsupportedMIMEError`
- **THEN** the hyperlink points to `kgfoundry_common.errors` (not `kgfoundry_common.exceptions`)

#### Scenario: Zero duplicate-target warnings for exceptions
- **WHEN** `make html` builds Sphinx documentation
- **THEN** build completes with zero duplicate-target warnings for `DownloadError` and `UnsupportedMIMEError`

#### Scenario: Code-level imports still work with legacy module
- **WHEN** Python code imports from `kgfoundry_common.exceptions`
- **THEN** imports succeed normally (Python-level imports unchanged, only documentation indexing affected)

### Requirement: Comprehensive Type Override Coverage
The system SHALL maintain comprehensive mappings in `QUALIFIED_NAME_OVERRIDES` covering all commonly-used external and custom types with 100+ entries.

#### Scenario: All numpy scalar types have mappings
- **WHEN** checking `QUALIFIED_NAME_OVERRIDES` dictionary
- **THEN** all numpy scalar types (`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `complex64`, `complex128`) have both full and short alias mappings

#### Scenario: All numpy typing types have mappings
- **WHEN** checking `QUALIFIED_NAME_OVERRIDES` dictionary
- **THEN** numpy typing types (`numpy.typing.NDArray`, `numpy.typing.ArrayLike`, `numpy.dtype`) have mappings

#### Scenario: All pyarrow core types have mappings
- **WHEN** checking `QUALIFIED_NAME_OVERRIDES` dictionary
- **THEN** pyarrow core types (`Table`, `Schema`, `Field`, `DataType`, `Array`, `RecordBatch`) have mappings

#### Scenario: All pydantic types have mappings
- **WHEN** checking `QUALIFIED_NAME_OVERRIDES` dictionary
- **THEN** pydantic types (`Field`, `ValidationError`, `ConfigDict`, `field_validator`, `model_validator`) have mappings

#### Scenario: Standard library types have mappings
- **WHEN** checking `QUALIFIED_NAME_OVERRIDES` dictionary
- **THEN** commonly-used stdlib types (`pathlib.Path`, `collections.defaultdict`, `datetime.datetime`, `uuid.UUID`) have mappings

#### Scenario: Dictionary has organized structure with comments
- **WHEN** reviewing `QUALIFIED_NAME_OVERRIDES` dictionary in code
- **THEN** entries are grouped by category (numpy, pyarrow, pydantic, stdlib, custom), sorted within categories, and have inline comments documenting each group

### Requirement: Enhanced Intersphinx Integration
The system SHALL configure intersphinx mappings for all major Python libraries used in the codebase to enable external documentation cross-linking.

#### Scenario: Scientific Python ecosystem libraries mapped
- **WHEN** checking intersphinx configuration in `docs/conf.py`
- **THEN** numpy, scipy, and pandas are mapped to their official documentation sites

#### Scenario: Data libraries mapped
- **WHEN** checking intersphinx configuration
- **THEN** pyarrow and duckdb are mapped to their official documentation sites

#### Scenario: Web framework libraries mapped
- **WHEN** checking intersphinx configuration
- **THEN** pydantic, fastapi, typer, requests, and httpx are mapped to their official documentation sites

#### Scenario: Testing framework mapped
- **WHEN** checking intersphinx configuration
- **THEN** pytest is mapped to its official documentation site

#### Scenario: Intersphinx URLs are accessible and functional
- **WHEN** building documentation with intersphinx enabled
- **THEN** all configured intersphinx URLs return valid `objects.inv` inventory files

### Requirement: Type Resolution Test Coverage
The system SHALL include comprehensive tests validating that all type references resolve without warnings and links function correctly.

#### Scenario: Test suite validates numpy type resolution
- **WHEN** running `pytest tests/unit/test_type_resolution.py`
- **THEN** tests verify all numpy scalar types, typing types, and short aliases have valid mappings

#### Scenario: Test suite validates external library type resolution
- **WHEN** running type resolution tests
- **THEN** tests verify pyarrow, pydantic, and stdlib types have valid mappings

#### Scenario: Test suite validates duplicate-target elimination
- **WHEN** running type resolution tests
- **THEN** tests verify DownloadError and UnsupportedMIMEError have single canonical targets

#### Scenario: Test suite validates hyperlink functionality
- **WHEN** running type resolution tests
- **THEN** tests verify type references in generated HTML documentation are properly hyperlinked to correct destinations

