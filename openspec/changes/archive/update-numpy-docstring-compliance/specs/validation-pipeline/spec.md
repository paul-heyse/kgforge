## ADDED Requirements

### Requirement: Parameter/Return Parity Validation
The documentation validation pipeline SHALL ensure all documented parameters match function signatures and all return values are documented.

#### Scenario: Extra documented parameters detected
- **WHEN** a function's docstring documents a parameter that does not exist in the function signature
- **THEN** pydoclint reports an error with file and line number

#### Scenario: Missing parameter documentation
- **WHEN** a function parameter is not documented in the docstring
- **THEN** Ruff rule D417 reports the undocumented parameter

#### Scenario: Return value mismatch
- **WHEN** a function returns multiple values but the `Returns` section documents only one
- **THEN** pydoclint flags the missing return documentation

#### Scenario: Type consistency
- **WHEN** a parameter is annotated as `str` in code but documented as `int` in docstring
- **THEN** validation detects the mismatch

### Requirement: Ruff Docstring Rule Enforcement
Pre-commit hooks SHALL enforce Ruff docstring linting rules including D417 (all arguments documented) and D401 (imperative summaries).

#### Scenario: All arguments documented (D417)
- **WHEN** a function has parameters
- **THEN** Ruff D417 requires all parameters in the `Parameters` section

#### Scenario: Imperative summary mood (D401)
- **WHEN** a docstring summary exists
- **THEN** Ruff D401 requires summary to start with imperative verb (e.g., "Return config" not "Returns config")

#### Scenario: Other Ruff D rules applied
- **WHEN** pre-commit runs
- **THEN** Ruff checks D1xx (missing docstrings), D2xx (whitespace), D3xx (blank lines), D4xx (docstring content)

### Requirement: pydoclint Pre-Commit Integration
Pre-commit configuration SHALL include pydoclint hook for NumPy-style validation.

#### Scenario: pydoclint runs on all Python files
- **WHEN** code is committed
- **THEN** pre-commit hook `pydoclint --style numpy src` validates all docstrings

#### Scenario: pydoclint violations block commits
- **WHEN** pydoclint detects style violations
- **THEN** commit is rejected with error message showing file, line, and violation type

#### Scenario: Supported NumPy sections
- **WHEN** pydoclint validates a docstring
- **THEN** it recognizes standard NumPy sections: Parameters, Returns, Yields, Raises, Examples, Attributes, See Also, Notes, Warnings, References

### Requirement: Interrogate Coverage Enforcement
Docstring coverage validator SHALL enforce 90% minimum coverage on `src/` directory.

#### Scenario: Coverage check passes
- **WHEN** docstring coverage is >= 90%
- **THEN** interrogate exits with code 0 and build continues

#### Scenario: Coverage check fails
- **WHEN** docstring coverage drops below 90%
- **THEN** interrogate reports missing docstrings and build fails
