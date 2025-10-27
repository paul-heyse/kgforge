## MODIFIED Requirements

### Requirement: NumPy Docstring Template Format
Docstring generation templates SHALL emit only NumPy-style sections with consistent formatting.

#### Scenario: Function parameter documentation
- **WHEN** a function has parameters
- **THEN** template generates `Parameters` section with format: `name : type, optional` and defaults in description (e.g., "Default is 2.")

#### Scenario: Return value documentation
- **WHEN** a function returns a value
- **THEN** template generates `Returns` section with `name : type` and description, or single `type` line if unnamed

#### Scenario: No Google-style sections
- **WHEN** docstring generation runs
- **THEN** no `Args:`, `Kwargs:`, or other Google-style sections are emitted

#### Scenario: Type annotation format
- **WHEN** a parameter has type annotation
- **THEN** type is rendered in NumPy format: `str`, `int`, `ndarray of shape (n,)`, `Sequence[str] | None`, not `str or None`

### Requirement: Auto Docstring Generator Compliance
Fallback auto_docstrings.py generator SHALL produce NumPy-compliant docstrings.

#### Scenario: Generated docstring completeness
- **WHEN** auto_docstrings.py creates a docstring
- **THEN** it includes: Summary (imperative mood), Parameters, Returns, Raises (if applicable), Examples (skeleton)

#### Scenario: No TODO placeholders remain
- **WHEN** docstring generation completes
- **THEN** all docstrings contain actual descriptions, not placeholder text like "TODO"

#### Scenario: Imperative summary style
- **WHEN** a function docstring is generated
- **THEN** summary starts with verb: "Return config" not "Returns config", "Load data" not "Loads data"

## ADDED Requirements

### Requirement: Raises Section Generation
All generated docstrings for functions that raise exceptions SHALL include a `Raises` section.

#### Scenario: Exception documentation
- **WHEN** a function raises `ValueError`, `TypeError`, etc.
- **THEN** docstring includes `Raises` section documenting each exception type and when it occurs

#### Scenario: ValueError with constraint
- **WHEN** a function validates input and raises `ValueError`
- **THEN** docstring documents: `ValueError: If input is empty or invalid format.`

#### Scenario: Built-in exceptions
- **WHEN** a function can raise `AttributeError`, `KeyError`, `IndexError`
- **THEN** docstring documents: `AttributeError: If required attribute is missing.`

### Requirement: Examples Section Scaffold
All generated docstrings SHALL include an `Examples` section with doctestable scaffold.

#### Scenario: Minimal working example
- **WHEN** a function is documented
- **THEN** docstring includes `Examples` section with `>>>` prompt showing minimal usage

#### Scenario: Examples are doctestable
- **WHEN** examples include imports and actual function calls
- **THEN** `pytest --doctest-modules` can execute the example without errors

#### Scenario: Example output shown
- **WHEN** a function returns a value
- **THEN** example shows both the function call and expected output

### Requirement: Attributes Section for Classes
Class docstrings SHALL include an `Attributes` section documenting all public attributes.

#### Scenario: Class attribute documentation
- **WHEN** a class has public attributes (instance or class-level)
- **THEN** docstring includes `Attributes` section with `name : type` format

#### Scenario: Attribute details
- **WHEN** an attribute is complex (e.g., ndarray, dictionary)
- **THEN** description includes shape/size constraints and format: `weights : ndarray of shape (n_features,)`

#### Scenario: Mutability indicators
- **WHEN** an attribute is read-only or computed
- **THEN** description indicates: "Read-only. Computed from training data."

### Requirement: See Also Section for Related Symbols
All generated docstrings for public API functions/classes SHALL include a `See Also` section.

#### Scenario: Related functions linked
- **WHEN** a function has related functionality
- **THEN** docstring includes `See Also` section with `name : description` pairs

#### Scenario: Cross-references resolvable
- **WHEN** a `See Also` entry references another symbol
- **THEN** Sphinx can resolve and create a hyperlink to that symbol

### Requirement: Notes Section for Contracts and Constraints
Public API docstrings SHALL include a `Notes` section documenting contracts, constraints, or complexity where applicable.

#### Scenario: Input constraints documented
- **WHEN** a function has input requirements
- **THEN** Notes section states: "Input must be UTF-8 text; empty documents are ignored."

#### Scenario: Complexity notation
- **WHEN** a function has known complexity characteristics
- **THEN** Notes section includes: "Time complexity: O(n log n). Space complexity: O(n)."

#### Scenario: Thread-safety notes
- **WHEN** a class or function has thread-safety implications
- **THEN** Notes section states: "Thread-safe for concurrent reads; writes require locking."
