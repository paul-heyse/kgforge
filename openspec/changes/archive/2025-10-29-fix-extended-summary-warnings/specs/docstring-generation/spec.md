## MODIFIED Requirements

### Requirement: Automatic Docstring Generation
The system SHALL generate NumPy-style docstrings for all Python functions, classes, and modules with complete extended summaries that pass numpydoc ES01 validation.

#### Scenario: Magic method docstrings include extended summaries
- **WHEN** `python tools/generate_docstrings.py` generates docstrings for Python magic methods
- **THEN** each magic method receives a context-rich extended summary explaining when and how it's invoked, what protocol it implements, and why it exists

#### Scenario: All 100+ Python magic methods have extended summaries
- **WHEN** docstring generation processes a class with magic methods
- **THEN** all magic methods receive appropriate extended summaries from `MAGIC_METHOD_EXTENDED_SUMMARIES` dictionary covering object lifecycle, attribute access, descriptors, pickling, numeric operators (binary/reverse/in-place), unary operators, type conversions, collections, type system, and miscellaneous special methods

#### Scenario: Pydantic artifact docstrings include extended summaries
- **WHEN** `python tools/generate_docstrings.py` generates docstrings for Pydantic model attributes
- **THEN** each Pydantic artifact (`model_*` methods, `__pydantic_*__` attributes) receives extended summary from `PYDANTIC_ARTIFACT_SUMMARIES` dictionary explaining its role in validation, serialization, or schema generation

#### Scenario: Generic fallbacks provide multi-sentence extended summaries
- **WHEN** docstring generation encounters a magic method or Pydantic artifact not explicitly mapped in dictionaries
- **THEN** generic fallback returns a multi-sentence extended summary providing context about Python's object protocol or Pydantic's internals, not just a one-line summary

#### Scenario: Documentation build passes ES01 validation with zero warnings
- **WHEN** `make html` builds Sphinx documentation with `numpydoc_validation_checks = {"ES01"}` enabled
- **THEN** build completes with zero ES01 warnings indicating all docstrings have proper extended summaries

#### Scenario: Documentation build succeeds with warnings-as-errors enabled
- **WHEN** `SPHINXOPTS="-W" make html` builds documentation treating warnings as errors
- **THEN** build completes successfully without any ES01-related failures

#### Scenario: Generated API documentation includes narrative context
- **WHEN** viewing generated HTML documentation for a class with magic methods
- **THEN** each magic method's documentation page displays extended summary as narrative paragraph before parameter tables, making the purpose and usage clear to human readers

## ADDED Requirements

### Requirement: Comprehensive Magic Method Extended Summary Coverage
The system SHALL provide context-rich extended summaries for all 100+ Python magic methods organized by functional category.

#### Scenario: Object lifecycle methods have appropriate extended summaries
- **WHEN** generating docstrings for `__new__`, `__del__`, or `__init_subclass__`
- **THEN** extended summaries explain allocation/destruction timing, cleanup responsibilities, or subclass customization hooks

#### Scenario: Attribute access methods have appropriate extended summaries
- **WHEN** generating docstrings for `__getattr__`, `__getattribute__`, `__setattr__`, `__delattr__`, or `__dir__`
- **THEN** extended summaries explain fallback lookup behavior, override semantics, validation opportunities, deletion handling, or introspection customization

#### Scenario: Descriptor protocol methods have appropriate extended summaries
- **WHEN** generating docstrings for `__get__`, `__set__`, `__delete__`, or `__set_name__`
- **THEN** extended summaries explain descriptor protocol behavior, attribute management, and name binding hooks

#### Scenario: Pickling methods have appropriate extended summaries
- **WHEN** generating docstrings for `__getstate__`, `__setstate__`, `__reduce__`, `__reduce_ex__`, `__getnewargs__`, or `__getnewargs_ex__`
- **THEN** extended summaries explain serialization state preparation, deserialization restoration, reconstruction protocols, and protocol version differences

#### Scenario: Binary numeric operators have appropriate extended summaries
- **WHEN** generating docstrings for `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__divmod__`, `__pow__`, `__matmul__`, `__lshift__`, `__rshift__`, `__and__`, `__xor__`, or `__or__`
- **THEN** extended summaries explain which operator syntax is enabled (e.g., `+`, `-`, `*`, `/`, `//`, `%`, `**`, `@`, `<<`, `>>`, `&`, `^`, `|`)

#### Scenario: Reverse numeric operators have appropriate extended summaries
- **WHEN** generating docstrings for `__radd__`, `__rsub__`, `__rmul__`, and other reverse operators
- **THEN** extended summaries explain reflection behavior when left operand doesn't support the operation

#### Scenario: In-place numeric operators have appropriate extended summaries
- **WHEN** generating docstrings for `__iadd__`, `__isub__`, `__imul__`, and other in-place operators
- **THEN** extended summaries explain augmented assignment behavior (e.g., `+=`, `-=`, `*=`) and mutability expectations

#### Scenario: Unary operators have appropriate extended summaries
- **WHEN** generating docstrings for `__neg__`, `__pos__`, `__abs__`, or `__invert__`
- **THEN** extended summaries explain unary operator syntax (`-x`, `+x`, `abs(x)`, `~x`) and expected return values

#### Scenario: Type conversion methods have appropriate extended summaries
- **WHEN** generating docstrings for `__int__`, `__float__`, `__complex__`, `__index__`, `__round__`, `__trunc__`, `__floor__`, or `__ceil__`
- **THEN** extended summaries explain conversion semantics, built-in function integration, and precision expectations

#### Scenario: Collection methods have appropriate extended summaries
- **WHEN** generating docstrings for `__reversed__`, `__length_hint__`, or `__missing__`
- **THEN** extended summaries explain reverse iteration, optimization hints, or dict subclass key miss handling

#### Scenario: Type system methods have appropriate extended summaries
- **WHEN** generating docstrings for `__instancecheck__`, `__subclasscheck__`, or `__class_getitem__`
- **THEN** extended summaries explain metaclass customization of `isinstance()`/`issubclass()` or generic type subscripting

#### Scenario: Miscellaneous special methods have appropriate extended summaries
- **WHEN** generating docstrings for `__bytes__`, `__format__`, `__sizeof__`, `__fspath__`, `__buffer__`, or `__release_buffer__`
- **THEN** extended summaries explain bytes conversion, format string protocol, memory footprint reporting, file system path representation, or buffer protocol mechanics

### Requirement: Extended Summary Test Coverage
The system SHALL include comprehensive tests validating that all magic methods and Pydantic artifacts receive appropriate extended summaries.

#### Scenario: Test suite validates all magic method categories have coverage
- **WHEN** running `pytest tests/unit/test_auto_docstrings_extended_summaries.py`
- **THEN** tests verify object lifecycle, attribute access, descriptors, pickling, all numeric operator categories, type conversions, collections, type system, and miscellaneous methods all have extended summaries

#### Scenario: Test suite validates extended summary format
- **WHEN** running extended summary tests
- **THEN** tests verify summaries are multi-sentence (not one-liners), end with periods, provide context and usage information, and are grammatically correct

#### Scenario: Test suite validates fallback behavior
- **WHEN** testing generic fallbacks for unmapped magic methods or Pydantic artifacts
- **THEN** tests verify fallbacks return multi-sentence extended summaries with meaningful context, not empty strings or single-sentence generic text

#### Scenario: Test suite validates Pydantic artifact coverage
- **WHEN** running Pydantic artifact tests
- **THEN** tests verify all known `model_*` methods and `__pydantic_*__` attributes have extended summaries, detection function works correctly, and fallback provides extended summary for unknown artifacts

