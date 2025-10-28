# Proposal: Fix ES01 Extended Summary Warnings for Magic Methods and Pydantic Artifacts

## Why

The current documentation build generates **hundreds of ES01 warnings** from numpydoc validation, specifically:
- `ES01: No extended summary found`

### Root Cause Analysis

**Numpydoc Configuration:** `docs/conf.py` enables strict validation including `ES01` checks:
```python
numpydoc_validation_checks = {
    "GL01",  # Docstring should start in the line immediately after the quotes
    "SS01",  # No summary found
    "ES01",  # No extended summary found  ← FAILING
    "RT01",  # No Returns section found
    "PR01",  # Parameters {missing_params} not documented
}
```

**Current Coverage in `auto_docstrings.py`:**

`MAGIC_METHOD_EXTENDED_SUMMARIES` dictionary covers only **28 magic methods**:
- ✅ Covered: `__repr__`, `__str__`, `__len__`, `__iter__`, `__next__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__bool__`, `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`, `__hash__`, `__call__`, `__enter__`, `__exit__`, `__aenter__`, `__aexit__`, `__await__`, `__aiter__`, `__anext__`, `__copy__`, `__deepcopy__`

**Missing: ~80+ magic methods** including:
- Object lifecycle: `__new__`, `__del__`, `__init_subclass__`
- Attribute access: `__getattr__`, `__getattribute__`, `__setattr__`, `__delattr__`, `__dir__`
- Descriptors: `__get__`, `__set__`, `__delete__`, `__set_name__`
- Pickling: `__getstate__`, `__setstate__`, `__reduce__`, `__reduce_ex__`, `__getnewargs__`, `__getnewargs_ex__`
- Numeric operators: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`, `__matmul__`
- Reverse operators: `__radd__`, `__rsub__`, `__rmul__`, etc. (20+ methods)
- In-place operators: `__iadd__`, `__isub__`, `__imul__`, etc. (13+ methods)
- Unary operators: `__neg__`, `__pos__`, `__abs__`, `__invert__`
- Type conversions: `__int__`, `__float__`, `__complex__`, `__index__`, `__round__`, `__trunc__`, `__floor__`, `__ceil__`
- Collection methods: `__reversed__`, `__length_hint__`, `__missing__`
- Type system: `__instancecheck__`, `__subclasscheck__`, `__class_getitem__`
- Other: `__bytes__`, `__format__`, `__sizeof__`, `__fspath__`, `__buffer__`, `__release_buffer__`

`PYDANTIC_ARTIFACT_SUMMARIES` dictionary covers **35 Pydantic attributes**, but the fallback in `extended_summary()` function (line 430) only applies this when `_is_pydantic_artifact(name)` returns `True`.

**The Problem:**
1. When a magic method NOT in `MAGIC_METHOD_EXTENDED_SUMMARIES` is processed, the code falls through to line 437: `"Special method customising Python's object protocol for this class."` - This is a **one-line summary** but NumPy expects an **extended summary** (paragraph after the one-line summary).

2. For Pydantic artifacts, the extended summary is retrieved from the dictionary, but if the artifact is new or uncommon, it also falls back to a generic message without proper extended summary.

3. NumPy docstring format requires:
   ```
   """One-line summary.

   Extended summary paragraph providing context, usage notes,
   and important details about the function's behavior.
   
   Parameters
   ----------
   ...
   ```

### Impact on Build & Documentation

**Build Noise:**
- Hundreds of warnings during `sphinx-build` execution
- Makes real issues hard to find
- Slows down documentation generation
- Fails with `-W` flag (warnings as errors)

**Documentation Quality:**
- API pages start with parameter tables only, no narrative
- Hard for humans to skim and understand purpose
- AI agents struggle to extract intent without context
- Missing opportunity to explain when/why to use each method

### Examples of Current Failures

**Example 1: `__getstate__` (pickle support)**
```python
def __getstate__(self) -> dict[str, Any]:
    """Special method customising Python's object protocol for this class.

    Returns
    -------
    dict[str, Any]
        Description of return value.
    """
```
**Problem:** No extended summary explaining pickle serialization context.

**Example 2: `__reduce__` (pickle protocol)**
```python
def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
    """Special method customising Python's object protocol for this class.

    Returns
    -------
    tuple[type, tuple[Any, ...]]
        Description of return value.
    """
```
**Problem:** Generic fallback doesn't explain reconstruction protocol.

**Example 3: `__add__` (numeric operator)**
```python
def __add__(self, other: Vector) -> Vector:
    """Special method customising Python's object protocol for this class.

    Parameters
    ----------
    other : Vector
        Description.

    Returns
    -------
    Vector
        Description of return value.
    """
```
**Problem:** Doesn't explain operator overloading for `+` syntax.

## What Changes

### 1. Complete Magic Method Coverage (80+ Methods)

Expand `MAGIC_METHOD_EXTENDED_SUMMARIES` dictionary to include **all Python magic methods** with context-rich extended summaries.

**Categories to add:**

#### Object Lifecycle & Construction
- `__new__`: "Allocate and initialize a new instance before ``__init__`` runs."
- `__del__`: "Clean up resources when the instance is about to be destroyed."
- `__init_subclass__`: "Customize subclass creation and enforce invariants on derived classes."

#### Attribute Access & Descriptors
- `__getattr__`: "Intercept attribute lookups that failed through normal mechanisms."
- `__getattribute__`: "Override all attribute access, called before checking the instance dictionary."
- `__setattr__`: "Control attribute assignment and enforce validation rules."
- `__delattr__`: "Handle attribute deletion and maintain consistency."
- `__dir__`: "Customize the list of attributes returned by ``dir()`` for introspection."
- `__get__`, `__set__`, `__delete__`, `__set_name__`: Descriptor protocol methods

#### Pickling & Serialization
- `__getstate__`: "Prepare the instance state for serialization via ``pickle``."
- `__setstate__`: "Restore instance state when unpickling from serialized data."
- `__reduce__`: "Return reconstruction arguments for pickle protocol version 0-2."
- `__reduce_ex__`: "Return reconstruction arguments with support for newer pickle protocols."
- `__getnewargs__`, `__getnewargs_ex__`: Constructor argument serialization

#### Numeric Operators (Binary)
- 20+ methods: `__add__`, `__sub__`, `__mul__`, `__matmul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__divmod__`, `__pow__`, `__lshift__`, `__rshift__`, `__and__`, `__xor__`, `__or__`

#### Numeric Operators (Reverse & In-Place)
- 20+ reverse methods: `__radd__`, `__rsub__`, etc.
- 13+ in-place methods: `__iadd__`, `__isub__`, etc.

#### Unary Operators & Conversions
- `__neg__`, `__pos__`, `__abs__`, `__invert__`
- `__int__`, `__float__`, `__complex__`, `__index__`
- `__round__`, `__trunc__`, `__floor__`, `__ceil__`

#### Collection & Container Methods
- `__reversed__`: "Yield items in reverse order for ``reversed()`` built-in."
- `__length_hint__`: "Provide an estimated length for optimization purposes."
- `__missing__`: "Handle dictionary key misses in subclasses of ``dict``."

#### Type System & Introspection
- `__instancecheck__`: "Customize ``isinstance()`` behavior for metaclasses."
- `__subclasscheck__`: "Customize ``issubclass()`` behavior for metaclasses."
- `__class_getitem__`: "Enable generic type subscripting like ``MyClass[int]``."

#### Other Special Methods
- `__bytes__`: "Return a bytes object representing the instance."
- `__format__`: "Format the instance according to the provided format specification."
- `__sizeof__`: "Return the memory footprint in bytes for ``sys.getsizeof()``."
- `__fspath__`: "Return the file system path representation for ``os.fspath()``."

### 2. Enhanced Pydantic Artifact Coverage

Expand `PYDANTIC_ARTIFACT_SUMMARIES` to include newly discovered Pydantic 2.x internals and improve existing summaries with extended context.

**Add missing artifacts:**
- `__pydantic_private__`: Private attributes not included in serialization
- `__pydantic_init_subclass__`: Pydantic hook for subclass initialization
- `__get_pydantic_core_schema__`: Method for custom schema generation
- `__get_pydantic_json_schema__`: Method for JSON schema customization

**Enhance existing summaries** with extended context explaining when/why they're used.

### 3. Modify `extended_summary()` Function

Update `tools/auto_docstrings.py` `extended_summary()` function to:

1. **Return extended summaries** (multi-sentence) not just one-liners
2. **Add fallback extended summary** for unrecognized magic methods
3. **Add extended summary** to generic Pydantic artifact fallback
4. **Structure output** to match NumPy expectations

**Current code (lines 434-437):**
```python
if kind == "function" and name in MAGIC_METHOD_EXTENDED_SUMMARIES:
    return MAGIC_METHOD_EXTENDED_SUMMARIES[name]
if kind == "function" and _is_magic(name):
    return "Special method customising Python's object protocol for this class."
```

**New code:**
```python
if kind == "function" and name in MAGIC_METHOD_EXTENDED_SUMMARIES:
    return MAGIC_METHOD_EXTENDED_SUMMARIES[name]
if kind == "function" and _is_magic(name):
    # Extended fallback for magic methods not explicitly mapped
    return (
        "Special method customising Python's object protocol for this class. "
        "This method enables integration with Python's built-in operators, functions, "
        "or runtime services. Consult the Python data model documentation for details "
        "on when and how this method is invoked."
    )
```

### 4. Add Validation & Testing

- **Add test** verifying all 100+ magic methods get extended summaries
- **Add test** verifying all known Pydantic artifacts get extended summaries
- **Add linter check** that catches new magic methods without coverage
- **Document** the extended summary requirements in `auto_docstrings.py` module docstring

## Impact

### Breaking Changes

**None.** All changes are additive—expanding coverage to methods that previously had generic fallbacks.

### Affected Components

- **`tools/auto_docstrings.py`** - Expand dictionaries, modify `extended_summary()` function
- **Documentation builds** - Warnings will disappear, clean `-W` builds
- **Generated API docs** - All magic methods and Pydantic artifacts will have context-rich extended summaries
- **Tests** - New test file `tests/unit/test_auto_docstrings_extended_summaries.py`

### Benefits

- **Silent builds**: No more ES01 warnings flooding the console
- **Strict validation passes**: `-W` flag (warnings as errors) will succeed
- **Better documentation**: Every magic method explained with context
- **Machine-readable**: AI agents can extract intent from extended summaries
- **Future-proof**: New magic methods fall back to informative generic text
- **Maintainable**: Clear mapping of method names to human-readable explanations

### Migration Path

1. Expand `MAGIC_METHOD_EXTENDED_SUMMARIES` dictionary (80+ methods)
2. Expand `PYDANTIC_ARTIFACT_SUMMARIES` dictionary (new artifacts)
3. Modify `extended_summary()` fallback logic
4. Regenerate all docstrings: `python tools/generate_docstrings.py`
5. Run documentation build: `make html`
6. Verify no ES01 warnings
7. Commit updated docstrings

**Timeline**: 4-6 hours
- 2 hours: Research and write extended summaries for 80+ magic methods
- 1 hour: Research and add missing Pydantic artifacts
- 1 hour: Modify `extended_summary()` function logic
- 1 hour: Add tests and validation
- 1 hour: Regenerate docstrings and verify build

### Rollback Strategy

If issues arise:
1. Revert `tools/auto_docstrings.py` changes
2. Revert regenerated docstrings (git checkout)
3. Temporarily disable ES01 check in `docs/conf.py` if needed

No breaking changes means rollback is low-risk.

## Validation Against Codebase

### Current State Confirmed

✅ **Numpydoc validation enabled** (`docs/conf.py` lines 97-103)
✅ **ES01 check active** (extended summary validation)
✅ **Existing coverage** (28/100+ magic methods)
✅ **Existing Pydantic coverage** (35 artifacts with summaries)
✅ **Generic fallback exists** (line 437 in `auto_docstrings.py`)
✅ **Problem reproduced** (build generates ES01 warnings)

### Gap Analysis

❌ **80+ magic methods missing** from `MAGIC_METHOD_EXTENDED_SUMMARIES`
❌ **Generic fallback** returns one-liner, not extended summary
❌ **Pydantic artifacts** may have new 2.x internals not yet covered
❌ **No tests** validating extended summary coverage
❌ **No documentation** explaining extended summary requirements

### Success Criteria

✅ **Zero ES01 warnings** in documentation build
✅ **All 100+ magic methods** have extended summaries
✅ **All Pydantic artifacts** have extended summaries
✅ **Generic fallback** returns multi-sentence extended summary
✅ **Tests pass** validating coverage
✅ **`-W` build succeeds** (warnings as errors mode)
✅ **API docs improved** (all methods have narrative context)

