# Proposal: Fix Unresolved Cross-References for Type Names

## Why

The current documentation build generates **unresolved cross-reference warnings** and **duplicate-target warnings** from Sphinx, causing:
- Broken links in API documentation
- Warning noise that masks real regression signals
- Poor documentation quality for type-heavy APIs

### Root Cause Analysis

**Sphinx Cross-Reference System:**
Sphinx relies on two mechanisms to resolve type references:
1. **Intersphinx**: Cross-project documentation links (numpy, pyarrow, pydantic, etc.)
2. **Internal mappings**: `QUALIFIED_NAME_OVERRIDES` dictionary teaches Sphinx how to resolve custom types

**Current Problems:**

####1. Missing Intersphinx Inventory Items

**Current `intersphinx_mapping` in `docs/conf.py` (lines 164-173):**
```python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "duckdb": ("https://duckdb.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "typer": ("https://typer.tiangolo.com/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
}
```

**Problem:**  
Some libraries don't provide complete intersphinx inventories or Sphinx can't find specific types in their inventory. Types that fail resolution include:
- `numpy.typing.NDArray` - Not in numpy's intersphinx inventory (newer addition)
- `numpy.float32` - Numpy scalar types not reliably in inventory
- `pyarrow.Schema` - PyArrow docs may not expose all types
- `pyarrow.schema` - Lowercase module reference

#### 2. Incomplete `QUALIFIED_NAME_OVERRIDES`

**Current coverage in `tools/auto_docstrings.py` (lines 148-204):**
- ✅ Has mappings for some numpy types: `NDArray`, `numpy.typing.NDArray`, `numpy.ndarray`, `numpy.float32`
- ✅ Has mappings for pyarrow: `pyarrow.schema`, `pyarrow.Schema`
- ✅ Has mappings for custom types: `VecArray`, `StrArray`, `FloatArray`, `IntArray`
- ❌ Missing alternative names and import variations
- ❌ Missing numpy scalar types: `numpy.int32`, `numpy.int64`, `numpy.uint8`, etc.
- ❌ Missing numpy typing variations: `numpy.typing.ArrayLike`, `numpy.dtype`
- ❌ Missing pyarrow variations: `pyarrow.Table`, `pyarrow.Field`, `pyarrow.DataType`

**Current dictionary has 57 entries**, but needs expansion for comprehensive coverage.

#### 3. Duplicate-Target Warnings

**Problem:** `DownloadError` and `UnsupportedMIMEError` are defined in two places:

**Primary definition:** `src/kgfoundry_common/errors.py` (lines 51-61)
```python
class DownloadError(Exception):
    """Describe DownloadError."""
    ...

class UnsupportedMIMEError(Exception):
    """Describe UnsupportedMIMEError."""
    ...
```

**Re-export/alias:** `src/kgfoundry_common/exceptions.py` (lines 46-54)
```python
DownloadError = _DownloadError  # Imported from errors
UnsupportedMIMEError = _UnsupportedMIMEError  # Imported from errors
```

**Sphinx behavior:**
- Sphinx finds both modules exporting the same class
- Creates duplicate targets: `kgfoundry_common.errors.DownloadError` and `kgfoundry_common.exceptions.DownloadError`
- Issues warnings: `WARNING: duplicate object description of kgfoundry_common.exceptions.DownloadError`

**Current `QUALIFIED_NAME_OVERRIDES` (lines 165-166):**
```python
"DownloadError": "src.kgfoundry_common.errors.DownloadError",
"UnsupportedMIMEError": "src.kgfoundry_common.errors.UnsupportedMIMEError",
```

This maps to `errors` module but doesn't prevent Sphinx from also indexing the `exceptions` module aliases.

### Impact on Build & Documentation

**Build Warnings:**
- Unresolved reference warnings for every occurrence of `numpy.typing.NDArray`, `numpy.float32`, `pyarrow.Schema`, etc.
- Duplicate-target warnings for `DownloadError` and `UnsupportedMIMEError`
- Warnings flood the console, making real issues hard to find

**Documentation Quality:**
- **Broken links**: Type references in signatures appear as plain text, not hyperlinks
- **Poor navigation**: Developers can't click through to type definitions
- **Incomplete API docs**: Missing cross-references reduce documentation value
- **Confusing duplicates**: Two targets for same exception type

**Developer Impact:**
- Hard to identify real documentation regressions
- Difficult to validate documentation changes
- Poor DX for users trying to understand types

### Examples of Current Failures

**Example 1: `numpy.typing.NDArray` reference**
```python
def embed_texts(texts: list[str]) -> numpy.typing.NDArray[numpy.float32]:
    """Embed text strings into dense vectors."""
```

**Rendered:** `numpy.typing.NDArray` appears as plain text, not a link  
**Expected:** Should link to numpy documentation

**Example 2: `pyarrow.Schema` reference**
```python
def read_parquet(path: str) -> tuple[pyarrow.Table, pyarrow.Schema]:
    """Read parquet file and return table with schema."""
```

**Rendered:** `pyarrow.Schema` appears as plain text  
**Expected:** Should link to pyarrow documentation

**Example 3: Duplicate exception warnings**
```
WARNING: duplicate object description of kgfoundry_common.exceptions.DownloadError, 
         other instance in autoapi/kgfoundry_common/errors/index, use :no-index: for one of them
```

**Problem:** Sphinx indexes both `errors.DownloadError` and `exceptions.DownloadError`

## What Changes

### 1. Expand `QUALIFIED_NAME_OVERRIDES` Dictionary

Add comprehensive mappings for all commonly-used type references:

#### Numpy Types (20+ additions)
- Scalar types: `numpy.int8`, `numpy.int16`, `numpy.int32`, `numpy.int64`, `numpy.uint8`, `numpy.uint16`, `numpy.uint32`, `numpy.uint64`, `numpy.float16`, `numpy.float64`, `numpy.complex64`, `numpy.complex128`
- Typing variations: `numpy.typing.ArrayLike`, `numpy.dtype`, `numpy.random.Generator`
- Short aliases: `np.int32`, `np.float64`, etc.

#### PyArrow Types (15+ additions)
- Core types: `pyarrow.Table`, `pyarrow.Field`, `pyarrow.DataType`, `pyarrow.Array`
- Schema types: `pyarrow.schema` (module), `pyarrow.Schema` (class)
- Type variations: `pyarrow.Int64Type`, `pyarrow.StringType`, `pyarrow.TimestampType`

#### Pydantic Types (10+ additions)
- Core types: `pydantic.Field`, `pydantic.ValidationError`, `pydantic.ConfigDict`
- Validators: `pydantic.field_validator`, `pydantic.model_validator`

#### Standard Library Types (10+ additions)
- Typing extensions: `typing_extensions.Self`, `typing_extensions.TypeAlias`
- Collections: `collections.defaultdict`, `collections.Counter`, `collections.OrderedDict`
- Pathlib: `pathlib.Path`, `pathlib.PurePath`

### 2. Enhance Intersphinx Configuration

Update `docs/conf.py` to add more comprehensive external documentation links:

```python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),  # NEW
    "pandas": ("https://pandas.pydata.org/docs/", None),  # NEW
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "duckdb": ("https://duckdb.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "typer": ("https://typer.tiangolo.com/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "httpx": ("https://www.python-httpx.org/", None),  # NEW
    "pytest": ("https://docs.pytest.org/en/stable/", None),  # NEW
}
```

### 3. Resolve Duplicate-Target Warnings

**Option A: Exclude `exceptions` module from AutoAPI** (Recommended)

Add to `docs/conf.py`:
```python
autoapi_ignore = [
    "*/kgfoundry_common/exceptions.py",  # Legacy aliases, use errors.py as canonical
]
```

**Rationale:**
- `exceptions.py` is explicitly marked as "deprecated" in its navmap (stability: deprecated, deprecated_in: 0.3.0)
- Exceptions are re-imports from `errors.py`
- `errors.py` is the canonical source (stability: stable)
- Excluding prevents duplicate targets while maintaining backward compatibility in code

**Option B: Add `:no-index:` directive to `exceptions.py` aliases**

This requires modifying how AutoAPI processes the module, which is complex. Option A is simpler and cleaner.

### 4. Add Fallback for Missing Intersphinx Items

Some types may still not resolve via intersphinx (e.g., new numpy.typing items not yet in their inventory). Add explicit external links:

In `docs/conf.py`:
```python
# Explicit external links for types not in intersphinx inventories
extlinks = {
    "numpy-type": ("https://numpy.org/doc/stable/reference/generated/%s.html", "%s"),
    "pyarrow-type": ("https://arrow.apache.org/docs/python/generated/%s.html", "%s"),
}
```

### 5. Add Validation & Testing

- **Add test** verifying all known type references resolve without warnings
- **Add linter check** that catches new type references without mappings
- **Document** the type resolution system in `tools/auto_docstrings.py`

## Impact

### Breaking Changes

**None.** All changes are additive or improve existing behavior.

**Note on `exceptions.py` exclusion:**
- Code imports still work (Python-level imports unchanged)
- Only affects Sphinx documentation indexing
- Canonical `errors.py` documentation remains fully available

### Affected Components

- **`tools/auto_docstrings.py`** - Expand `QUALIFIED_NAME_OVERRIDES` dictionary (+50 entries)
- **`docs/conf.py`** - Add intersphinx mappings, add `autoapi_ignore`, add `extlinks`
- **Generated API docs** - All type references will become hyperlinks
- **Tests** - New test file `tests/unit/test_type_resolution.py`

### Benefits

- **Clean builds**: No more unresolved reference warnings
- **No duplicate-target warnings**: Single canonical source for each exception
- **Better documentation**: All type references are hyperlinked
- **Improved navigation**: Click through from signatures to type definitions
- **Signal clarity**: Real documentation regressions become visible
- **Future-proof**: Comprehensive coverage of commonly-used types

### Migration Path

1. Expand `QUALIFIED_NAME_OVERRIDES` dictionary (50+ types)
2. Update `docs/conf.py` intersphinx mappings
3. Add `autoapi_ignore` for `exceptions.py`
4. Add `extlinks` for fallback resolution
5. Build documentation: `make html`
6. Verify zero unresolved reference warnings
7. Verify zero duplicate-target warnings
8. Verify all type links work

**Timeline**: 3-4 hours
- 1.5 hours: Research and catalog all type references in codebase
- 1 hour: Add all mappings to `QUALIFIED_NAME_OVERRIDES`
- 30 minutes: Update `docs/conf.py` configuration
- 30 minutes: Add tests and validation
- 30 minutes: Build docs, verify links, validate

### Rollback Strategy

If issues arise:
1. Revert `tools/auto_docstrings.py` changes
2. Revert `docs/conf.py` changes
3. Documentation build still works (just with warnings)

No code-level changes means rollback is zero-risk.

## Validation Against Codebase

### Current State Confirmed

✅ **Intersphinx configured** (`docs/conf.py` lines 164-173, 8 external projects)
✅ **QUALIFIED_NAME_OVERRIDES exists** (`tools/auto_docstrings.py` lines 148-204, 57 entries)
✅ **Duplicate targets confirmed** (`errors.py` and `exceptions.py` both export same exceptions)
✅ **Missing type coverage** (numpy scalars, pyarrow types, pydantic types not mapped)
✅ **Problem reproduced** (build generates unresolved reference warnings)

### Gap Analysis

❌ **50+ type references missing** from `QUALIFIED_NAME_OVERRIDES`
❌ **Intersphinx mappings incomplete** (missing scipy, pandas, httpx, pytest)
❌ **No exclusion for `exceptions.py`** (causes duplicate-target warnings)
❌ **No fallback mechanism** for types not in intersphinx inventories
❌ **No tests** validating type resolution
❌ **No documentation** explaining type resolution system

### Success Criteria

✅ **Zero unresolved reference warnings** in documentation build
✅ **Zero duplicate-target warnings** for exceptions
✅ **All numpy types resolve** (scalars, typing, dtype)
✅ **All pyarrow types resolve** (Table, Schema, Field, DataType)
✅ **All pydantic types resolve** (Field, ValidationError, ConfigDict)
✅ **Type links work** (click through from API docs to external docs)
✅ **Tests pass** validating type resolution coverage
✅ **`-W` build succeeds** (warnings as errors mode)

