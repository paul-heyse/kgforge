# OpenSpec Change Proposal: Fix Unresolved Cross-References for Type Names

## Executive Summary

This proposal addresses **unresolved cross-reference warnings** and **duplicate-target warnings** in Sphinx documentation builds. The root causes are incomplete type mappings in `QUALIFIED_NAME_OVERRIDES`, missing intersphinx configurations, and duplicate exception definitions across modules.

## Problem Validation

### Confirmed Issues

✅ **Unresolved type references**: numpy.typing.NDArray, numpy.float32, pyarrow types appear as plain text
✅ **Incomplete QUALIFIED_NAME_OVERRIDES**: Current coverage 57 entries, needs 100+
✅ **Missing intersphinx mappings**: scipy, pandas, httpx, pytest not configured
✅ **Duplicate-target warnings**: DownloadError and UnsupportedMIMEError published in both `errors.py` and `exceptions.py`
✅ **Broken links**: Type references don't hyperlink to external documentation
✅ **Warning noise**: Masks real documentation regressions

### Gap Analysis

**Current `QUALIFIED_NAME_OVERRIDES`: 57 entries**
- ✅ Covered: Basic numpy types, basic pyarrow types, some pydantic types, custom aliases
- ❌ Missing: Numpy scalar types (int8-complex128), numpy typing variations, pyarrow core types, pydantic validators, typing_extensions, stdlib collections

**Current `intersphinx_mapping`: 8 libraries**
- ✅ Covered: python, numpy, pyarrow, duckdb, pydantic, fastapi, typer, requests
- ❌ Missing: scipy, pandas, httpx, pytest

**Duplicate targets**: 2 exception types
- `DownloadError`: Defined in `errors.py` (canonical), re-exported in `exceptions.py` (deprecated)
- `UnsupportedMIMEError`: Same pattern

## Proposal Overview

### Scope

**111 total tasks** across 7 phases:

1. **Research & Type Catalog** (14 tasks, 1 hour)
   - Catalog all type references in codebase
   - Research intersphinx inventories
   - Identify missing mappings

2. **Expand `QUALIFIED_NAME_OVERRIDES`** (70 tasks, 1.5 hours)
   - Numpy scalar types (12 full + 12 aliases = 24)
   - Numpy typing variations (5)
   - PyArrow core types (8)
   - Pydantic types (8)
   - Typing extensions (5)
   - Standard library types (10)
   - Organization & formatting (10)

3. **Update Sphinx Configuration** (13 tasks, 30 minutes)
   - Expand intersphinx mappings (+4 libraries)
   - Configure AutoAPI to exclude `exceptions.py`
   - Add `extlinks` for fallback resolution
   - Document configuration changes

4. **Testing & Validation** (22 tasks, 1 hour)
   - Create comprehensive test suite
   - Test all type categories
   - Test duplicate-target resolution
   - Test intersphinx functionality

5. **Documentation Build & Verification** (18 tasks, 30 minutes)
   - Build with warnings capture
   - Verify warning reduction
   - Build with strict validation
   - Verify hyperlink functionality

6. **Quality Assurance** (15 tasks, 30 minutes)
   - Code quality checks
   - Test suite verification
   - Integration testing
   - Documentation review

7. **Final Validation** (19 tasks, 30 minutes)
   - Compare before/after metrics
   - Verify success criteria
   - Commit with comprehensive message

**Total Estimated Time: 5.5 hours**

## Key Changes

### File: `tools/auto_docstrings.py`

**Expand `QUALIFIED_NAME_OVERRIDES` from 57 to 100+ entries:**

```python
QUALIFIED_NAME_OVERRIDES: dict[str, str] = {
    # === Numpy Scalar Types (24 new entries) ===
    "numpy.int8": "numpy.int8",
    "numpy.int16": "numpy.int16",
    "numpy.int32": "numpy.int32",
    "numpy.int64": "numpy.int64",
    # ... +8 more integer types
    "numpy.float16": "numpy.float16",
    "numpy.float64": "numpy.float64",
    # ... +2 more float types
    "numpy.complex64": "numpy.complex64",
    "numpy.complex128": "numpy.complex128",
    
    # Short aliases
    "np.int8": "numpy.int8",
    "np.int16": "numpy.int16",
    # ... +10 more short aliases
    
    # === PyArrow Types (8 new entries) ===
    "pyarrow.Table": "pyarrow.Table",
    "pyarrow.Field": "pyarrow.Field",
    "pyarrow.DataType": "pyarrow.DataType",
    # ... +5 more pyarrow types
    
    # === Pydantic Types (8 new entries) ===
    "pydantic.Field": "pydantic.Field",
    "pydantic.ValidationError": "pydantic.ValidationError",
    # ... +6 more pydantic types
    
    # === Standard Library (10 new entries) ===
    "pathlib.Path": "pathlib.Path",
    "collections.defaultdict": "collections.defaultdict",
    # ... +8 more stdlib types
    
    # ... existing 57 entries remain ...
}
```

### File: `docs/conf.py`

**1. Expand intersphinx (lines ~164-173):**
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

**2. Exclude `exceptions.py` from AutoAPI:**
```python
autoapi_ignore = [
    "*/kgfoundry_common/exceptions.py",  # Legacy aliases; use errors.py as canonical
]
```

**3. Add fallback links:**
```python
extlinks = {
    "numpy-type": ("https://numpy.org/doc/stable/reference/generated/%s.html", "%s"),
    "pyarrow-type": ("https://arrow.apache.org/docs/python/generated/%s.html", "%s"),
}
```

### File: `tests/unit/test_type_resolution.py` (NEW)

Comprehensive test suite covering:
- Numpy type resolution (scalar types, typing types, short aliases)
- PyArrow type resolution (Table, Schema, Field, etc.)
- Pydantic type resolution (Field, ValidationError, validators)
- Custom type resolution (VecArray, StrArray, custom exceptions)
- Duplicate-target elimination (DownloadError, UnsupportedMIMEError)
- Intersphinx functionality

## Success Criteria

### Quantitative Metrics

- ✅ Unresolved reference warnings: **Many → Zero**
- ✅ Duplicate-target warnings: **2 → Zero**
- ✅ QUALIFIED_NAME_OVERRIDES entries: **57 → 100+**
- ✅ Intersphinx mappings: **8 → 12**

### Qualitative Metrics

- ✅ All numpy types hyperlinked in API docs
- ✅ All pyarrow types hyperlinked in API docs
- ✅ All pydantic types hyperlinked in API docs
- ✅ DownloadError/UnsupportedMIMEError have single canonical target
- ✅ Type links navigate to correct external documentation
- ✅ Build passes with `-W` flag (warnings as errors)
- ✅ Test suite passes (100% pass rate)
- ✅ Code quality passes (ruff, black, mypy)

## Dependencies & Risks

### Dependencies

- None. All changes are configuration and mapping updates.
- No external library upgrades required.
- No breaking changes to code or API.

### Risks

**Very Low Risk:**
- ✅ All changes are additive (expanding coverage)
- ✅ Existing 57 mappings remain unchanged
- ✅ AutoAPI exclusion doesn't affect Python imports
- ✅ Tests validate correctness
- ✅ Easy rollback (revert 2-3 files)

**Mitigations:**
- Comprehensive testing before committing
- Visual verification of hyperlinks
- Before/after metric comparison
- Pre-commit hooks catch issues

## Impact

### Immediate Benefits

1. **Clean builds** - No unresolved reference or duplicate-target warnings
2. **Functional hyperlinks** - All type references clickable
3. **Better navigation** - Click through to external type documentation
4. **Signal clarity** - Real regressions become visible

### Long-term Benefits

1. **Maintainability** - Clear mapping of types to canonical names
2. **Comprehensive coverage** - 100+ types mapped (not just 57)
3. **Future-proof** - New libraries easily added to intersphinx
4. **Developer experience** - API docs fully navigable
5. **Documentation quality** - Professional-grade cross-referencing

## Files Changed

```
Modified:
- tools/auto_docstrings.py
  - QUALIFIED_NAME_OVERRIDES: +50 entries (57 → 100+)
  - Organized by category with inline comments
  - Sorted within categories

- docs/conf.py
  - intersphinx_mapping: +4 libraries (scipy, pandas, httpx, pytest)
  - autoapi_ignore: +1 pattern (exceptions.py)
  - extlinks: +2 fallback templates (numpy-type, pyarrow-type)
  - Inline documentation for all changes

Added:
- tests/unit/test_type_resolution.py
  - 20+ test functions
  - Coverage for all type categories
  - Duplicate-target tests
  - Intersphinx tests
```

## Next Steps

### For You (Review & Approval)

1. **Review proposal.md** - Confirm problem analysis and solution
2. **Review tasks.md** - Verify 111-task breakdown
3. **Review spec deltas** - Ensure requirements complete
4. **Approve** - Give go-ahead to start implementation

### For Me (After Approval)

1. **Phase 1: Research** - Catalog all type references (1 hour)
2. **Phase 2: Expand overrides** - Add 50+ mappings (1.5 hours)
3. **Phase 3: Update config** - Modify docs/conf.py (30 min)
4. **Phase 4: Test** - Create test suite (1 hour)
5. **Phase 5: Build & verify** - Check warnings (30 min)
6. **Phase 6: QA** - Code quality, integration (30 min)
7. **Phase 7: Finalize** - Metrics, commit (30 min)

## Timeline

**Total: 5.5 hours** (can complete in one work session)

## Comparison with Problem Statement

| Your Requirement | My Solution |
|------------------|-------------|
| "numpy.typing.NDArray, numpy.float32 can't be resolved" | ✅ Add to QUALIFIED_NAME_OVERRIDES + intersphinx |
| "pyarrow.schema and similar types unresolved" | ✅ Add 8+ pyarrow types to mappings |
| "Custom aliases (VecArray, StrArray, Concept) lack targets" | ✅ Already in overrides, verify resolution |
| "Duplicate-target warnings for DownloadError/UnsupportedMIMEError" | ✅ Exclude exceptions.py from AutoAPI |
| "QUALIFIED_NAME_OVERRIDES needs expansion" | ✅ Add 50+ entries (57 → 100+) |
| "intersphinx mappings need updating" | ✅ Add 4 libraries (scipy, pandas, httpx, pytest) |
| "API pages have broken links" | ✅ All types hyperlinked after changes |
| "Warnings mask regression signals" | ✅ Zero warnings = clear signal |

## Questions?

1. **Priority?** Should I start implementation immediately?
2. **Phasing?** All at once or incremental delivery?
3. **Review points?** Check in after each phase or at end?
4. **Timeline?** Any urgency or deadline constraints?

**Proposal is complete, validated, and ready for implementation! 🚀**

