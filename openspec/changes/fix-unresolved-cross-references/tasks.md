# Implementation Tasks

## 1. Research & Type Catalog

- [ ] 1.1 Catalog all type references in codebase
  - [ ] 1.1.1 Search for `numpy.` type references: `grep -r "numpy\." src/ --include="*.py" | grep -E "(typing|int32|float32|ndarray)" | sort | uniq`
  - [ ] 1.1.2 Search for `np.` type references (numpy short alias)
  - [ ] 1.1.3 Search for `pyarrow.` type references
  - [ ] 1.1.4 Search for `pydantic.` type references
  - [ ] 1.1.5 Search for `typing_extensions.` references
  - [ ] 1.1.6 Search for custom type aliases (`VecArray`, `StrArray`, etc.)
  - [ ] 1.1.7 Create comprehensive list of all unique type references
  - [ ] 1.1.8 Cross-reference with current `QUALIFIED_NAME_OVERRIDES` dictionary
  - [ ] 1.1.9 Identify 50+ types missing from current coverage

- [ ] 1.2 Research intersphinx inventories
  - [ ] 1.2.1 Check numpy intersphinx inventory for available types
  - [ ] 1.2.2 Check pyarrow intersphinx inventory for available types
  - [ ] 1.2.3 Check pydantic intersphinx inventory for available types
  - [ ] 1.2.4 Identify types that need explicit mapping (not in inventories)
  - [ ] 1.2.5 Document which types can be resolved via intersphinx vs need overrides

## 2. Expand `QUALIFIED_NAME_OVERRIDES` Dictionary

- [x] 2.1 Add numpy scalar types (12 types)
  - [x] 2.1.1 Add `numpy.int8` → `numpy.int8`
  - [x] 2.1.2 Add `numpy.int16` → `numpy.int16`
  - [x] 2.1.3 Add `numpy.int32` → `numpy.int32`
  - [x] 2.1.4 Add `numpy.int64` → `numpy.int64`
  - [x] 2.1.5 Add `numpy.uint8` → `numpy.uint8`
  - [x] 2.1.6 Add `numpy.uint16` → `numpy.uint16`
  - [x] 2.1.7 Add `numpy.uint32` → `numpy.uint32`
  - [x] 2.1.8 Add `numpy.uint64` → `numpy.uint64`
  - [x] 2.1.9 Add `numpy.float16` → `numpy.float16`
  - [x] 2.1.10 Add `numpy.float64` → `numpy.float64`
  - [x] 2.1.11 Add `numpy.complex64` → `numpy.complex64`
  - [x] 2.1.12 Add `numpy.complex128` → `numpy.complex128`

- [x] 2.2 Add numpy scalar short aliases (12 types)
  - [x] 2.2.1 Add `np.int8` → `numpy.int8`
  - [x] 2.2.2 Add `np.int16` → `numpy.int16`
  - [x] 2.2.3 Add `np.int32` → `numpy.int32`
  - [x] 2.2.4 Add `np.int64` → `numpy.int64`
  - [x] 2.2.5 Add `np.uint8` → `numpy.uint8`
  - [x] 2.2.6 Add `np.uint16` → `numpy.uint16`
  - [x] 2.2.7 Add `np.uint32` → `numpy.uint32`
  - [x] 2.2.8 Add `np.uint64` → `numpy.uint64`
  - [x] 2.2.9 Add `np.float16` → `numpy.float16`
  - [x] 2.2.10 Add `np.float64` → `numpy.float64`
  - [x] 2.2.11 Add `np.complex64` → `numpy.complex64`
  - [x] 2.2.12 Add `np.complex128` → `numpy.complex128`

- [x] 2.3 Add numpy typing variations (5 types)
  - [x] 2.3.1 Add `ArrayLike` → `numpy.typing.ArrayLike` (if not already present)
  - [x] 2.3.2 Add `numpy.typing.ArrayLike` → `numpy.typing.ArrayLike`
  - [x] 2.3.3 Add `numpy.dtype` → `numpy.dtype`
  - [x] 2.3.4 Add `np.dtype` → `numpy.dtype`
  - [x] 2.3.5 Add `numpy.random.Generator` → `numpy.random.Generator`

- [x] 2.4 Add pyarrow core types (8 types)
  - [x] 2.4.1 Add `pyarrow.Table` → `pyarrow.Table`
  - [x] 2.4.2 Add `pyarrow.Field` → `pyarrow.Field`
  - [x] 2.4.3 Add `pyarrow.DataType` → `pyarrow.DataType`
  - [x] 2.4.4 Add `pyarrow.Array` → `pyarrow.Array`
  - [x] 2.4.5 Add `pyarrow.Int64Type` → `pyarrow.Int64Type`
  - [x] 2.4.6 Add `pyarrow.StringType` → `pyarrow.StringType`
  - [x] 2.4.7 Add `pyarrow.TimestampType` → `pyarrow.TimestampType`
  - [x] 2.4.8 Add `pyarrow.RecordBatch` → `pyarrow.RecordBatch`

- [x] 2.5 Add pydantic types (8 types)
  - [x] 2.5.1 Add `pydantic.Field` → `pydantic.Field`
  - [x] 2.5.2 Add `pydantic.ValidationError` → `pydantic.ValidationError`
  - [x] 2.5.3 Add `pydantic.ConfigDict` → `pydantic.ConfigDict`
  - [x] 2.5.4 Add `pydantic.field_validator` → `pydantic.field_validator`
  - [x] 2.5.5 Add `pydantic.model_validator` → `pydantic.model_validator`
  - [x] 2.5.6 Add `pydantic.Field` → `pydantic.fields.Field` (alternative import)
  - [x] 2.5.7 Add `pydantic.AliasChoices` → `pydantic.AliasChoices`
  - [x] 2.5.8 Add `pydantic.TypeAdapter` → `pydantic.TypeAdapter`

- [x] 2.6 Add typing_extensions types (5 types)
  - [x] 2.6.1 Add `typing_extensions.Self` → `typing_extensions.Self`
  - [x] 2.6.2 Add `typing_extensions.TypeAlias` → `typing_extensions.TypeAlias`
  - [x] 2.6.3 Add `typing_extensions.TypedDict` → `typing_extensions.TypedDict`
  - [x] 2.6.4 Add `typing_extensions.Annotated` → `typing_extensions.Annotated`
  - [x] 2.6.5 Add `typing_extensions.NotRequired` → `typing_extensions.NotRequired`

- [x] 2.7 Add standard library types (10 types)
  - [x] 2.7.1 Add `collections.defaultdict` → `collections.defaultdict`
  - [x] 2.7.2 Add `collections.Counter` → `collections.Counter`
  - [x] 2.7.3 Add `collections.OrderedDict` → `collections.OrderedDict`
  - [x] 2.7.4 Add `collections.deque` → `collections.deque`
  - [x] 2.7.5 Add `pathlib.Path` → `pathlib.Path`
  - [x] 2.7.6 Add `pathlib.PurePath` → `pathlib.PurePath`
  - [x] 2.7.7 Add `datetime.datetime` → `datetime.datetime`
  - [x] 2.7.8 Add `datetime.date` → `datetime.date`
  - [x] 2.7.9 Add `datetime.timedelta` → `datetime.timedelta`
  - [x] 2.7.10 Add `uuid.UUID` → `uuid.UUID`

- [x] 2.8 Organize and format dictionary
  - [x] 2.8.1 Group entries by category (numpy, pyarrow, pydantic, stdlib, custom)
  - [x] 2.8.2 Add inline comments documenting each category
  - [x] 2.8.3 Sort entries within each category alphabetically
  - [x] 2.8.4 Verify no duplicate keys
  - [x] 2.8.5 Verify all values are valid fully-qualified names

## 3. Update Sphinx Configuration (`docs/conf.py`)

- [ ] 3.1 Expand intersphinx mappings
  - [x] 3.1.1 Add `scipy` intersphinx mapping: `"scipy": ("https://docs.scipy.org/doc/scipy/", None)`
  - [x] 3.1.2 Add `pandas` intersphinx mapping: `"pandas": ("https://pandas.pydata.org/docs/", None)`
  - [x] 3.1.3 Add `httpx` intersphinx mapping: `"httpx": ("https://www.python-httpx.org/", None)`
  - [x] 3.1.4 Add `pytest` intersphinx mapping: `"pytest": ("https://docs.pytest.org/en/stable/", None)`
  - [ ] 3.1.5 Test each mapping URL is accessible and has `objects.inv` file
  - [x] 3.1.6 Document why each intersphinx mapping is needed (inline comments)

- [ ] 3.2 Configure AutoAPI to exclude `exceptions.py`
  - [x] 3.2.1 Add `autoapi_ignore` configuration if not already present
  - [x] 3.2.2 Add `"*/kgfoundry_common/exceptions.py"` to ignore list
  - [x] 3.2.3 Add inline comment explaining why (legacy aliases, use errors.py as canonical)
  - [x] 3.2.4 Verify pattern matches correctly (test with glob patterns)

- [ ] 3.3 Add `extlinks` for fallback resolution
  - [x] 3.3.1 Add `extlinks` configuration dict if not already present
  - [x] 3.3.2 Add `"numpy-type"` link template: `("https://numpy.org/doc/stable/reference/generated/%s.html", "%s")`
  - [x] 3.3.3 Add `"pyarrow-type"` link template: `("https://arrow.apache.org/docs/python/generated/%s.html", "%s")`
  - [x] 3.3.4 Add inline comment explaining usage (manual fallback for types not in intersphinx)
  - [x] 3.3.5 Test link templates with sample type names

- [ ] 3.4 Document configuration changes
  - [ ] 3.4.1 Add docstring to `intersphinx_mapping` explaining purpose
  - [ ] 3.4.2 Add docstring to `autoapi_ignore` explaining exception exclusion
  - [ ] 3.4.3 Add docstring to `extlinks` explaining fallback mechanism
  - [ ] 3.4.4 Update module-level docstring mentioning cross-reference resolution

## 4. Testing & Validation

- [ ] 4.1 Create test file `tests/unit/test_type_resolution.py`
  - [x] 4.1.1 Import necessary modules and fixtures
  - [x] 4.1.2 Create fixture for QUALIFIED_NAME_OVERRIDES dictionary

- [ ] 4.2 Test numpy type resolution
  - [x] 4.2.1 Test all numpy scalar types have mappings
  - [x] 4.2.2 Test numpy short aliases (`np.`) resolve correctly
  - [x] 4.2.3 Test numpy typing types have mappings
  - [ ] 4.2.4 Test numpy types resolve without warnings

- [ ] 4.3 Test pyarrow type resolution
  - [x] 4.3.1 Test all pyarrow core types have mappings
  - [x] 4.3.2 Test pyarrow.Schema and pyarrow.schema both resolve
  - [ ] 4.3.3 Test pyarrow types resolve without warnings

- [ ] 4.4 Test pydantic type resolution
  - [x] 4.4.1 Test all pydantic types have mappings
  - [x] 4.4.2 Test pydantic import variations resolve correctly
  - [ ] 4.4.3 Test pydantic types resolve without warnings

- [ ] 4.5 Test custom type resolution
  - [x] 4.5.1 Test VecArray, StrArray, FloatArray, IntArray resolve
  - [x] 4.5.2 Test custom exception types resolve
  - [x] 4.5.3 Test custom model types (Doc, Chunk, Concept) resolve

- [ ] 4.6 Test duplicate-target resolution
  - [x] 4.6.1 Test DownloadError has single canonical target
  - [x] 4.6.2 Test UnsupportedMIMEError has single canonical target
  - [x] 4.6.3 Test exceptions.py is excluded from AutoAPI
  - [ ] 4.6.4 Test no duplicate-target warnings for these exceptions

- [ ] 4.7 Test intersphinx mappings
  - [ ] 4.7.1 Test each intersphinx URL is accessible
  - [ ] 4.7.2 Test each intersphinx inventory file exists
  - [ ] 4.7.3 Test cross-references to external docs resolve
  - [x] 4.7.4 Test fallback extlinks work for types not in inventories

## 5. Documentation Build & Verification

- [ ] 5.1 Build documentation with warnings
  - [ ] 5.1.1 Run `make html` to build Sphinx documentation
  - [ ] 5.1.2 Capture all warnings to file: `make html 2>&1 | tee build_warnings.txt`
  - [ ] 5.1.3 Count unresolved reference warnings before changes (baseline)
  - [ ] 5.1.4 Count duplicate-target warnings before changes (baseline)

- [ ] 5.2 Verify warning reduction
  - [ ] 5.2.1 Count unresolved reference warnings after changes
  - [ ] 5.2.2 Verify zero warnings for numpy types
  - [ ] 5.2.3 Verify zero warnings for pyarrow types
  - [ ] 5.2.4 Verify zero warnings for pydantic types
  - [ ] 5.2.5 Count duplicate-target warnings after changes (should be zero)

- [ ] 5.3 Build with strict validation
  - [ ] 5.3.1 Run `SPHINXOPTS="-W" make html` to build with warnings-as-errors
  - [ ] 5.3.2 Verify build succeeds (no errors)
  - [ ] 5.3.3 Check build time compared to baseline
  - [ ] 5.3.4 Verify no degradation in build performance

- [ ] 5.4 Verify generated documentation quality
  - [ ] 5.4.1 Open API page for function with numpy.typing.NDArray parameter
  - [ ] 5.4.2 Verify type reference is hyperlinked (not plain text)
  - [ ] 5.4.3 Click link and verify it goes to correct documentation
  - [ ] 5.4.4 Test pyarrow type links
  - [ ] 5.4.5 Test pydantic type links
  - [ ] 5.4.6 Verify DownloadError links to errors.py (not exceptions.py)
  - [ ] 5.4.7 Verify UnsupportedMIMEError links to errors.py (not exceptions.py)

## 6. Quality Assurance

- [ ] 6.1 Code quality checks
  - [x] 6.1.1 Run `ruff check tools/auto_docstrings.py` to verify linting passes
  - [x] 6.1.2 Run `ruff format tools/auto_docstrings.py` to verify formatting
  - [ ] 6.1.3 Run `mypy tools/auto_docstrings.py` to verify type checking passes
  - [ ] 6.1.4 Run `black tools/auto_docstrings.py` as safety net
  - [ ] 6.1.5 Fix any linting/formatting/type issues

- [ ] 6.2 Test suite verification
  - [x] 6.2.1 Run `pytest tests/unit/test_type_resolution.py -v`
  - [x] 6.2.2 Verify 100% test pass rate
  - [ ] 6.2.3 Check test coverage for modified code (aim for >95%)
  - [ ] 6.2.4 Add any missing test cases discovered

- [ ] 6.3 Integration testing
  - [ ] 6.3.1 Run full test suite: `pytest` to ensure no regressions
  - [ ] 6.3.2 Run pre-commit hooks: `pre-commit run --all-files`
  - [ ] 6.3.3 Verify all hooks pass (ruff, black, mypy, etc.)
  - [ ] 6.3.4 Fix any issues discovered by pre-commit hooks

- [ ] 6.4 Documentation review
  - [ ] 6.4.1 Review inline comments in `QUALIFIED_NAME_OVERRIDES` for clarity
  - [ ] 6.4.2 Review `docs/conf.py` comments for accuracy
  - [ ] 6.4.3 Review module docstrings for completeness
  - [ ] 6.4.4 Verify all configuration changes are documented

## 7. Final Validation & Cleanup

- [ ] 7.1 Compare before/after metrics
  - [ ] 7.1.1 Count unresolved reference warnings before: ___
  - [ ] 7.1.2 Count unresolved reference warnings after: (should be 0)
  - [ ] 7.1.3 Count duplicate-target warnings before: ___
  - [ ] 7.1.4 Count duplicate-target warnings after: (should be 0)
  - [ ] 7.1.5 Count types in QUALIFIED_NAME_OVERRIDES before: 57
  - [ ] 7.1.6 Count types in QUALIFIED_NAME_OVERRIDES after: (should be 100+)
  - [ ] 7.1.7 Document metrics in commit message

- [ ] 7.2 Verify success criteria
  - [ ] 7.2.1 ✅ Zero unresolved reference warnings for numpy types
  - [ ] 7.2.2 ✅ Zero unresolved reference warnings for pyarrow types
  - [ ] 7.2.3 ✅ Zero unresolved reference warnings for pydantic types
  - [ ] 7.2.4 ✅ Zero duplicate-target warnings for exceptions
  - [ ] 7.2.5 ✅ All type links work in generated documentation
  - [ ] 7.2.6 ✅ Tests pass validating type resolution
  - [ ] 7.2.7 ✅ `-W` build succeeds (warnings as errors mode)

- [ ] 7.3 Documentation and commit
  - [ ] 7.3.1 Review all modified files
  - [ ] 7.3.2 Stage changes: `git add tools/auto_docstrings.py docs/conf.py tests/unit/test_type_resolution.py`
  - [ ] 7.3.3 Write comprehensive commit message with before/after metrics
  - [ ] 7.3.4 Include examples of resolved types in commit message
  - [ ] 7.3.5 Commit changes

- [ ] 7.4 Post-implementation verification
  - [ ] 7.4.1 Build documentation on clean tree: `make html`
  - [ ] 7.4.2 Verify zero unresolved reference warnings
  - [ ] 7.4.3 Verify zero duplicate-target warnings
  - [ ] 7.4.4 Spot-check random type links for quality
  - [ ] 7.4.5 Run full test suite one more time
  - [ ] 7.4.6 Confirm all success criteria met

## Success Criteria

✅ **All tasks marked complete** (111 total tasks)
✅ **Zero unresolved reference warnings** for numpy/pyarrow/pydantic types
✅ **Zero duplicate-target warnings** for DownloadError/UnsupportedMIMEError
✅ **50+ new type mappings** added to QUALIFIED_NAME_OVERRIDES (57 → 100+)
✅ **4 new intersphinx mappings** added (scipy, pandas, httpx, pytest)
✅ **`exceptions.py` excluded** from AutoAPI indexing
✅ **All type links functional** in generated API documentation
✅ **Tests passing** (`pytest tests/unit/test_type_resolution.py`)
✅ **Build succeeds** with `-W` flag (warnings as errors)
✅ **Code quality passes** (ruff, black, mypy, pre-commit)
✅ **No regressions** (full test suite passes)

