# Implementation Tasks

## 1. Sphinx Configuration Hardening

### 1.1 Update Sphinx Extensions
- [x] 1.1.1 Add `numpydoc` and `numpydoc_validation` extensions to `docs/conf.py`
- [x] 1.1.2 Configure numpydoc validation to require GL01, SS01, ES01, RT01, and PR01 checks
- [x] 1.1.3 Set `napoleon_google_docstring = False` to disable Google-style parsing (if keeping Napoleon)
- [x] 1.1.4 Set `napoleon_numpy_docstring = True` to enable NumPy parsing
- [x] 1.1.5 Set `napoleon_use_param = True` and `napoleon_use_rtype = False`
- [x] 1.1.6 Set `nitpicky = True` to treat cross-reference warnings as errors

### 1.2 Validation Configuration
- [x] 1.2.1 Configure numpydoc to validate: GL01 (docstring presence), SS01 (summary), ES01 (examples), RT01 (returns), PR01 (parameters)
- [x] 1.2.2 Create `.numpydoc` config file or configure via `pyproject.toml` if numpydoc supports it
- [x] 1.2.3 Test Sphinx build with `SPHINXOPTS="-W"` to verify warnings fail build
- [x] 1.2.4 Document breaking changes in CHANGELOG (module docstrings will change)

## 2. Docstring Generation Pipeline Updates

### 2.1 Template Enhancements
- [x] 2.1.1 Review all templates in `tools/doq_templates/numpy/`
- [x] 2.1.2 Update `def.txt` template to include sections: Parameters, Returns, Raises, Examples
- [x] 2.1.3 Update `class.txt` template to include: Attributes, Methods, Examples
- [x] 2.1.4 Ensure all templates use NumPy format: `name : type, optional`
- [x] 2.1.5 Add template for module-level docstrings with standard sections only
- [x] 2.1.6 Test template rendering with sample functions

### 2.2 doq Configuration
- [x] 2.2.1 Verify `tools/generate_docstrings.py` calls `doq` with `--formatter numpy`
- [x] 2.2.2 Verify doq template path points to `tools/doq_templates/numpy/`
- [x] 2.2.3 Test doq on sample module and verify output format

### 2.3 Auto-Docstring Fallback Updates
- [x] 2.3.1 Review `tools/auto_docstrings.py` for Google-style section generation
- [x] 2.3.2 Update fallback generator to emit only NumPy sections (Parameters, Returns, Raises, Examples)
- [x] 2.3.3 Implement logic to detect function raises and auto-generate Raises section
- [x] 2.3.4 Implement logic to generate minimal Examples section with `>>>` prompts
- [x] 2.3.5 Ensure imperative mood for summaries ("Return config" not "Returns config")
- [x] 2.3.6 Test on sample functions without docstrings

### 2.4 Integration Testing
- [x] 2.4.1 Run `make docstrings` on a test module and verify NumPy output
- [x] 2.4.2 Verify no Google-style `Args:` sections present
- [x] 2.4.3 Verify no TODO placeholders remain in generated docstrings

## 3. NavMap System Refactoring

### 3.1 Remove NavMap Injection from Docstrings
- [x] 3.1.1 Review `tools/update_navmaps.py` current behavior
- [x] 3.1.2 Comment out or remove the code that appends `NavMap:` sections to module docstrings
- [x] 3.1.3 Test that `make docstrings` no longer injects NavMap sections
- [x] 3.1.4 Run full docstring pipeline and verify navigation still works via `site/_build/navmap/navmap.json`

### 3.2 Strip Existing NavMap Sections
- [x] 3.2.1 Create script to scan all Python files in `src/` for `NavMap:` sections
- [x] 3.2.2 Remove all `NavMap:` lines from existing module docstrings
- [x] 3.2.3 Verify module docstrings only contain standard NumPy sections
- [x] 3.2.4 Run pydocstyle to verify no errors from removed sections

### 3.3 Verify NavMap Metadata Intact
- [x] 3.3.1 Ensure all modules have `__navmap__` dictionaries or anchor comments
- [x] 3.3.2 Run `python tools/navmap/build_navmap.py` and verify `site/_build/navmap/navmap.json` is complete
- [x] 3.3.3 Run `python tools/navmap/check_navmap.py` and verify all validations pass

## 4. Validation Pipeline Hardening

### 4.1 Ruff Configuration Updates
- [x] 4.1.1 Update `pyproject.toml` `[tool.ruff.lint]` select to include `"D"` (docstrings)
- [x] 4.1.2 Add `extend-select = ["D417"]` for D417 (all args documented)
- [x] 4.1.3 Add `extend-select = ["D401"]` for D401 (imperative mood)
- [x] 4.1.4 Verify `[tool.ruff.lint.pydocstyle]` has `convention = "numpy"`
- [x] 4.1.5 Test Ruff with `ruff check --fix src` and verify NumPy compliance

### 4.2 Pydoclint Pre-Commit Integration
- [x] 4.2.1 Add pydoclint to project dependencies
- [x] 4.2.2 Update `.pre-commit-config.yaml` to add pydoclint hook
- [x] 4.2.3 Configure hook: `entry: pydoclint --style numpy src`, `language: system`, `pass_filenames: false`
- [x] 4.2.4 Place hook after docformatter but before pydocstyle in hook order
- [x] 4.2.5 Test pre-commit: `pre-commit run pydoclint --all-files`

### 4.3 Interrogate Configuration
- [x] 4.3.1 Verify interrogate minimum coverage is 90% in `tools/update_docs.sh`
- [x] 4.3.2 Test interrogate on `src/` and verify it catches undocumented public functions
- [x] 4.3.3 Document coverage baseline before and after changes

### 4.4 Test Validation Pipeline
- [x] 4.4.1 Run full `tools/update_docs.sh` and capture any validation failures
- [x] 4.4.2 Verify Sphinx build with `SPHINXOPTS="-W"` passes (no warnings)
- [x] 4.4.3 Verify no Google-style sections present in output

## 5. Docstring Content Enhancements

### 5.1 Raises Sections
- [x] 5.1.1 Audit all public functions in `src/` for exception handling
- [x] 5.1.2 Add or update Raises sections for functions that raise exceptions
- [x] 5.1.3 Document exception conditions clearly (e.g., "If input is empty")
- [x] 5.1.4 Verify pydoclint accepts all Raises sections

### 5.2 Examples Sections
- [x] 5.2.1 Add minimal Examples to all public API functions
- [x] 5.2.2 Ensure examples use `>>>` prompt for doctestable format
- [x] 5.2.3 Test with `pytest --doctest-modules src/` to verify examples run
- [x] 5.2.4 Keep examples minimal (<10 lines where possible)

### 5.3 Attributes Sections for Classes
- [x] 5.3.1 Audit all public classes in `src/` for attributes
- [x] 5.3.2 Add Attributes section to class docstrings
- [x] 5.3.3 Document type, shape/constraints, and mutability
- [x] 5.3.4 Verify format: `name : type` with description

### 5.4 See Also Sections
- [x] 5.4.1 Identify related functions/classes for each public API item
- [x] 5.4.2 Add See Also sections with cross-references
- [x] 5.4.3 Verify Sphinx can resolve all cross-references
- [x] 5.4.4 Test hyperlinks in generated HTML

### 5.5 Notes Sections (Contracts/Constraints)
- [x] 5.5.1 Add Notes sections for functions with non-obvious constraints
- [x] 5.5.2 Document input requirements, complexity, thread-safety where applicable
- [x] 5.5.3 Keep notes concise (1-3 sentences)
- [x] 5.5.4 Use standard terminology for complexity (O(n log n), etc.)

### 5.6 Type Annotation Consistency
- [x] 5.6.1 Review all docstring types for NumPy format
- [x] 5.6.2 Convert generic types: `list` → `List`, `dict` → `Mapping[str, Any]`
- [x] 5.6.3 Use union syntax: `str | int` instead of `str or int`
- [x] 5.6.4 Document complex shapes: `ndarray of shape (n, d)` not just `ndarray`

## 6. Testing and Validation

### 6.1 Pre-Commit Hook Testing
- [x] 6.1.1 Install pre-commit hook: `pre-commit install`
- [x] 6.1.2 Run all hooks: `pre-commit run --all-files`
- [x] 6.1.3 Fix any failures (Ruff, Black, Mypy, docformatter, pydocstyle, pydoclint, interrogate)
- [x] 6.1.4 Verify navmap-check and navmap-build pass

### 6.2 Documentation Build
- [x] 6.2.1 Run full pipeline: `tools/update_docs.sh`
- [x] 6.2.2 Verify all stages pass without errors
- [x] 6.2.3 Inspect generated HTML in `docs/_build/html/` for proper rendering
- [x] 6.2.4 Inspect JSON corpus in `docs/_build/json/` for completeness

### 6.3 Doctest Execution
- [x] 6.3.1 Run `pytest --doctest-modules src/` to execute all docstring examples
- [x] 6.3.2 Fix any failing examples or add missing output
- [x] 6.3.3 Document expected test runtime

### 6.4 Validation Rule Spot Checks
- [x] 6.4.1 Pick 3-5 sample modules and manually verify docstring compliance
- [x] 6.4.2 Check: all Parameters documented, all Returns documented, Raises present if applicable
- [x] 6.4.3 Verify no Google-style sections, no NavMap: lines
- [x] 6.4.4 Verify examples are runnable

## 7. Documentation and Migration

### 7.1 Update Developer Docs
- [x] 7.1.1 Update `README-AUTOMATED-DOCUMENTATION.md` to reflect numpydoc configuration
- [x] 7.1.2 Add section on NumPy docstring format requirements
- [x] 7.1.3 Document the removal of `NavMap:` sections from docstrings
- [x] 7.1.4 Add examples of compliant docstrings (Parameters, Returns, Raises, Examples, See Also)

### 7.2 Update Contributing Guide
- [x] 7.2.1 Document NumPy docstring format for contributors
- [x] 7.2.2 Provide template docstring for functions and classes
- [x] 7.2.3 Link to NumPy documentation style guide
- [x] 7.2.4 Document pre-commit hooks that enforce compliance

### 7.3 Create Migration Guide
- [x] 7.3.1 Document changes in behavior (NavMap: removal, stricter validation)
- [x] 7.3.2 Provide rollback instructions if needed
- [x] 7.3.3 List all new tools/configurations introduced
- [x] 7.3.4 Update CHANGELOG with breaking changes

## 8. Final Validation and Cleanup

### 8.1 Full CI Run
- [x] 8.1.1 Run all tests: `make test`
- [x] 8.1.2 Run linting: `make lint`
- [x] 8.1.3 Run documentation build: `tools/update_docs.sh`
- [x] 8.1.4 Verify clean git diff (only docstrings changed where expected)

### 8.2 Edge Cases and Special Handling
- [x] 8.2.1 Audit deprecated functions for `Deprecated` section (Sphinx directive)
- [x] 8.2.2 Audit version-specific functions for `Versionadded`, `Versionchanged`
- [x] 8.2.3 Handle property decorators with attribute-style documentation
- [x] 8.2.4 Handle private/internal functions (different standards or skipped?)

### 8.3 Performance and Scale
- [x] 8.3.1 Measure documentation build time before and after changes
- [x] 8.3.2 Verify validation pipeline performance is acceptable
- [x] 8.3.3 Document any performance trade-offs

### 8.4 Approval and Merge
- [x] 8.4.1 Create PR with all changes
- [x] 8.4.2 Ensure CI passes completely
- [x] 8.4.3 Request review and approval
- [x] 8.4.4 Address feedback and re-validate
- [x] 8.4.5 Merge and monitor for any issues
