# Implementation Tasks

## 1. Sphinx Configuration Hardening

### 1.1 Update Sphinx Extensions
- [ ] 1.1.1 Add `numpydoc` and `numpydoc_validation` extensions to `docs/conf.py`
- [ ] 1.1.2 Configure `numpydoc_validation_checks = {"all"}` for comprehensive validation
- [ ] 1.1.3 Set `napoleon_google_docstring = False` to disable Google-style parsing (if keeping Napoleon)
- [ ] 1.1.4 Set `napoleon_numpy_docstring = True` to enable NumPy parsing
- [ ] 1.1.5 Set `napoleon_use_param = True` and `napoleon_use_rtype = False`
- [ ] 1.1.6 Set `nitpicky = True` to treat cross-reference warnings as errors

### 1.2 Validation Configuration
- [ ] 1.2.1 Configure numpydoc to validate: GL01 (docstring presence), SS01 (summary), ES01 (examples), RT01 (returns), PR01 (parameters)
- [ ] 1.2.2 Create `.numpydoc` config file or configure via `pyproject.toml` if numpydoc supports it
- [ ] 1.2.3 Test Sphinx build with `SPHINXOPTS="-W"` to verify warnings fail build
- [ ] 1.2.4 Document breaking changes in CHANGELOG (module docstrings will change)

## 2. Docstring Generation Pipeline Updates

### 2.1 Template Enhancements
- [ ] 2.1.1 Review all templates in `tools/doq_templates/numpy/`
- [ ] 2.1.2 Update `def.txt` template to include sections: Parameters, Returns, Raises, Examples
- [ ] 2.1.3 Update `class.txt` template to include: Attributes, Methods, Examples
- [ ] 2.1.4 Ensure all templates use NumPy format: `name : type, optional`
- [ ] 2.1.5 Add template for module-level docstrings with standard sections only
- [ ] 2.1.6 Test template rendering with sample functions

### 2.2 doq Configuration
- [ ] 2.2.1 Verify `tools/generate_docstrings.py` calls `doq` with `--formatter numpy`
- [ ] 2.2.2 Verify doq template path points to `tools/doq_templates/numpy/`
- [ ] 2.2.3 Test doq on sample module and verify output format

### 2.3 Auto-Docstring Fallback Updates
- [ ] 2.3.1 Review `tools/auto_docstrings.py` for Google-style section generation
- [ ] 2.3.2 Update fallback generator to emit only NumPy sections (Parameters, Returns, Raises, Examples)
- [ ] 2.3.3 Implement logic to detect function raises and auto-generate Raises section
- [ ] 2.3.4 Implement logic to generate minimal Examples section with `>>>` prompts
- [ ] 2.3.5 Ensure imperative mood for summaries ("Return config" not "Returns config")
- [ ] 2.3.6 Test on sample functions without docstrings

### 2.4 Integration Testing
- [ ] 2.4.1 Run `make docstrings` on a test module and verify NumPy output
- [ ] 2.4.2 Verify no Google-style `Args:` sections present
- [ ] 2.4.3 Verify no TODO placeholders remain in generated docstrings

## 3. NavMap System Refactoring

### 3.1 Remove NavMap Injection from Docstrings
- [ ] 3.1.1 Review `tools/update_navmaps.py` current behavior
- [ ] 3.1.2 Comment out or remove the code that appends `NavMap:` sections to module docstrings
- [ ] 3.1.3 Test that `make docstrings` no longer injects NavMap sections
- [ ] 3.1.4 Run full docstring pipeline and verify navigation still works via `site/_build/navmap/navmap.json`

### 3.2 Strip Existing NavMap Sections
- [ ] 3.2.1 Create script to scan all Python files in `src/` for `NavMap:` sections
- [ ] 3.2.2 Remove all `NavMap:` lines from existing module docstrings
- [ ] 3.2.3 Verify module docstrings only contain standard NumPy sections
- [ ] 3.2.4 Run pydocstyle to verify no errors from removed sections

### 3.3 Verify NavMap Metadata Intact
- [ ] 3.3.1 Ensure all modules have `__navmap__` dictionaries or anchor comments
- [ ] 3.3.2 Run `python tools/navmap/build_navmap.py` and verify `site/_build/navmap/navmap.json` is complete
- [ ] 3.3.3 Run `python tools/navmap/check_navmap.py` and verify all validations pass

## 4. Validation Pipeline Hardening

### 4.1 Ruff Configuration Updates
- [ ] 4.1.1 Update `pyproject.toml` `[tool.ruff.lint]` select to include `"D"` (docstrings)
- [ ] 4.1.2 Add `extend-select = ["D417"]` for D417 (all args documented)
- [ ] 4.1.3 Add `extend-select = ["D401"]` for D401 (imperative mood)
- [ ] 4.1.4 Verify `[tool.ruff.lint.pydocstyle]` has `convention = "numpy"`
- [ ] 4.1.5 Test Ruff with `ruff check --fix src` and verify NumPy compliance

### 4.2 Pydoclint Pre-Commit Integration
- [ ] 4.2.1 Add pydoclint to project dependencies
- [ ] 4.2.2 Update `.pre-commit-config.yaml` to add pydoclint hook
- [ ] 4.2.3 Configure hook: `entry: pydoclint --style numpy src`, `language: system`, `pass_filenames: false`
- [ ] 4.2.4 Place hook after docformatter but before pydocstyle in hook order
- [ ] 4.2.5 Test pre-commit: `pre-commit run pydoclint --all-files`

### 4.3 Interrogate Configuration
- [ ] 4.3.1 Verify interrogate minimum coverage is 90% in `tools/update_docs.sh`
- [ ] 4.3.2 Test interrogate on `src/` and verify it catches undocumented public functions
- [ ] 4.3.3 Document coverage baseline before and after changes

### 4.4 Test Validation Pipeline
- [ ] 4.4.1 Run full `tools/update_docs.sh` and capture any validation failures
- [ ] 4.4.2 Verify Sphinx build with `SPHINXOPTS="-W"` passes (no warnings)
- [ ] 4.4.3 Verify no Google-style sections present in output

## 5. Docstring Content Enhancements

### 5.1 Raises Sections
- [ ] 5.1.1 Audit all public functions in `src/` for exception handling
- [ ] 5.1.2 Add or update Raises sections for functions that raise exceptions
- [ ] 5.1.3 Document exception conditions clearly (e.g., "If input is empty")
- [ ] 5.1.4 Verify pydoclint accepts all Raises sections

### 5.2 Examples Sections
- [ ] 5.2.1 Add minimal Examples to all public API functions
- [ ] 5.2.2 Ensure examples use `>>>` prompt for doctestable format
- [ ] 5.2.3 Test with `pytest --doctest-modules src/` to verify examples run
- [ ] 5.2.4 Keep examples minimal (<10 lines where possible)

### 5.3 Attributes Sections for Classes
- [ ] 5.3.1 Audit all public classes in `src/` for attributes
- [ ] 5.3.2 Add Attributes section to class docstrings
- [ ] 5.3.3 Document type, shape/constraints, and mutability
- [ ] 5.3.4 Verify format: `name : type` with description

### 5.4 See Also Sections
- [ ] 5.4.1 Identify related functions/classes for each public API item
- [ ] 5.4.2 Add See Also sections with cross-references
- [ ] 5.4.3 Verify Sphinx can resolve all cross-references
- [ ] 5.4.4 Test hyperlinks in generated HTML

### 5.5 Notes Sections (Contracts/Constraints)
- [ ] 5.5.1 Add Notes sections for functions with non-obvious constraints
- [ ] 5.5.2 Document input requirements, complexity, thread-safety where applicable
- [ ] 5.5.3 Keep notes concise (1-3 sentences)
- [ ] 5.5.4 Use standard terminology for complexity (O(n log n), etc.)

### 5.6 Type Annotation Consistency
- [ ] 5.6.1 Review all docstring types for NumPy format
- [ ] 5.6.2 Convert generic types: `list` → `List`, `dict` → `Mapping[str, Any]`
- [ ] 5.6.3 Use union syntax: `str | int` instead of `str or int`
- [ ] 5.6.4 Document complex shapes: `ndarray of shape (n, d)` not just `ndarray`

## 6. Testing and Validation

### 6.1 Pre-Commit Hook Testing
- [ ] 6.1.1 Install pre-commit hook: `pre-commit install`
- [ ] 6.1.2 Run all hooks: `pre-commit run --all-files`
- [ ] 6.1.3 Fix any failures (Ruff, Black, Mypy, docformatter, pydocstyle, pydoclint, interrogate)
- [ ] 6.1.4 Verify navmap-check and navmap-build pass

### 6.2 Documentation Build
- [ ] 6.2.1 Run full pipeline: `tools/update_docs.sh`
- [ ] 6.2.2 Verify all stages pass without errors
- [ ] 6.2.3 Inspect generated HTML in `docs/_build/html/` for proper rendering
- [ ] 6.2.4 Inspect JSON corpus in `docs/_build/json/` for completeness

### 6.3 Doctest Execution
- [ ] 6.3.1 Run `pytest --doctest-modules src/` to execute all docstring examples
- [ ] 6.3.2 Fix any failing examples or add missing output
- [ ] 6.3.3 Document expected test runtime

### 6.4 Validation Rule Spot Checks
- [ ] 6.4.1 Pick 3-5 sample modules and manually verify docstring compliance
- [ ] 6.4.2 Check: all Parameters documented, all Returns documented, Raises present if applicable
- [ ] 6.4.3 Verify no Google-style sections, no NavMap: lines
- [ ] 6.4.4 Verify examples are runnable

## 7. Documentation and Migration

### 7.1 Update Developer Docs
- [ ] 7.1.1 Update `README-AUTOMATED-DOCUMENTATION.md` to reflect numpydoc configuration
- [ ] 7.1.2 Add section on NumPy docstring format requirements
- [ ] 7.1.3 Document the removal of `NavMap:` sections from docstrings
- [ ] 7.1.4 Add examples of compliant docstrings (Parameters, Returns, Raises, Examples, See Also)

### 7.2 Update Contributing Guide
- [ ] 7.2.1 Document NumPy docstring format for contributors
- [ ] 7.2.2 Provide template docstring for functions and classes
- [ ] 7.2.3 Link to NumPy documentation style guide
- [ ] 7.2.4 Document pre-commit hooks that enforce compliance

### 7.3 Create Migration Guide
- [ ] 7.3.1 Document changes in behavior (NavMap: removal, stricter validation)
- [ ] 7.3.2 Provide rollback instructions if needed
- [ ] 7.3.3 List all new tools/configurations introduced
- [ ] 7.3.4 Update CHANGELOG with breaking changes

## 8. Final Validation and Cleanup

### 8.1 Full CI Run
- [ ] 8.1.1 Run all tests: `make test`
- [ ] 8.1.2 Run linting: `make lint`
- [ ] 8.1.3 Run documentation build: `tools/update_docs.sh`
- [ ] 8.1.4 Verify clean git diff (only docstrings changed where expected)

### 8.2 Edge Cases and Special Handling
- [ ] 8.2.1 Audit deprecated functions for `Deprecated` section (Sphinx directive)
- [ ] 8.2.2 Audit version-specific functions for `Versionadded`, `Versionchanged`
- [ ] 8.2.3 Handle property decorators with attribute-style documentation
- [ ] 8.2.4 Handle private/internal functions (different standards or skipped?)

### 8.3 Performance and Scale
- [ ] 8.3.1 Measure documentation build time before and after changes
- [ ] 8.3.2 Verify validation pipeline performance is acceptable
- [ ] 8.3.3 Document any performance trade-offs

### 8.4 Approval and Merge
- [ ] 8.4.1 Create PR with all changes
- [ ] 8.4.2 Ensure CI passes completely
- [ ] 8.4.3 Request review and approval
- [ ] 8.4.4 Address feedback and re-validate
- [ ] 8.4.5 Merge and monitor for any issues
