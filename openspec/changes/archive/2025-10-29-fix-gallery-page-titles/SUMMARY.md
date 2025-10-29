# Fix Gallery Page Titles and Cross-References - Summary

## Overview

This OpenSpec change proposal addresses Sphinx-Gallery warnings that occur when building documentation. The warnings indicate that gallery example pages lack proper titles or cross-reference anchors that Sphinx can resolve.

**Current warnings:**
```
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_00_quickstart.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_10_data_contracts_minimal.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_20_search_smoke.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery__utils.py'
```

**Goal:** Eliminate all gallery-related cross-reference warnings by ensuring example scripts follow Sphinx-Gallery conventions for title extraction and reference label generation.

## Problem Analysis

### Root Causes

1. **Title Format Mismatch**: Current example docstrings use a custom format with `:orphan:`, custom `.. _gallery_*:` labels, and non-standard title structures. Sphinx-Gallery expects the first line of the docstring to be a plain text title followed by an underline of `=` characters.

2. **Reference Label Conflicts**: Custom reference labels (`.. _gallery_quickstart:`) conflict with auto-generated `sphx_glr_` labels, causing Sphinx to fail resolution.

3. **Configuration Gaps**: The `sphinx_gallery_conf` in `docs/conf.py` lacks explicit options for title extraction and reference label handling.

4. **No Validation**: There's no automated check to ensure gallery examples comply with Sphinx-Gallery conventions before documentation builds.

### Impact

* **Build warnings**: ~4 warnings per documentation build
* **Masked regressions**: Warnings make it harder to spot real documentation issues
* **CI/CD blocking**: Cannot gate builds on clean documentation
* **Broken cross-references**: Links to gallery examples from other docs may fail
* **Maintenance burden**: Unclear what format examples should follow

## Solution Design

### 1. Standardize Example Docstring Format

Transform all `examples/*.py` files to follow Sphinx-Gallery conventions:

**Before:**
```python
"""Module description.

.. _gallery_example_name:

Title — Description
===================

...
```

**After:**
```python
"""
Title — Description
===================

Extended description.

.. tags:: tag1, tag2

Constraints
-----------
- Time: <2s
- GPU: no
"""
```

**Key changes:**
* Remove `:orphan:` directive
* Remove custom `.. _gallery_*:` labels
* First line is plain text title (≤79 chars)
* Second line is `=` underline matching title length
* Blank line after underline
* Standard reST section headers (not bold)

### 2. Enhance Sphinx-Gallery Configuration

Update `docs/conf.py` to explicitly configure title extraction:

```python
sphinx_gallery_conf = {
    # ... existing options ...
    "first_notebook_cell": None,  # Don't prepend cells
    "line_numbers": False,        # Cleaner code blocks
    "reference_url": {
        "sphinx_gallery": None,   # Don't link to Sphinx-Gallery docs
    },
}
```

### 3. Create Gallery Validation Tool

Implement `tools/validate_gallery.py` to check:

* Title format (first line + underline)
* No `:orphan:` directive
* No custom reference labels
* Proper reST structure
* Returns exit code 0 (pass) or 1 (fail)

**CLI interface:**
```bash
python tools/validate_gallery.py [--examples-dir PATH] [--strict] [--verbose]
```

### 4. Integrate Validation into Pipeline

Add validation at three checkpoints:

1. **Pre-commit hook**: Block commits with invalid gallery examples
2. **Build script**: Fail `tools/update_docs.sh` if validation fails
3. **CI/CD**: Dedicated step before docs build

### 5. Document Gallery Conventions

Create comprehensive documentation:

* `docs/how-to/contributing-gallery-examples.md` - Complete guide with templates
* `README-AUTOMATED-DOCUMENTATION.md` - Add gallery system section
* Inline comments in `docs/conf.py` - Explain each Sphinx-Gallery option

## Implementation Plan

### Phase 1: Investigation (2-4 hours)
* Research Sphinx-Gallery title extraction mechanism
* Audit current example formats
* Analyze generated `.rst` output
* Identify cross-reference pattern mismatches

### Phase 2: Format Standardization (4-6 hours)
* Create docstring template
* Update `examples/00_quickstart.py`
* Update `examples/10_data_contracts_minimal.py`
* Update `examples/20_search_smoke.py`
* Update `examples/_utils.py`
* Verify doctests still pass

### Phase 3: Configuration Enhancement (2-3 hours)
* Update `sphinx_gallery_conf` in `docs/conf.py`
* Test configuration changes
* Verify zero gallery warnings
* Validate cross-reference resolution

### Phase 4: Validation Tool (4-6 hours)
* Implement `tools/validate_gallery.py`
* Add title format validation
* Add directive checks
* Add CLI with argparse
* Test on valid and invalid examples

### Phase 5: Pipeline Integration (2-3 hours)
* Add pre-commit hook
* Update `tools/update_docs.sh`
* Update CI configuration
* Test failure and success paths

### Phase 6: Documentation (2-3 hours)
* Create `docs/how-to/contributing-gallery-examples.md`
* Update `README-AUTOMATED-DOCUMENTATION.md`
* Update project README
* Add navmap metadata

### Phase 7: Testing (3-4 hours)
* End-to-end build verification
* Cross-reference testing
* Validation script testing
* Pre-commit hook testing
* CI/CD pipeline testing
* Doctest compatibility
* Regression testing

### Phase 8: Cleanup (1-2 hours)
* Code formatting and linting
* Type hints and docstrings
* Documentation proofreading
* Final verification
* OpenSpec archival preparation

**Total estimated effort:** 20-31 hours across 8 phases

## Expected Outcomes

### Immediate Benefits

* ✅ **Zero gallery warnings**: All `sphx_glr_` cross-references resolve
* ✅ **Clean builds**: Can gate CI on warning-free documentation
* ✅ **Automated validation**: Pre-commit and CI enforce format compliance
* ✅ **Clear standards**: Documented template and conventions for contributors

### Long-Term Benefits

* 📊 **Better regression detection**: Warnings signal real issues, not format noise
* 🔗 **Reliable cross-references**: Gallery links work throughout documentation
* 📚 **Improved contributor experience**: Clear format rules with automated checks
* 🛠️ **Maintainability**: Validation tool catches format drift early

### Non-Goals

* ❌ Changing gallery page URLs or functionality
* ❌ Executing examples during Sphinx build (remains disabled)
* ❌ Adding new gallery features beyond title/reference fixes

## Files Changed

### Modified Files (11)

1. `examples/00_quickstart.py` - Reformat docstring
2. `examples/10_data_contracts_minimal.py` - Reformat docstring
3. `examples/20_search_smoke.py` - Reformat docstring
4. `examples/_utils.py` - Reformat docstring
5. `examples/GALLERY_HEADER.rst` - Minor updates (if needed)
6. `docs/conf.py` - Enhance `sphinx_gallery_conf`
7. `.pre-commit-config.yaml` - Add gallery validation hook
8. `tools/update_docs.sh` - Add validation step
9. `README-AUTOMATED-DOCUMENTATION.md` - Add gallery section
10. `docs/how-to/index.md` - Link to new gallery guide
11. CI configuration file - Add validation step

### New Files (2)

1. `tools/validate_gallery.py` - Gallery validation script
2. `docs/how-to/contributing-gallery-examples.md` - Gallery contribution guide

### Affected Specs (1)

1. `openspec/specs/documentation-generation/spec.md` - Add gallery validation requirements

## Risks and Mitigations

### Risk: Title extraction fails after format change

**Mitigation:**
* Test title extraction with minimal example before mass updates
* Verify Sphinx-Gallery version compatibility
* Document exact format requirements with examples
* Validation tool enforces correct format

### Risk: Existing cross-references to gallery pages break

**Mitigation:**
* Sphinx-Gallery generates consistent `sphx_glr_` labels from filenames
* Filenames remain unchanged, so labels remain stable
* Test all cross-references after changes
* Regression test suite verifies link stability

### Risk: Validation tool has false positives

**Mitigation:**
* Comprehensive testing with edge cases
* Clear error messages with fix instructions
* `--strict` flag for optional stricter rules
* Can be temporarily bypassed in emergencies (pre-commit skip)

### Risk: CI pipeline becomes brittle

**Mitigation:**
* Validation step is isolated and fast (<1 second)
* Clear exit codes and error messages
* Failing examples can be temporarily excluded in config
* Validation tool is well-tested before integration

## Validation Checklist

Before archiving this change, verify:

- [ ] All gallery examples pass `tools/validate_gallery.py`
- [ ] `sphinx-build -b html docs docs/_build/html` produces zero gallery warnings
- [ ] All `:ref:sphx_glr_gallery_*` cross-references resolve correctly
- [ ] Gallery index page renders correctly with thumbnails and download buttons
- [ ] All doctests pass: `pytest --doctest-modules examples/`
- [ ] Pre-commit hook blocks commits with invalid gallery examples
- [ ] CI pipeline fails on gallery validation errors
- [ ] Documentation guide is complete and accurate
- [ ] All code passes `ruff`, `mypy`, and pre-commit hooks

## Related Changes

This proposal complements other documentation improvements:

* **`update-numpy-docstring-compliance`**: Ensures docstrings are NumPy-style compliant
* **`fix-extended-summary-warnings`**: Fixes ES01 warnings for magic methods
* **`fix-unresolved-cross-references`**: Resolves type name cross-references
* **`enhance-readme-generation`**: Improves package-level documentation

Together, these changes achieve a clean, warning-free documentation build that serves both human readers and AI agents effectively.

## Next Steps

1. **Review** the proposal, tasks, and spec deltas
2. **Approve** if scope and approach are acceptable
3. **Implement** following the 8-phase plan systematically
4. **Validate** using the checklist above
5. **Archive** the change with `openspec archive fix-gallery-page-titles`

---

**Proposal Status:** ✅ Valid (validated with `openspec validate --strict`)

**Change ID:** `fix-gallery-page-titles`

**Created:** 2025-10-28

**Target Spec:** `openspec/specs/documentation-generation/spec.md`

