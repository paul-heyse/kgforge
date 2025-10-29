# Fix Gallery Page Titles and Cross-References

## Why

Sphinx builds currently produce warnings for every gallery example:

```
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_00_quickstart.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_10_data_contracts_minimal.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery_20_search_smoke.py'
WARNING: Failed to create a cross reference. A title or caption not found: 'sphx_glr_gallery__utils.py'
```

**Root cause:**

Sphinx-Gallery generates `.rst` files from Python example scripts (`examples/*.py`) into `docs/gallery/`. The gallery index (`docs/gallery/index.rst`) contains `:ref:` directives that try to link to these generated pages using auto-generated reference labels like `sphx_glr_gallery_00_quickstart.py`.

The problem has multiple layers:

1. **Title extraction mismatch**: Sphinx-Gallery attempts to extract titles from the first line of module docstrings, but the current example format uses a custom structure with `:orphan:`, `.. _gallery_*:` labels, and section headers that may not match what the index expects
2. **Reference label inconsistency**: The gallery index uses `:ref:` directives with `sphx_glr_` prefixed labels, but the generated `.rst` files might not have matching labels or the labels might be in the wrong format
3. **Configuration gaps**: The `sphinx_gallery_conf` in `docs/conf.py` doesn't specify title patterns or custom reference label formats
4. **No validation**: There's no automated check to ensure gallery pages have proper titles and cross-reference anchors before building docs

**Impact:**

* Gallery pages render correctly but produce ~4 warnings per build (one per example)
* Warnings mask actual documentation regressions
* CI/CD cannot gate on clean documentation builds
* Cross-references from other parts of the documentation to gallery examples may be broken or unreliable
* The gallery index page has non-functional or poorly-formatted links

## What Changes

### 1. Standardize Gallery Example Format

**BREAKING**: All example scripts in `examples/` must follow the Sphinx-Gallery standard:

```python
"""
Title: <Short Title>
====================

Longer description here.

.. tags:: tag1, tag2

Constraints
-----------
- Time: <2s
- GPU: no
- Network: no
"""
```

**Key requirements:**

* First line must be the title (79 chars max, centered)
* Second line must be `=` underline matching title length
* No `:orphan:` or custom `.. _gallery_*:` labels (Sphinx-Gallery generates these)
* Optional `.. tags::` directive for metadata
* Constraints section can follow

### 2. Update Sphinx-Gallery Configuration

In `docs/conf.py`, enhance `sphinx_gallery_conf`:

```python
sphinx_gallery_conf = {
    "examples_dirs": str(EXAMPLES_DIR),
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "doc_module": tuple(_PACKAGES),
    "run_stale_examples": False,
    "plot_gallery": False,
    "backreferences_dir": "gen_modules/backrefs",
    # NEW: Explicit title extraction
    "first_notebook_cell": None,  # Don't prepend cells
    "line_numbers": False,        # Cleaner output
    # Ensure proper cross-reference labels
    "reference_url": {
        "sphinx_gallery": None,  # Don't link to Sphinx-Gallery docs
    },
}
```

### 3. Create Gallery Header Template

Create `examples/GALLERY_HEADER.rst` (updated):

```rst
Welcome to the kgfoundry examples gallery
=========================================

The snippets below are lightweight, CPU-only demonstrations that mirror the
APIs documented elsewhere in the site. They never execute during the Sphinx
build; execution happens separately under the doctest smoke stage so these
pages render quickly while remaining trustworthy. Use the download buttons to
pull the ``.py`` or ``.ipynb`` versions and adapt them to your needs.
```

### 4. Update All Example Scripts

Transform each `examples/*.py` file to follow the standard format:

* Remove `:orphan:` directives
* Remove custom `.. _gallery_*:` reference labels
* Restructure the docstring to start with a proper title and `=` underline
* Move constraints into a standard `Constraints` section
* Keep `.. tags::` directive for metadata

### 5. Add Gallery Validation Script

Create `tools/validate_gallery.py`:

```python
"""Validate gallery examples for Sphinx-Gallery compatibility.

Ensures:
- Title format is correct (line 1: title, line 2: ===)
- No conflicting reference labels
- No :orphan: directives
- Proper docstring structure
"""
```

### 6. Integrate into Pre-Commit and CI

* Add `validate_gallery.py` as a pre-commit hook
* Run during `tools/update_docs.sh`
* Fail build if validation errors are found

### 7. Update Documentation

* Add `docs/how-to/contributing-gallery-examples.md` explaining the format
* Update `README-AUTOMATED-DOCUMENTATION.md` with gallery requirements
* Document the title extraction pattern and cross-reference behavior

## Impact

**Affected files:**

* `examples/00_quickstart.py` - Reformat docstring
* `examples/10_data_contracts_minimal.py` - Reformat docstring
* `examples/20_search_smoke.py` - Reformat docstring
* `examples/_utils.py` - Reformat docstring
* `examples/GALLERY_HEADER.rst` - Already correct, may need minor updates
* `docs/conf.py` - Enhance `sphinx_gallery_conf`
* `tools/validate_gallery.py` - NEW
* `.pre-commit-config.yaml` - Add gallery validation hook
* `tools/update_docs.sh` - Add validation step
* `docs/how-to/contributing-gallery-examples.md` - NEW
* `README-AUTOMATED-DOCUMENTATION.md` - Add gallery section

**Affected specs:**

* `openspec/specs/documentation-generation/spec.md` - Add gallery validation requirements

**Build impact:**

* Eliminates ~4 warnings per Sphinx build
* Enables clean build gating in CI
* Improves cross-reference reliability
* Makes gallery format explicit and checkable

**User impact:**

* Gallery pages remain functionally identical
* URLs stay the same
* Download buttons still work
* Examples remain doctestable

**Breaking changes:**

* Example script format changes (but examples are internal-only)
* Gallery validation becomes mandatory for new examples
* CI builds will fail if gallery validation fails

