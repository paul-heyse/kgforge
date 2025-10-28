# Implementation Tasks

## Phase 1: Investigation and Design (2-4 hours)

### 1.1 Sphinx-Gallery Title Extraction Research
- [ ] 1.1.1 Read Sphinx-Gallery documentation on title extraction from docstrings
- [ ] 1.1.2 Identify the exact pattern Sphinx-Gallery uses for first-line titles
- [ ] 1.1.3 Document how Sphinx-Gallery generates `sphx_glr_` reference labels
- [ ] 1.1.4 Verify whether `:orphan:` directives interfere with title extraction
- [ ] 1.1.5 Test title extraction with a minimal example locally

### 1.2 Current Format Audit
- [ ] 1.2.1 Analyze `examples/00_quickstart.py` docstring structure
- [ ] 1.2.2 Analyze `examples/10_data_contracts_minimal.py` docstring structure
- [ ] 1.2.3 Analyze `examples/20_search_smoke.py` docstring structure
- [ ] 1.2.4 Analyze `examples/_utils.py` docstring structure
- [ ] 1.2.5 Document current vs. required format differences

### 1.3 Generated Output Analysis
- [ ] 1.3.1 Inspect `docs/gallery/00_quickstart.rst` for title structure
- [ ] 1.3.2 Inspect `docs/gallery/10_data_contracts_minimal.rst` for title structure
- [ ] 1.3.3 Inspect `docs/gallery/20_search_smoke.rst` for title structure
- [ ] 1.3.4 Inspect `docs/gallery/_utils.rst` for title structure
- [ ] 1.3.5 Compare generated titles with what the index expects

### 1.4 Cross-Reference Pattern Investigation
- [ ] 1.4.1 Examine `docs/gallery/index.rst` for `:ref:` directive patterns
- [ ] 1.4.2 Identify all `sphx_glr_` labels used in the index
- [ ] 1.4.3 Verify if generated `.rst` files contain matching labels
- [ ] 1.4.4 Document the mismatch between expected and actual labels
- [ ] 1.4.5 Test cross-reference resolution with `sphinx-build -n` (nitpicky mode)

### 1.5 Configuration Gap Analysis
- [ ] 1.5.1 Review current `sphinx_gallery_conf` in `docs/conf.py`
- [ ] 1.5.2 Identify missing configuration options that affect titles
- [ ] 1.5.3 Review Sphinx-Gallery config options for reference label control
- [ ] 1.5.4 Document recommended configuration changes
- [ ] 1.5.5 Create a test configuration to validate fixes

## Phase 2: Example Format Standardization (4-6 hours)

### 2.1 Create Standard Format Template
- [ ] 2.1.1 Design docstring template following Sphinx-Gallery conventions
- [ ] 2.1.2 Include title line (first line, plain text, ≤79 chars)
- [ ] 2.1.3 Include title underline (second line, `=` matching title length)
- [ ] 2.1.4 Include blank line after title
- [ ] 2.1.5 Include longer description paragraph
- [ ] 2.1.6 Include `.. tags::` directive for metadata
- [ ] 2.1.7 Include `Constraints` section for runtime requirements
- [ ] 2.1.8 Add doctest code block if applicable
- [ ] 2.1.9 Document the template in `docs/how-to/contributing-gallery-examples.md`

### 2.2 Update `examples/00_quickstart.py`
- [ ] 2.2.1 Remove `:orphan:` directive from docstring
- [ ] 2.2.2 Remove custom `.. _gallery_quickstart:` reference label
- [ ] 2.2.3 Reformat first line to plain title: `Quickstart — minimal import smoke test`
- [ ] 2.2.4 Add title underline: `===============================================`
- [ ] 2.2.5 Keep description paragraph starting with "Ensure the..."
- [ ] 2.2.6 Keep `.. tags:: getting-started, smoke` directive
- [ ] 2.2.7 Move `**Constraints**` section to `Constraints` header
- [ ] 2.2.8 Keep doctest code block
- [ ] 2.2.9 Verify the file still passes `pytest --doctest-modules examples/00_quickstart.py`

### 2.3 Update `examples/10_data_contracts_minimal.py`
- [ ] 2.3.1 Remove `:orphan:` directive from docstring
- [ ] 2.3.2 Remove custom `.. _gallery_data_contracts_minimal:` reference label
- [ ] 2.3.3 Reformat first line to plain title: `Data contracts — schema export smoke`
- [ ] 2.3.4 Add title underline: `========================================`
- [ ] 2.3.5 Keep description paragraph starting with "Demonstrate..."
- [ ] 2.3.6 Keep `.. tags:: schema, pydantic` directive
- [ ] 2.3.7 Move `**Constraints**` section to `Constraints` header
- [ ] 2.3.8 Keep doctest code block
- [ ] 2.3.9 Verify the file still passes `pytest --doctest-modules examples/10_data_contracts_minimal.py`

### 2.4 Update `examples/20_search_smoke.py`
- [ ] 2.4.1 Remove `:orphan:` directive from docstring
- [ ] 2.4.2 Remove custom `.. _gallery_search_smoke:` reference label (if present)
- [ ] 2.4.3 Reformat first line to plain title (extract from existing docstring)
- [ ] 2.4.4 Add title underline with correct length
- [ ] 2.4.5 Keep description paragraph
- [ ] 2.4.6 Keep or add `.. tags::` directive
- [ ] 2.4.7 Move `**Constraints**` section to `Constraints` header
- [ ] 2.4.8 Keep doctest code block
- [ ] 2.4.9 Verify the file still passes `pytest --doctest-modules examples/20_search_smoke.py`

### 2.5 Update `examples/_utils.py`
- [ ] 2.5.1 Check if `:orphan:` or custom labels are present and remove them
- [ ] 2.5.2 Reformat docstring to start with plain title: `Utility helpers used by Sphinx gallery examples`
- [ ] 2.5.3 Add title underline: `==================================================`
- [ ] 2.5.4 Keep description paragraph starting with "The helpers..."
- [ ] 2.5.5 Add `.. tags:: utils, helpers` directive
- [ ] 2.5.6 Add `Constraints` section if applicable
- [ ] 2.5.7 Verify the file still passes `pytest --doctest-modules examples/_utils.py`

### 2.6 Update `examples/GALLERY_HEADER.rst`
- [ ] 2.6.1 Review current header content
- [ ] 2.6.2 Ensure title matches gallery page structure
- [ ] 2.6.3 Update description if needed to match new format
- [ ] 2.6.4 Remove any `:orphan:` or custom labels
- [ ] 2.6.5 Verify header renders correctly in generated `docs/gallery/index.rst`

## Phase 3: Sphinx-Gallery Configuration Enhancement (2-3 hours)

### 3.1 Update `docs/conf.py` Configuration
- [ ] 3.1.1 Add `"first_notebook_cell": None` to `sphinx_gallery_conf`
- [ ] 3.1.2 Add `"line_numbers": False` to `sphinx_gallery_conf`
- [ ] 3.1.3 Add `"reference_url": {"sphinx_gallery": None}` to `sphinx_gallery_conf`
- [ ] 3.1.4 Add `"capture_repr": ()` to prevent unwanted output capture
- [ ] 3.1.5 Add `"expected_failing_examples": []` for explicit failure tracking
- [ ] 3.1.6 Add `"min_reported_time": 0` to show all execution times
- [ ] 3.1.7 Document each configuration option with inline comments
- [ ] 3.1.8 Verify configuration syntax with Python AST parsing

### 3.2 Test Configuration Changes
- [ ] 3.2.1 Run `sphinx-build -b html docs docs/_build/html` after config changes
- [ ] 3.2.2 Check for new warnings or errors
- [ ] 3.2.3 Verify gallery pages are generated correctly
- [ ] 3.2.4 Verify cross-references resolve (no "Failed to create a cross reference" warnings)
- [ ] 3.2.5 Test thumbnail generation (check `docs/gallery/images/thumb/` directory)

### 3.3 Gallery Index Verification
- [ ] 3.3.1 Inspect generated `docs/gallery/index.rst` structure
- [ ] 3.3.2 Verify `:ref:` directives use correct labels
- [ ] 3.3.3 Verify thumbnail images are present
- [ ] 3.3.4 Verify download links work (.py, .ipynb, .zip)
- [ ] 3.3.5 Test gallery page navigation in rendered HTML

## Phase 4: Gallery Validation Tool (4-6 hours)

### 4.1 Create `tools/validate_gallery.py` Script
- [ ] 4.1.1 Add module docstring explaining purpose and usage
- [ ] 4.1.2 Import necessary modules (Path, ast, re, sys)
- [ ] 4.1.3 Define `GalleryValidationError` exception class
- [ ] 4.1.4 Define `validate_title_format(docstring: str) -> tuple[bool, str]` function
- [ ] 4.1.5 Define `check_orphan_directive(docstring: str) -> bool` function
- [ ] 4.1.6 Define `check_custom_labels(docstring: str) -> list[str]` function
- [ ] 4.1.7 Define `validate_example_file(file_path: Path) -> list[str]` function
- [ ] 4.1.8 Define `main(examples_dir: Path) -> int` function
- [ ] 4.1.9 Add CLI argument parsing with argparse

### 4.2 Implement Title Format Validation
- [ ] 4.2.1 Parse docstring first line as title
- [ ] 4.2.2 Check second line is all `=` characters
- [ ] 4.2.3 Verify underline length matches title length (±1 char tolerance)
- [ ] 4.2.4 Check title length ≤ 79 characters
- [ ] 4.2.5 Verify blank line after underline
- [ ] 4.2.6 Return detailed error message if validation fails

### 4.3 Implement Directive Checks
- [ ] 4.3.1 Check for `:orphan:` directive (should not be present)
- [ ] 4.3.2 Check for custom `.. _gallery_*:` labels (should not be present)
- [ ] 4.3.3 Check for `.. tags::` directive (optional but recommended)
- [ ] 4.3.4 Verify docstring structure follows reST conventions
- [ ] 4.3.5 Return list of found issues

### 4.4 Implement Main Validation Logic
- [ ] 4.4.1 Scan `examples/` directory for `*.py` files
- [ ] 4.4.2 Skip `__pycache__` and hidden files
- [ ] 4.4.3 Extract docstring from each file using AST
- [ ] 4.4.4 Run all validation checks on each file
- [ ] 4.4.5 Collect and aggregate errors
- [ ] 4.4.6 Print summary report
- [ ] 4.4.7 Return exit code 0 (success) or 1 (failures)

### 4.5 Add CLI and Help Text
- [ ] 4.5.1 Add `--examples-dir` argument (default: `examples/`)
- [ ] 4.5.2 Add `--strict` flag for stricter validation
- [ ] 4.5.3 Add `--fix` flag for automatic fixes (future enhancement)
- [ ] 4.5.4 Add `--verbose` flag for detailed output
- [ ] 4.5.5 Add `--help` documentation

### 4.6 Test Validation Script
- [ ] 4.6.1 Run on current `examples/` directory (should fail initially)
- [ ] 4.6.2 Run on updated `examples/` directory (should pass)
- [ ] 4.6.3 Create test examples with known errors
- [ ] 4.6.4 Verify error messages are clear and actionable
- [ ] 4.6.5 Test all CLI flags and options

## Phase 5: Integration into Build Pipeline (2-3 hours)

### 5.1 Add Pre-Commit Hook
- [ ] 5.1.1 Open `.pre-commit-config.yaml`
- [ ] 5.1.2 Add new hook entry under `repos` section
- [ ] 5.1.3 Set `id: validate-gallery`
- [ ] 5.1.4 Set `name: Validate gallery examples`
- [ ] 5.1.5 Set `entry: python tools/validate_gallery.py`
- [ ] 5.1.6 Set `language: system`
- [ ] 5.1.7 Set `files: ^examples/.*\.py$` to trigger only on example changes
- [ ] 5.1.8 Set `pass_filenames: false`
- [ ] 5.1.9 Test hook with `pre-commit run validate-gallery --all-files`

### 5.2 Integrate into `tools/update_docs.sh`
- [ ] 5.2.1 Open `tools/update_docs.sh`
- [ ] 5.2.2 Find the appropriate section for validation steps
- [ ] 5.2.3 Add `python tools/validate_gallery.py` before Sphinx build
- [ ] 5.2.4 Capture exit code and fail build if validation fails
- [ ] 5.2.5 Add informative error message if validation fails
- [ ] 5.2.6 Test the entire `update_docs.sh` script
- [ ] 5.2.7 Verify Sphinx warnings are eliminated

### 5.3 Update CI Pipeline
- [ ] 5.3.1 Identify CI configuration file (e.g., `.github/workflows/*.yml`)
- [ ] 5.3.2 Add `python tools/validate_gallery.py` step before docs build
- [ ] 5.3.3 Set job to fail if validation fails
- [ ] 5.3.4 Add step name: "Validate gallery examples"
- [ ] 5.3.5 Test CI pipeline with intentionally broken example
- [ ] 5.3.6 Verify CI fails appropriately
- [ ] 5.3.7 Test CI pipeline with fixed examples
- [ ] 5.3.8 Verify CI passes and docs build cleanly

## Phase 6: Documentation Updates (2-3 hours)

### 6.1 Create `docs/how-to/contributing-gallery-examples.md`
- [ ] 6.1.1 Add frontmatter and title
- [ ] 6.1.2 Write introduction explaining gallery purpose
- [ ] 6.1.3 Document required docstring format with examples
- [ ] 6.1.4 Show correct title and underline pattern
- [ ] 6.1.5 Explain `.. tags::` directive usage
- [ ] 6.1.6 Document `Constraints` section format
- [ ] 6.1.7 Provide full working example
- [ ] 6.1.8 List common validation errors and fixes
- [ ] 6.1.9 Add section on testing examples locally
- [ ] 6.1.10 Add section on debugging Sphinx-Gallery issues

### 6.2 Update `README-AUTOMATED-DOCUMENTATION.md`
- [ ] 6.2.1 Find the appropriate section for gallery documentation
- [ ] 6.2.2 Add new subsection: "Gallery Examples: `examples/`"
- [ ] 6.2.3 Explain Sphinx-Gallery integration
- [ ] 6.2.4 Document title extraction pattern
- [ ] 6.2.5 Document cross-reference label generation
- [ ] 6.2.6 Link to `docs/how-to/contributing-gallery-examples.md`
- [ ] 6.2.7 Add troubleshooting section for common gallery issues
- [ ] 6.2.8 Update the pipeline diagram to include gallery validation

### 6.3 Update Project README (if applicable)
- [ ] 6.3.1 Check if `README.md` mentions gallery or examples
- [ ] 6.3.2 Add or update section on contributing examples
- [ ] 6.3.3 Link to gallery documentation
- [ ] 6.3.4 Mention validation requirements

### 6.4 Update NavMap Metadata
- [ ] 6.4.1 Add `__navmap__` to `tools/validate_gallery.py`
- [ ] 6.4.2 Set `category: "docs"` and `stability: "stable"`
- [ ] 6.4.3 Run `python tools/update_navmaps.py`
- [ ] 6.4.4 Verify navmap index is updated

## Phase 7: Testing and Validation (3-4 hours)

### 7.1 End-to-End Documentation Build
- [ ] 7.1.1 Clean build artifacts: `rm -rf docs/_build`
- [ ] 7.1.2 Run full documentation build: `bash tools/update_docs.sh`
- [ ] 7.1.3 Verify zero Sphinx warnings related to gallery
- [ ] 7.1.4 Check for any new warnings introduced by changes
- [ ] 7.1.5 Verify gallery index page renders correctly
- [ ] 7.1.6 Verify all example pages render correctly
- [ ] 7.1.7 Test thumbnail images display
- [ ] 7.1.8 Test download buttons work (.py, .ipynb, .zip)

### 7.2 Cross-Reference Testing
- [ ] 7.2.1 Verify `:ref:sphx_glr_gallery_00_quickstart.py` resolves correctly
- [ ] 7.2.2 Verify `:ref:sphx_glr_gallery_10_data_contracts_minimal.py` resolves correctly
- [ ] 7.2.3 Verify `:ref:sphx_glr_gallery_20_search_smoke.py` resolves correctly
- [ ] 7.2.4 Verify `:ref:sphx_glr_gallery__utils.py` resolves correctly
- [ ] 7.2.5 Test cross-references from other documentation pages
- [ ] 7.2.6 Test intersphinx links if applicable

### 7.3 Gallery Validation Script Testing
- [ ] 7.3.1 Run `python tools/validate_gallery.py` on clean examples
- [ ] 7.3.2 Verify exit code 0 (success)
- [ ] 7.3.3 Create test example with missing title underline
- [ ] 7.3.4 Verify validation fails with clear error message
- [ ] 7.3.5 Create test example with `:orphan:` directive
- [ ] 7.3.6 Verify validation fails with appropriate error
- [ ] 7.3.7 Test `--verbose` flag output
- [ ] 7.3.8 Test `--strict` flag behavior

### 7.4 Pre-Commit Hook Testing
- [ ] 7.4.1 Stage changes to an example file
- [ ] 7.4.2 Run `git commit` (should trigger pre-commit)
- [ ] 7.4.3 Verify gallery validation runs automatically
- [ ] 7.4.4 Introduce validation error in staged file
- [ ] 7.4.5 Verify commit is blocked
- [ ] 7.4.6 Fix error and retry commit
- [ ] 7.4.7 Verify commit succeeds

### 7.5 CI/CD Pipeline Testing
- [ ] 7.5.1 Create PR with all changes
- [ ] 7.5.2 Verify CI runs gallery validation step
- [ ] 7.5.3 Verify docs build succeeds
- [ ] 7.5.4 Check CI logs for gallery-related warnings (should be none)
- [ ] 7.5.5 Introduce intentional validation error in PR
- [ ] 7.5.6 Verify CI fails appropriately
- [ ] 7.5.7 Fix error and push again
- [ ] 7.5.8 Verify CI passes

### 7.6 Doctest Compatibility Verification
- [ ] 7.6.1 Run `pytest --doctest-modules examples/00_quickstart.py`
- [ ] 7.6.2 Run `pytest --doctest-modules examples/10_data_contracts_minimal.py`
- [ ] 7.6.3 Run `pytest --doctest-modules examples/20_search_smoke.py`
- [ ] 7.6.4 Run `pytest --doctest-modules examples/_utils.py`
- [ ] 7.6.5 Verify all doctests pass
- [ ] 7.6.6 Run `pytest --doctest-modules examples/` (all at once)
- [ ] 7.6.7 Verify exit code 0

### 7.7 Regression Testing
- [ ] 7.7.1 Compare old vs. new gallery HTML output
- [ ] 7.7.2 Verify URLs remain the same
- [ ] 7.7.3 Verify content is functionally identical
- [ ] 7.7.4 Check that no existing links to gallery pages are broken
- [ ] 7.7.5 Verify JSON schema links still work
- [ ] 7.7.6 Test gallery page search functionality
- [ ] 7.7.7 Verify gallery page meta tags are correct

## Phase 8: Cleanup and Finalization (1-2 hours)

### 8.1 Code Review Preparation
- [ ] 8.1.1 Run `uvx ruff check --fix && uvx ruff format` on all modified files
- [ ] 8.1.2 Run `uvx mypy --strict tools/validate_gallery.py`
- [ ] 8.1.3 Add type hints to all functions in `validate_gallery.py`
- [ ] 8.1.4 Add docstrings to all functions in `validate_gallery.py`
- [ ] 8.1.5 Run pre-commit hooks on all files: `pre-commit run --all-files`

### 8.2 Documentation Review
- [ ] 8.2.1 Proofread `docs/how-to/contributing-gallery-examples.md`
- [ ] 8.2.2 Proofread `README-AUTOMATED-DOCUMENTATION.md` updates
- [ ] 8.2.3 Verify all links work
- [ ] 8.2.4 Check code examples for syntax errors
- [ ] 8.2.5 Verify formatting is consistent

### 8.3 Final Verification
- [ ] 8.3.1 Clean all build artifacts
- [ ] 8.3.2 Run full build pipeline from scratch
- [ ] 8.3.3 Verify zero gallery-related warnings
- [ ] 8.3.4 Verify all tests pass
- [ ] 8.3.5 Verify CI pipeline passes
- [ ] 8.3.6 Review all changed files one more time
- [ ] 8.3.7 Ensure no temporary files or debug code remain

### 8.4 OpenSpec Archival Preparation
- [ ] 8.4.1 Verify all tasks are complete
- [ ] 8.4.2 Update `tasks.md` checkboxes to `[x]` for completed items
- [ ] 8.4.3 Document any deviations from original plan
- [ ] 8.4.4 Prepare for archival with `openspec archive fix-gallery-page-titles`

---

**Total estimated time:** 20-31 hours across 8 phases

**Key milestones:**
1. Phase 2 complete: All examples reformatted
2. Phase 4 complete: Validation tool working
3. Phase 7 complete: Zero gallery warnings in build
4. Phase 8 complete: Ready for PR and archival

