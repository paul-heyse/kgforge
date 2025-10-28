# Spec Deltas for Documentation Generation

## ADDED Requirements

### Requirement: Gallery Example Format Validation

The documentation build system SHALL validate all gallery example scripts for Sphinx-Gallery compatibility before generating documentation.

#### Scenario: Title format validation

- **WHEN** a gallery example script (`examples/*.py`) is processed
- **THEN** the validator SHALL verify:
  - First line of module docstring is a plain text title (≤79 characters)
  - Second line is an underline of `=` characters matching the title length (±1 char)
  - A blank line follows the underline
  - No `:orphan:` directive is present
  - No custom `.. _gallery_*:` reference labels are present

#### Scenario: Validation failure

- **WHEN** a gallery example script fails format validation
- **THEN** the build system SHALL:
  - Print a clear error message identifying the file and issue
  - Return a non-zero exit code
  - Prevent documentation build from proceeding

#### Scenario: Pre-commit validation

- **WHEN** a developer commits changes to `examples/*.py` files
- **THEN** the pre-commit hook SHALL:
  - Run `tools/validate_gallery.py` automatically
  - Block the commit if validation fails
  - Display actionable error messages

### Requirement: Sphinx-Gallery Title Extraction

The documentation build system SHALL configure Sphinx-Gallery to correctly extract titles and generate cross-reference labels from gallery example scripts.

#### Scenario: Title extraction from docstring

- **WHEN** Sphinx-Gallery processes an example script
- **THEN** it SHALL:
  - Extract the first line of the module docstring as the page title
  - Generate a `sphx_glr_gallery_<filename>` reference label
  - Create proper reST header structure in the generated `.rst` file
  - Ensure the title is cross-referenceable via `:ref:` directives

#### Scenario: Gallery index generation

- **WHEN** Sphinx-Gallery generates the gallery index page
- **THEN** it SHALL:
  - Create `:ref:` directives for each example using `sphx_glr_` labels
  - Include thumbnail images
  - Include download buttons for `.py`, `.ipynb`, and `.zip` formats
  - Display titles extracted from example docstrings

#### Scenario: Cross-reference resolution

- **WHEN** Sphinx builds documentation containing gallery references
- **THEN** all `:ref:sphx_glr_gallery_*` directives SHALL resolve without warnings
- **AND** the generated HTML SHALL contain working hyperlinks to gallery pages

### Requirement: Gallery Validation Tool

The project SHALL provide a `tools/validate_gallery.py` script that validates gallery example format compliance.

#### Scenario: Validation script execution

- **WHEN** `python tools/validate_gallery.py` is executed
- **THEN** the script SHALL:
  - Scan all `*.py` files in `examples/` directory
  - Extract module docstrings using AST parsing
  - Validate title format (first line + underline)
  - Check for prohibited directives (`:orphan:`, custom labels)
  - Report all validation errors with file names and line numbers
  - Return exit code 0 if all examples pass, 1 if any fail

#### Scenario: Validation script CLI options

- **WHEN** the validation script is invoked with CLI flags
- **THEN** it SHALL support:
  - `--examples-dir PATH` to specify custom examples directory
  - `--strict` for stricter validation rules
  - `--verbose` for detailed output
  - `--help` for usage documentation

#### Scenario: Validation error reporting

- **WHEN** a gallery example fails validation
- **THEN** the error message SHALL include:
  - File path of the failing example
  - Specific validation rule that failed
  - Current state vs. expected state
  - Actionable fix instructions

## MODIFIED Requirements

### Requirement: Documentation Build Pipeline

The documentation build pipeline SHALL include gallery validation as a mandatory step before Sphinx execution.

**Previous behavior:**
- `tools/update_docs.sh` ran docstring generation, then Sphinx build, without gallery validation

**New behavior:**
- `tools/update_docs.sh` SHALL:
  1. Generate/update docstrings
  2. **Validate gallery examples with `tools/validate_gallery.py`**
  3. **Fail immediately if validation fails**
  4. Run Sphinx build
  5. Generate package READMEs
  6. Update navmap index

#### Scenario: Build failure on invalid gallery

- **WHEN** `tools/update_docs.sh` is executed and gallery validation fails
- **THEN** the script SHALL:
  - Display the validation error output
  - Print a message: "Gallery validation failed. Fix errors before building docs."
  - Exit with non-zero code
  - NOT proceed to Sphinx build

#### Scenario: Clean build with valid gallery

- **WHEN** all gallery examples pass validation
- **THEN** the Sphinx build SHALL:
  - Generate gallery pages without warnings
  - Resolve all `sphx_glr_` cross-references
  - Produce zero "Failed to create a cross reference" warnings for gallery

### Requirement: Gallery Example Docstring Format

Gallery example scripts SHALL follow a standardized docstring format compatible with Sphinx-Gallery title extraction.

**Previous format:**
```python
"""Module description.

.. _gallery_example_name:

Title — Description
===================

Extended description.

.. tags:: tag1, tag2

**Constraints**
...
"""
```

**New format:**
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
...
"""
```

#### Scenario: Docstring structure requirements

- **WHEN** an example script is written or updated
- **THEN** the module docstring SHALL:
  - Start with a plain text title (first line)
  - Include an underline of `=` characters (second line)
  - Contain a blank line after the underline
  - Include an extended description paragraph
  - Optionally include `.. tags::` directive for metadata
  - Use `Constraints` as a section header (not bold `**Constraints**`)
  - Omit `:orphan:` directive
  - Omit custom `.. _gallery_*:` reference labels (Sphinx-Gallery generates these)

#### Scenario: Doctest integration

- **WHEN** an example includes doctest code
- **THEN** the doctest SHALL:
  - Be included in the module docstring or function docstrings
  - Use `>>>` prompts
  - Be executable via `pytest --doctest-modules`
  - Pass all assertions

## ADDED Requirements (Documentation)

### Requirement: Gallery Contribution Guide

The documentation SHALL provide a guide for contributing new gallery examples.

#### Scenario: Guide content

- **WHEN** a developer reads `docs/how-to/contributing-gallery-examples.md`
- **THEN** the guide SHALL explain:
  - Required docstring format with annotated example
  - Title and underline pattern
  - Use of `.. tags::` directive
  - `Constraints` section format
  - Doctest integration
  - Common validation errors and fixes
  - Local testing procedure
  - Debugging Sphinx-Gallery issues

#### Scenario: Template example

- **WHEN** a developer needs to create a new gallery example
- **THEN** the guide SHALL provide a complete template showing:
  - Correct docstring structure
  - Title with proper underline
  - Description paragraph
  - Tags directive
  - Constraints section
  - Doctest code block

### Requirement: Automated Documentation System Documentation

The `README-AUTOMATED-DOCUMENTATION.md` SHALL document the gallery system integration.

#### Scenario: Gallery system section

- **WHEN** a developer reads the automated documentation guide
- **THEN** the gallery section SHALL explain:
  - Sphinx-Gallery integration and configuration
  - Title extraction mechanism
  - Cross-reference label generation (`sphx_glr_` prefix)
  - Validation tool purpose and usage
  - Build pipeline integration
  - Troubleshooting common gallery issues

#### Scenario: Pipeline diagram update

- **WHEN** the documentation pipeline diagram is viewed
- **THEN** it SHALL include:
  - Gallery validation step before Sphinx build
  - Validation tool invocation
  - Failure path if validation fails

## ADDED Requirements (Configuration)

### Requirement: Sphinx-Gallery Configuration Options

The `docs/conf.py` SHALL configure Sphinx-Gallery with options that ensure proper title extraction and cross-reference generation.

#### Scenario: Required configuration options

- **WHEN** `docs/conf.py` is loaded by Sphinx
- **THEN** `sphinx_gallery_conf` SHALL include:
  - `"examples_dirs": str(EXAMPLES_DIR)` - source directory
  - `"gallery_dirs": "gallery"` - output directory
  - `"filename_pattern": r".*\.py"` - match all `.py` files
  - `"within_subsection_order": "FileNameSortKey"` - sort by filename
  - `"download_all_examples": True` - enable downloads
  - `"remove_config_comments": True` - clean output
  - `"doc_module": tuple(_PACKAGES)` - link to package docs
  - `"run_stale_examples": False` - don't execute during build
  - `"plot_gallery": False` - no image generation
  - `"backreferences_dir": "gen_modules/backrefs"` - backreferences location
  - `"first_notebook_cell": None` - no cell prepending
  - `"line_numbers": False` - cleaner code blocks

#### Scenario: Configuration comments

- **WHEN** developers read `sphinx_gallery_conf` in `docs/conf.py`
- **THEN** each option SHALL have an inline comment explaining its purpose

## REMOVED Requirements

None. This change is purely additive and does not remove existing functionality.

## Implementation Notes

### Title Extraction Pattern

Sphinx-Gallery extracts the title from the first line of the module docstring and expects the second line to be an underline of `=` characters. The title becomes the page heading and is used to generate the `sphx_glr_` reference label.

### Cross-Reference Label Format

Generated labels follow the pattern: `sphx_glr_gallery_<filename_without_extension>`. For example:
- `examples/00_quickstart.py` → `sphx_glr_gallery_00_quickstart.py`
- `examples/10_data_contracts_minimal.py` → `sphx_glr_gallery_10_data_contracts_minimal.py`

### Validation Timing

Gallery validation runs at three points:
1. **Pre-commit hook**: When `examples/*.py` files are committed
2. **Documentation build**: Before Sphinx execution in `tools/update_docs.sh`
3. **CI pipeline**: As a dedicated step before docs build

### Backward Compatibility

The changes are backward compatible with the exception of the example docstring format, which must be updated. Generated gallery pages remain at the same URLs and with the same functionality.

