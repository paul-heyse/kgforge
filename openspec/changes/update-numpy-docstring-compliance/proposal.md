# Proposal: Enforce Strict NumPy-Style Docstring Compliance

## Why

The current documentation system uses NumPy-style docstrings but lacks strict enforcement and contains non-standard sections (particularly the `NavMap:` section injected into module docstrings). This creates several problems:

1. **Inconsistent parsing**: Sphinx's Napoleon extension accepts both Google and NumPy styles by default, allowing drift between the two formats across the codebase.

2. **Non-standard sections break validation**: The `NavMap:` section is not a recognized NumPy docstring section, causing failures when strict NumPy validation tools (numpydoc, pydoclint) are enabled.

3. **Reduced machine-readability**: AI agents and documentation parsers expect standard NumPy sections (`Parameters`, `Returns`, `Raises`, `Examples`, `Attributes`, `See Also`, `Notes`). Non-standard sections reduce reliability of automated documentation consumption.

4. **Missing validation**: Current setup lacks parameter/return parity checks (ensuring all parameters documented, no extra documented parameters, return types match).

5. **Incomplete type information**: Some docstrings use Google-style type hints (`Args:`) or inconsistent formatting (`param (type)` vs `param : type`).

6. **Missing critical sections**: Many docstrings lack `Raises`, `Attributes`, `Examples`, and `See Also` sections that are essential for both humans and agents.

## What Changes

This proposal enforces **strict NumPy-style docstrings** across the entire codebase with comprehensive validation:

### Core Changes

1. **Sphinx Configuration** (`docs/conf.py`)
   - Replace `sphinx.ext.napoleon` with `numpydoc` for stricter validation
   - Enable `numpydoc_validation` with comprehensive checks
   - Set `napoleon_google_docstring = False` if keeping Napoleon
   - Enable `nitpicky = True` for strict cross-reference validation

2. **NavMap System Refactoring** (`tools/update_navmaps.py`)
   - **BREAKING**: Remove `NavMap:` sections from module docstrings
   - Move navigation metadata entirely to `__navmap__` dictionary
   - Optionally: Render abbreviated nav list in standard `Notes` section if needed for human readers

3. **Docstring Template Updates** (`tools/doq_templates/numpy/`)
   - Ensure templates emit proper NumPy sections with correct formatting
   - Add scaffolds for `Raises`, `Examples`, `See Also`, `Attributes`
   - Use `name : type, optional` format with defaults in descriptions

4. **Validation Pipeline Hardening** (`pyproject.toml`, `.pre-commit-config.yaml`)
   - Add `pydoclint` pre-commit hook for parameter/return parity checks
   - Enable Ruff docstring rules: `D417` (all args documented), `D401` (imperative mood)
   - Update interrogate configuration for stricter coverage

5. **Generator Script Updates** (`tools/generate_docstrings.py`, `tools/auto_docstrings.py`)
   - Ensure all generated docstrings use NumPy sections only
   - Never emit Google-style `Args:` sections
   - Add `Raises` sections where exceptions are raised
   - Include runnable `Examples` using `>>>` prompts

### Documentation Improvements (Agent-Friendly)

6. **Type Annotations** - Use specific types: `Sequence[str]`, `Mapping[str, Any]`, `PathLike[str] | str`

7. **See Also Sections** - Cross-link related functions for improved agent navigation

8. **Notes Sections** - Add contract/constraint one-liners (thread-safety, complexity, input requirements)

9. **Examples Sections** - Ensure all examples are doctestable (`pytest --doctest-modules`)

10. **Attributes Documentation** - Document class attributes with types, mutability, units/shapes

## Impact

### Breaking Changes

- **Module docstrings will change**: The `NavMap:` section will be removed from all module-level docstrings. Navigation metadata will exist only in `__navmap__` dictionaries.
- **Validation failures**: Existing docstrings that don't conform to strict NumPy style will fail validation and must be fixed.
- **CI pipeline changes**: Pre-commit hooks will add `pydoclint`, which may initially fail on existing code.

### Affected Components

- **Sphinx documentation build** - Configuration changes, validation enabled
- **Pre-commit hooks** - New validator added, stricter checks enabled
- **All Python modules in `src/`** - Docstrings must conform to strict NumPy style
- **Documentation generation pipeline** - `update_navmaps.py` behavior changes significantly
- **Tools and scripts** - Templates and generators updated

### Benefits

- **Guaranteed NumPy compliance** - No drift to Google-style or custom sections
- **Better machine-readability** - Standard sections make AI agents more reliable
- **Comprehensive validation** - Catch missing parameters, undocumented returns, type mismatches
- **Improved navigation** - `See Also` sections create explicit cross-reference graph
- **Runnable examples** - Doctest integration catches documentation bugs
- **Consistent quality** - Stricter rules prevent documentation debt

### Migration Path

1. Update Sphinx configuration and validation tools (non-blocking)
2. Update NavMap system to stop injecting sections into docstrings
3. Regenerate all docstrings with new templates
4. Fix validation failures iteratively (per-module or per-package)
5. Enable strict validation in CI once baseline is clean

### Rollback Strategy

If needed, rollback can be achieved by:
1. Reverting Sphinx configuration to use Napoleon without strict validation
2. Disabling `pydoclint` pre-commit hook
3. Re-enabling `NavMap:` section injection in `update_navmaps.py`

Standard sections (`Parameters`, `Returns`, etc.) remain compatible with both strict and lenient parsing, so no docstring changes need to be reverted.

