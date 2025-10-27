# Migration: Strict NumPy Docstring Compliance

This change removes legacy `NavMap:` blocks from module docstrings and enables
full NumPy-style validation across the documentation pipeline. Use this guide to
understand the impact and to verify your environment.

## Summary of Changes

- Sphinx now loads the `numpydoc` and `numpydoc_validation` extensions and treats
  `GL01`, `SS01`, `ES01`, `RT01`, and `PR01` violations as build failures (`nitpicky = True`).
- Custom doq templates generate NumPy-compliant docstrings (including `Raises` and
  `Examples` sections) for functions, classes, and modules.
- `tools/auto_docstrings.py` rewrites missing or placeholder docstrings to include
  `Parameters`, `Returns`, `Raises`, `Examples`, `See Also`, and `Notes` sections.
- `tools/update_navmaps.py` is now a validator that aborts if any docstring still
  contains a `NavMap:` section.
- Pre-commit gains a `pydoclint --style numpy src` step that enforces parameter/return parity.
- `.numpydoc` configures repository-wide validation checks during Sphinx builds.

### Performance

Docstring regeneration now triggers additional validation (numpydoc + pydoclint).
On a clean checkout this adds roughly a few seconds to `make docstrings`; monitor
`tools/update_docs.sh` runtimes after large refactors and adjust tooling caches if needed.

## What Contributors Need to Do

1. Regenerate docstrings after modifying public APIs:
   ```bash
   make docstrings
   ```
2. Run the documentation pipeline to confirm there are no warnings:
   ```bash
   tools/update_docs.sh
   ```
3. Execute the doctest suite to ensure `Examples` sections remain runnable:
   ```bash
   pytest --doctest-modules src
   ```
4. If pre-commit is not installed, install it and run all hooks:
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

## Rollback Strategy

To revert to the previous behaviour:

1. Remove the `numpydoc` extensions from `docs/conf.py` and restore the Napoleon defaults.
2. Delete `.numpydoc` and revert the docstring templates in `tools/doq_templates/numpy/`.
3. Restore `tools/update_navmaps.py` to inject `NavMap:` sections (and rerun the script).
4. Drop the `pydoclint` dependency and hook from `pyproject.toml` and `.pre-commit-config.yaml`.

Rolling back should be accompanied by regenerating docstrings (`make docstrings`) so
that older Google-style docstrings can be reintroduced if necessary.
