## Why

Our documentation pipeline still reports 122 warnings after a full run of `tools/update_docs.sh`.  The warning log points to systemic issues: an invalid Sphinx extension (`numpydoc_validation`) that never supported programmatic setup, stale intersphinx inventory links for Typer and DuckDB, docstrings generated without extended summaries for magic/data-model helpers, duplicate AutoAPI targets for shared type aliases, dozens of API pages that never land in a toctree, an empty schema toctree glob, and gallery examples that reference a non-existent top-level `kgfoundry` package.  We need to capture a cohesive plan so future documentation builds run with warnings-as-errors and remain green.

## What Changes

- Remove the `numpydoc_validation` Sphinx extension, replace it with CLI validation, and normalise `numpydoc_validation_exclude` to a `set`.
- Update intersphinx mappings to valid inventories (Typer’s current `latest/objects.inv`, DuckDB’s `python_api/objects.inv`) and add a validation check to fail fast on future 404s.
- Extend `tools/auto_docstrings.py` so every generated extended summary covers: exception helpers (`__cause__`, `__context__`), protocol shims (`SupportsHttp`, `SupportsResponse`), module metadata containers (`ModuleMeta`, `_NoopMetric`), parquet helpers, and any other data-model glue still producing ES01 warnings.
- De-duplicate AutoAPI targets for shared type aliases such as `NDArray[np.float32]` by canonicalising the alias or marking secondary references with `:no-index:`.
- Publish the generated API documentation (`docs/api/**/index.md`) through a stable hidden toctree so Sphinx recognises every page.
- Fix the schema catalogue glob by pointing it at the generated JSON directory or emitting the files where the glob expects them.
- Adjust gallery quickstart imports so `py:mod` references resolve, or provide a shim module just for gallery linking.
- Bake the above into CI: run `make html` with `-W` and treat missing sphinx-warn.log output as a failure.

## Impact

- Affects documentation tooling (`docs/conf.py`, `tools/update_docs.sh`, `tools/auto_docstrings.py`, generated AutoAPI artefacts, schema export scripts, gallery examples).
- Requires coordination with existing OpenSpec changes `update-docstring-extended-summaries`, `fix-extended-summary-warnings`, and `fix-unresolved-cross-references` to avoid duplicated scope; this proposal ties together the remaining warning categories.
- Produces repeatable zero-warning builds so warnings-as-errors can be enforced in CI without intermittent breakages.

