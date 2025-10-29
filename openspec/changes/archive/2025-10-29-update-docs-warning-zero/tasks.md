## 1. Baseline & Validation
- [ ] 1.1 Capture current warning inventory (`tools/update_docs.sh` with `-W` disabled, save `sphinx-warn.log`).
- [ ] 1.2 Add regression fixture: archive the warning log in `openspec/changes/update-docs-warning-zero/` for reference.
- [ ] 1.3 Confirm `tools/update_docs.sh` writes `sphinx-warn.log` every run (already patched); add a quick check in the script that fails if the log is missing.

## 2. Sphinx Configuration Hygiene
- [ ] 2.1 Remove `"numpydoc_validation"` from `docs/conf.py` extensions.
- [ ] 2.2 Replace runtime validation with a CLI step in `tools/update_docs.sh` (e.g., `uv run numpydoc validate …`).
- [ ] 2.3 Normalise `numpydoc_validation_exclude` to a `set` literal to satisfy Sphinx type expectations.
- [ ] 2.4 Add unit test (or doctest) that imports `docs.conf` and asserts `"numpydoc_validation"` is absent.

## 3. Intersphinx Repair
- [ ] 3.1 Update Typer mapping to `https://typer.tiangolo.com/latest/objects.inv`.
- [ ] 3.2 Update DuckDB mapping to `https://duckdb.org/docs/api/python_api/objects.inv`.
- [ ] 3.3 Add a helper in `docs/conf.py` that pings every intersphinx URL during config; fail fast (raise `ConfigError`) if inventories return 404/connection errors.
- [ ] 3.4 Extend intersphinx smoke test in CI (simple `requests.head` or `urllib`) so we catch future breakage before docs builds.

## 4. Auto Docstring Coverage
- [ ] 4.1 Catalogue every ES01 warning in the latest log and map it to the generating symbol.
- [ ] 4.2 Extend `_is_magic`, `_is_property`, `_is_pydantic_field`, and related helpers so `__cause__`, `__context__`, `_NoopMetric`, `ModuleMeta`, `_load_dense_from_parquet`, `SupportsHttp`, and `SupportsResponse` trigger specialised summaries.
- [ ] 4.3 Add stock extended summaries (single source of truth dictionaries) for each new helper/protocol.
- [ ] 4.4 Ensure fallbacks produce at least two sentences and end with a period.
- [ ] 4.5 Add targeted unit tests under `tests/unit/test_auto_docstrings_extended_summaries.py` covering each new symbol category.
- [ ] 4.6 Regenerate docstrings (`make docstrings`) and confirm no ES01 warnings remain.

## 5. AutoAPI Deduplication
- [ ] 5.1 Identify duplicate targets (currently `NDArray[np.float32]`).
- [ ] 5.2 Decide canonical home (likely `vectorstore_faiss.gpu`).
- [ ] 5.3 In non-canonical pages, add `:no-index:` or swap to cross references so Sphinx only indexes one target.
- [ ] 5.4 Run `make html` and verify duplicate-target warnings disappear.

## 6. API Navigation Integration
- [ ] 6.1 Create/restore `docs/api/index.md` with a hidden glob toctree (`.. toctree:: :maxdepth: 1 :hidden: api/*/index`).
- [ ] 6.2 Ensure AutoAPI output lands under `docs/api/` (adjust `autoapi_template_dir` or copy step if needed).
- [ ] 6.3 Include the API landing page from `docs/index.md` so navigation picks it up (possibly hidden, but present).
- [ ] 6.4 Add regression test: parse built `_build/json/index.fjson` to assert every `docs/api/**` page appears in a toctree.

## 7. Schema Catalogue Glob
- [ ] 7.1 Inspect schema export output path (currently `docs/_build/schemas`?).
- [ ] 7.2 Update `docs/reference/schemas/index.md` so `:glob:` points at the actual JSON location.
- [ ] 7.3 If necessary, move generated files into the referenced directory (update `tools/docs/export_schemas.py`).
- [ ] 7.4 Build docs to confirm the glob matches at least one document.

## 8. Gallery Import Fix
- [ ] 8.1 Audit gallery scripts (`examples/*.py`) for imports like `import kgfoundry`.
- [ ] 8.2 Replace with real packages (`from kgfoundry_common import …`) or add an importable shim inside `examples/` that exposes the public API.
- [ ] 8.3 Run `tools/update_docs.sh` and confirm the `py:mod` warning disappears.

## 9. Hard-Gate Zero Warnings
- [ ] 9.1 Update `tools/update_docs.sh` to pass `-W` to `sphinx-build` and fail if warnings are emitted.
- [ ] 9.2 Add a safety step that parses `sphinx-warn.log` post-build and asserts it is empty (or contains only allowed informational lines).
- [ ] 9.3 Document the zero-warning policy in `README-AUTOMATED-DOCUMENTATION.md` and ensure contributors know how to run the checks locally.
- [ ] 9.4 Capture final warning log (should be empty) and attach to this OpenSpec change as evidence.

