## 1. Inventory & Planning
- [ ] 1.1 Run `tools/update_docs.sh` (or review the latest log) and list every `ref.*` warning
      with the file path and missing target. Copy this list into a scratch document so the
      work is visible and can be ticked off.
- [ ] 1.2 Categorise the warnings into: (a) third-party annotations (NumPy, PyArrow, DuckDB,
      Typer, FastAPI), (b) our own aliases lacking anchors, and (c) duplicate targets caused
      by multi-module exports.
- [ ] 1.3 Decide which annotations should resolve via `QUALIFIED_NAME_OVERRIDES`, which
      should use intersphinx (`docs/conf.py`), and which might need targeted entries in
      `nitpick_ignore` if resolution is impossible.

## 2. Implementation
- [ ] 2.1 Update `QUALIFIED_NAME_OVERRIDES` in `tools/auto_docstrings.py` so our generator
      emits fully qualified names for the recurring third-party types (e.g., map
      ``NDArray``/``numpy.typing.NDArray`` to a resolvable target).
- [ ] 2.2 Adjust `docs/conf.py`:
      - Add missing intersphinx mappings for packages that expose those types (Typer,
        FastAPI, DuckDB, PyArrow etc.).
      - Prune/refresh `nitpick_ignore` so it contains only the truly unresolvable cases.
- [ ] 2.3 Ensure our internal aliases have anchors and single exports:
      - Add `[nav:anchor]` markers or module-level doc references for `VecArray`, `StrArray`,
        `Concept`, etc.
      - Re-export shared exceptions so only one module defines the class; others import it to
        avoid duplicate-reference warnings.
- [ ] 2.4 (Optional but helpful) Add targeted tests or a CI helper that runs
      `sphinx-build -b html docs docs/_build/html -w sphinx-warn.log` and asserts the log has
      zero `ref.*` warnings, to catch regression.

## 3. Validation
- [ ] 3.1 Run `make docstrings` (ensuring doc generation still succeeds) and
      `tools/update_docs.sh` to confirm the previous `ref.*` warnings are gone.
- [ ] 3.2 Open one or two affected AutoAPI pages (such as the FAISS adapter and vectorstore
      modules) in the generated HTML and manually verify the previously broken links now work.
- [ ] 3.3 If any warnings remain that we intentionally cannot resolve, document them in the
      change summary and add explicit `nitpick_ignore` entries so the build is clean.

