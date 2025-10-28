## Why
Extended-summary warnings dominate the Sphinx build because our docstring generator does
not emit narrative text for either (a) Python magic methods (``__eq__``, ``__iter__`` …)
or (b) the helper attributes/functions that Pydantic adds to each model. NumPy-style
docstrings require an extended summary section, so we currently see hundreds of ES01
warnings. This noise hides real regressions, makes the generated API docs hard to skim, and
prevents us from turning on the “warnings as errors” gate for documentation builds.

## What Changes
- Expand `tools/auto_docstrings.py::extended_summary()` with a table of stock summaries for
  the magic methods we keep documented (comparison operators, object protocol hooks,
  container methods, etc.).
- Extend the generator so it recognises the various ``__pydantic_*`` helpers and public
  ``model_*`` accessors that Pydantic attaches to models, providing consistent boilerplate
  summaries (or skipping members that should remain undocumented).
- Update the filtering logic so that any newly skipped members are excluded across the
  pipeline (Docstring generator, AutoAPI, tests).
- Add regression coverage that renders docstrings for sample classes/functions and asserts
  the extended summary is present. This ensures future refactors do not re-introduce ES01
  warnings.

## Impact
- Affected specs: none (tooling-only change; behaviour of generated documentation only).
- Affected code: `tools/auto_docstrings.py`, potential updates to `tests/docs/` (or new
  targeted tests under `tests/tools/`).
- Expected outcome: running `make docstrings` and `tools/update_docs.sh` produces zero
  numpydoc ES01 warnings for the covered magic/Pydantic cases, making it feasible to enable
  strict Sphinx builds in CI.

