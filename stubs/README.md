# Stub packages

This directory contains local type stub packages used by the docstring tooling. The
stubs intentionally mirror only the attributes we exercise in tests and runtime code
so that mypy can check those surfaces without depending on upstream packages
shipping official typings.

## Contributing guidelines

* **Prefer minimal updates.** Keep the signatures lean and focused on the symbols
  imported inside ``src/`` and ``tools/docstring_builder``.
* **Document new attributes.** Add a short comment next to newly stubbed members
  describing the source location in the runtime library, so future contributors can
  confirm behaviour quickly.
* **Regenerate drift reports.** Run ``make stubs-check`` (or ``uv run
  python tools/stubs/drift_check.py``) after editing stubs to ensure runtime modules
  expose the same attributes.
* **Add tests when behaviour changes.** If a new attribute unlocks functionality in
  our code, add or update tests demonstrating the new behaviour. This helps catch
  upstream drift early.

Following these guidelines keeps the stub packages maintainable while providing the
type coverage needed for strict mypy runs.
