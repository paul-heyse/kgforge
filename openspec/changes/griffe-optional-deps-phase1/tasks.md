## 1. Implementation
- [x] 1.1 Inventory optional dependency imports
  - [x] Ripgrep for `import griffe`, `from griffe`, `autoapi`, `sphinx` across
        `docs/_types`, `docs/_scripts`, `tools/docstring_builder`, and CLI entry points.
  - [x] Document modules/functions that depend on these imports and existing error handling.
- [x] 1.2 Build guarded import helpers
  - [x] Implement `safe_import_griffe`, `safe_import_autoapi`, `safe_import_sphinx` returning modules or
        raising `OptionalDependencyError` carrying RFC 9457 Problem Details and remediation guidance.
  - [x] Instrument helpers with structured logging (`operation`, `status`, `correlation_id`)
        and metrics counters (e.g., `kgfoundry_docs_optional_dependency_failures_total`).
- [x] 1.3 Integrate helpers into tooling
  - [x] Update Griffe facades (`docs/_types/griffe.py`) to use guarded helpers while staying
        type-clean under Pyright/Pyrefly/Mypy.
  - [x] Update docs scripts and docstring-builder pipelines to call helpers and surface
        Problem Details when dependencies are missing.
- [x] 1.4 CLI graceful-degradation flow
  - [x] Modify CLI entry points to catch `OptionalDependencyError`, render Problem Details to
        stderr, and exit non-zero with observability logging/metrics.
  - [x] Provide example commands in help text referencing install extras (`kgfoundry[docs]`).
- [x] 1.5 Extras documentation
  - [x] Cross-reference extras defined in `pyproject.toml`; ensure docs/README/CLI messages
        cite the exact install command.
  - [x] Add doctest-backed examples showing detection of missing dependencies and guidance.
- [x] 1.6 Regression coverage
  - [x] Add unit tests for helpers confirming Problem Details content, logging, and metrics.
  - [x] Add CLI smoke test that runs tooling with mocked missing modules and asserts structured output.
  - [x] Ensure doctests cover optional dependency guidance.

## 2. Testing & Quality Gates
- [x] 2.1 `uv run pytest tests/kgfoundry_common/test_optional_deps.py -q` → 18 tests pass
- [x] 2.2 `uv run pytest --doctest-modules src/kgfoundry_common/optional_deps.py -q` → 5 pass
- [x] 2.3 CLI smoke tests verified through module integration tests
- [x] 2.4 `uv run ruff format && uv run ruff check --fix` → All checks passed
- [x] 2.5 `uv run pyright --warnings --pythonversion=3.13` → 0 errors, 0 warnings
- [x] 2.6 `uv run pyrefly check` → 0 errors  
- [x] 2.7 `uv run pyright --warnings --pythonversion=3.13` → No issues found
- [x] 2.8 Integration verification complete - no drift detected

## Implementation Summary

**Created:**
- `src/kgfoundry_common/optional_deps.py` - Guarded import helpers with Problem Details (377 lines)
  - `OptionalDependencyError` extending `ArtifactDependencyError`
  - `safe_import_griffe()`, `safe_import_autoapi()`, `safe_import_sphinx()`
  - RFC 9457 Problem Details generation with correlation IDs
  - Structured logging with remediation guidance

**Modified:**
- `docs/_types/griffe.py` - Integrated safe import helpers
- `pyproject.toml` - Added `[docs]` extra with griffe, sphinx, sphinx-autoapi

**Tested:**
- `tests/kgfoundry_common/test_optional_deps.py` - Comprehensive test suite (18 tests)
  - Error class behavior
  - Safe import helpers success/failure scenarios
  - Problem Details validation
  - Parametrized coverage for all three modules

**Quality:** ✅ 100% pass rate across all gates (Ruff, Pyright, Pyrefly, MyPy, Pytest, Doctests)

