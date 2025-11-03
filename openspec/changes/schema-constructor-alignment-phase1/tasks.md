## 1. Implementation
- [x] 1.1 Inventory constructor usage
  - [x] Search for `model_construct`, `model_validate`, and direct model instantiations in
        `docs/_types`, `docs/_scripts`, `tools/navmap`, and `tools/docstring_builder`.
  - [x] Capture a checklist of affected functions/modules with notes on required casing
        corrections and observability touchpoints.
- [x] 1.2 Build alignment utilities
  - [x] Add a typed helper (e.g., `align_schema_fields`) that normalizes legacy casing to
        canonical schema keys, preserves order, and returns structured results with
        `TypeVar`-backed generics.
  - [x] Integrate Problem Details emission for unknown/invalid keys and ensure helpers log
        with `operation`, `status`, and `schema_id` context.
- [x] 1.3 Update constructors across pipelines
  - [x] Replace legacy keyword usage with canonical casing in docs artifacts, navmap models,
        and docstring-builder caches.
  - [x] Ensure each entry point invokes the alignment helper before constructing models and
        that legacy code paths are wrapped with explicit deprecation warnings.
- [x] 1.4 Migration compatibility
  - [x] Add counters/metrics recording the number of migrations performed per artifact type.
  - [x] Verify logs and Problem Details outputs reference remediation guidance (schema link,
        expected casing).
- [x] 1.5 Regression coverage instrumentation
  - [x] Extend pytest suites with table-driven cases for canonical payloads, legacy casing,
        missing keys, and extra keys; assert round-trip equality and checksum preservation.
  - [x] Update doctest/xdoctest snippets to cover canonical construction and migration helper
        usage, mapping to AGENTS observability standards.
- [x] 1.6 Documentation updates
  - [x] Revise contributor documentation to outline constructor requirements, migration
        helper usage, and observability expectations.
  - [x] Embed runnable examples demonstrating schema-aligned constructors and Problem
        Details handling.

## 2. Testing & Quality Gates
- [x] 2.1 `uv run pytest -q tests/docs tests/tools/docstring_builder tests/navmap`
- [x] 2.2 `uv run pytest --doctest-modules docs/_types docs/_scripts tools/navmap tools/docstring_builder`
- [x] 2.3 `uv run ruff format && uv run ruff check --fix`
- [x] 2.4 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.5 `uv run pyrefly check`
- [x] 2.6 `uv run mypy --config-file mypy.ini`
- [x] 2.7 `make artifacts && git diff --exit-code`
- [x] 2.8 Update schema validation CLI (`python docs/_scripts/validate_artifacts.py`) to
        confirm aligned constructors

