## 1. Implementation
- [x] 1.1 Establish artifact exception taxonomy
  - [x] Add `ArtifactModelError`, `ArtifactValidationError`, and
        `ArtifactSerializationError` under `kgfoundry_common.errors`
  - [x] Provide Problem Details builder with canonical example stored at
        `schema/examples/problem_details/docs-artifact-validation.json`
- [x] 1.2 Refactor artifact models
  - [x] Convert `docs/_types/artifacts.py` models to frozen Pydantic V2 classes
        with PEP 695 generics and schema metadata properties
  - [x] Replace long string literals with structured errors referencing the new
        taxonomy
  - [x] Remove `Any` by introducing typed helper functions and richer type
        aliases
- [x] 1.3 Update loaders and stubs
  - [x] Align `docs/_types/griffe.py` facades with the refined model contracts
  - [x] Rewrite `stubs/griffe/__init__.pyi` and `stubs/griffe/loader/__init__.pyi`
        to eliminate `Any` splats and match runtime signatures
- [x] 1.4 Harden docs build scripts
  - [x] Update `docs/_scripts/build_symbol_index.py` and related helpers to use
        schema-aware builders and emit structured logs
  - [x] Ensure optional dependency fallbacks raise/tolerate `ArtifactModelError`
        with preserved causes
- [x] 1.5 Extend regression coverage
  - [x] Add table-driven pytest suites for artifact round-trips, delta
        computation, and validation failures (`tests/docs/test_artifact_models.py`)
  - [x] Cover Problem Details payload assertions and optional dependency
        fallbacks (`tests/docs/test_symbol_delta.py`)
  - [x] Introduce doctest snippets in public docstrings demonstrating model
        construction and error handling
- [x] 1.6 Refresh schemas and documentation
  - [x] Regenerate/update JSON Schemas under `schema/docs/**` if fields change
  - [x] Document new workflows in `docs/contributing/docs-pipeline.md`
  - [x] Run `make artifacts` and inspect outputs for regression

## 2. Testing
- [x] 2.1 `uv run pytest -q tests/docs tests/tools/docstring_builder`
- [x] 2.2 `uv run pytest --doctest-modules docs/_types docs/_scripts kgfoundry_common`
- [x] 2.3 Validate schemas via `python docs/_scripts/validate_artifacts.py`
- [x] 2.4 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.5 `uv run pyrefly check`
- [x] 2.6 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.7 `uv run ruff format && uv run ruff check --fix`

## 3. Docs & Artifacts
- [ ] 3.1 `make artifacts && git diff --exit-code`
- [ ] 3.2 Update release notes / change log entries if schema deltas ship
- [ ] 3.3 Ensure Problem Details example is linked from docs reference pages

## 4. Rollout
- [ ] 4.1 Coordinate with docs owners for downtime-free deploy window
- [ ] 4.2 Prepare migration notes for downstream consumers (if any)
- [ ] 4.3 Archive change via `openspec archive artifact-models-hardening-phase1 --yes`

