## Why

Our documentation toolchain refactor removed the structured `msgspec` models and
typed loader facades that previously enforced type safety. The immediate fallout
is visible in mypy: over fifty errors now flag `Any` flows through the symbol
index builders, schema validators, MkDocs generator, and `docs/conf.py`. Those
errors are symptoms of a broader gap—JSON artifacts are no longer backed by
first-class models, optional dependency wiring leaks `Any`, and CLI error paths
raise generic `ToolExecutionError` instances without typed envelopes. The docs
quality hardening change partially addressed configuration and logging, but the
loss of typed boundaries leaves us short of the repository’s strict Ruff, Pyrefly,
and mypy gates.

## What Changes

- **Typed artifact core**
  - Reintroduce authoritative payload models under `docs/_types/artifacts.py`
    backed by `msgspec.Struct` (or equivalent) for the symbol index, delta, and
    reverse-lookup artifacts. Provide conversion helpers (`from_json`,
    `to_payload`) so writers consume and emit only typed structures.
  - Regenerate schema alignment tests and ensure all writers call the conversion
    helpers instead of assembling dictionaries by hand.
- **Griffe & MkDocs facades**
  - Define runtime-checkable protocols for the portions of Griffe we consume and
    expose typed wrapper functions in `docs/_types/griffe.py`. Replace ad-hoc
    casts in `build_symbol_index.py` and `mkdocs_gen_api.py` with a shared loader
    facade to eliminate lingering `Any` usage.
  - Establish typed logger adapters that satisfy the `WarningLogger` protocol so
    shared helpers (`resolve_git_sha`, metrics hooks) accept structured loggers
    without mypy complaints.
- **Validation & CLI ergonomics**
  - Refine `docs/_scripts/validation.py` and `docs/_scripts/validate_artifacts.py`
    so they operate on the typed payload models, introduce a dedicated
    `ArtifactValidationError`, and surface RFC 9457 Problem Details consistently.
  - Extend `tests/docs/test_doc_artifacts.py` with parametrised round-trip and
    schema failure cases using the authoritative models.
- **Sphinx / optional dependency shims**
  - Extract typed facades for Astroid, AutoAPI, and docstring helpers into
    `docs/_types/sphinx_optional.py`. Update `docs/conf.py` to consume those
    facades, removing remaining `Any` flows and redundant casts while keeping the
    existing fallback behaviour.
- **Quality gates & documentation**
  - Update contributor docs to describe the typed artifact workflow and ensure
    `make artifacts` continues to validate schemas. Run the full Ruff → Pyrefly →
    mypy → pytest → schema validation loop as part of acceptance.
- **Execution tracking**
  - Follow the fully expanded checklist in `tasks.md`, which now contains
    checkbox-tracked subtasks for every deliverable (model scaffolding, facade
    wiring, validation CLI enhancements, documentation updates) so progress can
    be monitored and reviewed systematically.

## Impact

- **Affected specs**: introduce a capability spec (`docs-artifacts-type-safety`)
  capturing the requirement that docs artifacts originate from schema-backed
  models with typed loaders and validation CLIs.
- **Affected code**: `docs/_scripts/build_symbol_index.py`,
  `docs/_scripts/symbol_delta.py`, `docs/_scripts/validate_artifacts.py`,
  `docs/_scripts/validation.py`, `docs/_scripts/mkdocs_gen_api.py`, `docs/conf.py`,
  plus new modules under `docs/_types/` and updated tests in
  `tests/docs/test_doc_artifacts.py`.
- **Rollout considerations**: land the typed models and loader facades first, then
  refactor writers and validation, followed by the `docs/conf.py` cleanup. Ensure
  no step regresses existing docs generation or schema outputs; gate merge on
  clean Ruff, Pyrefly, mypy, pytest, and schema validation results.

