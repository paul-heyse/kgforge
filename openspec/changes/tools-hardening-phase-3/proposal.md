## Why
The first wave of hardening work brought our Ruff and pyrefly suites back to green, yet the type system still leaks `Any` across critical tooling paths:

- msgspec-backed payloads (docstring builder cache, docs analytics, navmap, CLI envelopes, sitecustomize) instantiate untyped dictionaries, so `mypy --config-file mypy.ini tests/tools` reports hundreds of errors.
- LibCST codemods depend on incomplete or missing stubs, making even simple transformer edits opaque to junior contributors and forcing reviewers to ignore warnings.
- Tooling tests rely on untyped fixtures and decorators, meaning regressions can slip past reviewers because mypy does not cover the test harness.
- The published `tools` package does not carry a `py.typed` marker or precise stubs/exposed exports, so downstream automation cannot rely on our helpers without treating them as `Any`.

Without addressing these gaps, we cannot meet the repository’s “strict types everywhere” bar. The developer experience for junior engineers is especially painful: the IDE offers no completions, `mypy` produces overwhelming error dumps, and it is unclear which files must be touched to add a new payload or codemod. This follow-on change provides the missing structure and documentation so that even a new team member can confidently implement or extend the tooling stack.

## What Changes
- **MODIFIED**: Typed payload requirement to mandate msgspec struct definitions, schema-backed validators, and mypy-safe helpers for docstring caches, docs analytics, navmap documents, CLI envelopes, and `sitecustomize` scaffolding. The change explicitly maps each runtime payload to its schema, helper module, and regression tests so contributors have a checklist to follow.
- **ADDED**: LibCST typing requirement deltas covering a local `py.typed` stub package, extended `stubs/libcst/**`, and codemod helper refactors. Both production code and pytest fixtures must compile under strict typing, and we document a recipe for adding new transformers.
- **ADDED**: New requirement ensuring tooling tests declare types (parametrized fixtures, decorators, and helpers) and run cleanly under `mypy --config-file mypy.ini tests/tools` without lossy `Any` casts. We provide fixture templates, typing guidelines, and verification commands tailored to junior engineers.
- **MODIFIED**: Packaged tools requirement so the published distribution ships `py.typed`, public re-exports (`tools.*`), and precise stubs for Problem Details / CLI helpers that match runtime behavior. The proposal includes an installation smoke-test script and wheel verification checklist.

## Impact
- **Affected specs**: `tools-suite`.
- **Affected code**:
  - Typed payload modules: `tools/_shared/cli.py`, `tools/_shared/problem_details.py`, `tools/docstring_builder/{cache,config,harvest,ir,models,apply}.py`, `tools/docs/{build_agent_analytics,build_agent_catalog,render_agent_portal}.py`, `tools/navmap/{document_models,build_navmap,repair_navmaps}.py`, `sitecustomize.py`.
  - Stub packages & packaging: `stubs/msgspec/**`, `stubs/libcst/**`, new `tools/py.typed`, and refined `setup.cfg` / extras to surface exports.
  - Tests: `tests/tools/**` (docstring builder, docs analytics, navmap, sitecustomize, build_agent_catalog), plus shared fixtures and parametrizations.
- **Rollout**: Iterate per focus area (typed payloads → LibCST stubs → typed tests → packaging/stubs). Each step runs Ruff → pyrefly → mypy (`tests/tools` and relevant modules) → pytest targeted suites. Final gate requires full `mypy tests/tools` success and updated packaging metadata validated via `pip install .[tools]` smoke test.

## Implementation Plan (Step-by-step)

### 1. Typed payload surfaces
1. **Inventory payloads**: enumerate every JSON/CLI payload in scope (docstring cache, docfacts/IR models, analytics, navmap, CLI envelopes, sitecustomize). Capture the owning module, existing schema (if any), and runtime producer/consumer in a table inside the PR description.
2. **Define msgspec structs**: for each payload, create or refine a `msgspec.Struct` (immutable unless mutation is required) with precise field types, default values, and version fields. Place shared structs in `tools/_shared/cli.py` or `tools/navmap/document_models.py` as appropriate.
3. **Schema alignment**: update or add JSON Schemas under `schema/tools/**` so they mirror the struct definitions. Add normative examples illustrating both happy-path and error payloads.
4. **Conversion helpers**: replace ad-hoc `dict` construction with helper functions (`from_payload`, `to_payload`, `validate_*`) that perform schema validation and return typed structs. These helpers must be unit-tested with both valid and invalid inputs.
5. **Remove `Any` escapes**: audit each payload module and tests to eliminate `cast(Any, …)` or `dict[str, Any]` scaffolding. Use `mypy --config-file mypy.ini tools/<module>.py` to ensure the module compiles in isolation before moving on.
6. **Regression tests**: add or expand pytest coverage that round-trips payloads (model → JSON → model) and asserts the schema validator rejects malformed inputs.

### 2. LibCST typing & codemod ergonomics
1. **Stub coverage**: extend `stubs/libcst/__init__.pyi` (and related modules) to include every node, visitor, and helper used by our codemods. Organise the stub into logical sections (expressions, statements, transformers) with references to the upstream API docs.
2. **Publish `py.typed`**: add `tools/py.typed` and update packaging metadata so the stub package is shipped with the wheel.
3. **Codemod helpers**: refactor codemod modules (`tools/codemods/pathlib_fix.py`, `blind_except_fix.py`, etc.) to use typed helper functions when constructing CST nodes. Replace dynamic attribute access with explicit constructors (`cst.Call(...)`, `cst.Attribute(...)`).
4. **Test templates**: create reusable pytest fixtures for parsing source, applying transformers, and asserting results. Annotate fixtures with precise types (e.g., `Callable[[str], Module]`).
5. **Validation**: run `mypy --config-file mypy.ini tools/codemods` and `pyrefly check tools/codemods` to confirm stubs cover every use. Document the command sequence in the PR so junior contributors can reproduce it.

### 3. Typed tests harness
1. **Fixture guidelines**: draft a short developer guide (added to `tests/tools/README.md`) showing how to annotate fixtures, `pytest.mark.parametrize`, and context-manager helpers.
2. **Systematic cleanup**: iterate through `tests/tools/**` and:
   - add return types to test functions when decorators transform call signatures,
   - replace inline literals with typed factory helpers (e.g., `make_error_report()`),
   - use `typing.cast` only when narrowing third-party returns.
3. **Command verification**: after each directory is cleaned, run `mypy --config-file mypy.ini tests/tools/<subdir>` and capture the output in the PR so reviewers see progress.
4. **CI guard**: add a dedicated job (or document existing one) ensuring `mypy --config-file mypy.ini tests/tools` runs on every PR touching tooling.

### 4. Packaging & exported APIs
1. **Re-export audit**: list every helper we expect consumers to rely on (Problem Details builders, CLI envelopes, validation utilities). Ensure `tools/__init__.py` and `tools/docstring_builder/__init__.py` expose them with accurate signatures.
2. **Stub sync**: add precise `.pyi` files under `stubs/tools/**` that mirror the runtime exports. The stubs should import concrete types (e.g., `from tools._shared.problem_details import ProblemDetailsDict`) instead of aliasing to `Any`.
3. **Install smoke test**: script a smoke test (documented in `tasks.md`) that creates a temp venv, runs `pip install .[tools]`, imports the key modules, and executes a no-op CLI command. Capture the command output in the PR checklist.
4. **Wheel validation**: run `uv build`, inspect the generated wheel to ensure `py.typed` and stubs are included, and attach the `wheel unpack` tree in the PR description for transparency.

## Non-Goals
- Reworking the public API surface of docstring builder or navmap beyond typing and validation. Behavioural changes require a separate proposal.
- Introducing new CLI features or observability metrics; we only ensure existing metrics/spans remain after typing.
- Refactoring third-party dependencies (e.g., replacing msgspec) or rewriting codemods in another library.

## Risks & Mitigations
- **Risk**: msgspec struct changes break backward compatibility with stored caches. **Mitigation**: retain loaders that recognise previous versions and migrate them during validation, with regression tests demonstrating compatibility.
- **Risk**: Expanded stubs diverge from upstream LibCST releases. **Mitigation**: gate stubs behind a clearly documented version (e.g., “tested against LibCST 1.3.x”) and add a lightweight CI check importing the real library when available.
- **Risk**: Junior contributors may still struggle with the command flow. **Mitigation**: include a step-by-step “developer checklist” in the PR template and cross-link it from `tasks.md`.

## Acceptance Summary
Successful completion means:
- `uv run --no-sync pyrefly check tools` and `uv run --no-sync mypy --config-file mypy.ini tests/tools` both pass without ignoring diagnostics.
- Payload modules expose strongly typed helpers validated by schemata, with unit tests covering success/failure paths.
- LibCST codemods import without stub errors, and their pytest suites document how to add new transformations.
- The published package delivers `py.typed`, accurate stubs, and a smoke-tested installation story.

