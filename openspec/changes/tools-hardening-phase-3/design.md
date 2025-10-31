## Context
The initial hardening effort aligned our tooling suite with Ruff and pyrefly requirements, yet it left several structural gaps:

1. **Typed payload debt**: msgspec-backed payloads are still instantiated as untyped dictionaries, meaning mypy cannot reason about docstring builder caches, navmap documents, docs analytics, CLI envelopes, or `sitecustomize` payloads. This makes simple refactors risky and blocks downstream consumers who rely on our helpers.
2. **LibCST typing gaps**: codemods depend on incomplete stubs, forcing both production code and tests to use `Any`. Junior engineers struggle to add transformations because the IDE offers no completions and mypy produces verbose errors.
3. **Untyped testing harness**: `tests/tools/**` relies on implicit typing, so regressions can slip through when fixtures or parametrized helpers change.
4. **Packaging ambiguity**: despite runtime exports, the published `tools` package lacks a `py.typed` marker and precise stubs, so consumers treat the entire package as dynamically typed.

This design describes a phased plan to address each gap with explicit module-level responsibilities, interfaces, and validation steps aimed at an engineer new to the codebase.

## Goals
- Eliminate `Any` leakage across tooling payloads by introducing first-class msgspec structs, schema validators, and helper APIs.
- Provide a typed LibCST experience so codemods compile under strict mypy/pyrefly, backed by local stubs and contributor guidance.
- Bring the tooling test harness under strict typing without overwhelming contributors.
- Package the tooling suite with authoritative stubs and a `py.typed` marker, plus an automated smoke test.

## Non-Goals
- Changing business logic of payload producers/consumers beyond structural typing and validation.
- Replacing msgspec or LibCST with alternative libraries.
- Introducing new CLI functionality or observability surfaces (beyond retaining existing behaviour during refactors).
- Refactoring external service APIs (import-linter, Agent Catalog search) beyond the typing work.

## Detailed Plan

### 1. Typed Payload Surfaces

#### Current state
- Payloads are created as ad-hoc dictionaries (`dict[str, Any]`) in modules such as `tools/docstring_builder/cache.py`, `tools/docs/build_agent_analytics.py`, `tools/navmap/build_navmap.py`, and `sitecustomize.py`.
- The companion tests mirror this pattern, so `mypy --config-file mypy.ini tests/tools` emits hundreds of diagnostics.
- Existing JSON Schemas (e.g., `schema/tools/docstring_cache.json`) are manually maintained and not always consistent with runtime payloads.

#### Proposed structure
For each payload category:
1. **Struct definitions**: introduce or refine `msgspec.Struct` classes representing the payload (e.g., `DocstringCacheDocument`, `AnalyticsDocument`, `NavmapDocument`, `CliEnvelope`). Fields should include version fields (`version: str`) and optional metadata keys. The structs live alongside their producers (`tools/docstring_builder/models.py`, `tools/docs/analytics_models.py`, `tools/navmap/document_models.py`, `tools/_shared/cli.py`).
2. **Helper API**: provide functions like `to_document(payload: LegacyPayload | Struct) -> Struct`, `from_document(document: Struct) -> dict[str, object]`, and `validate_document(document: Struct) -> None`. These helpers should be the only surface other modules consume.
3. **Schema alignment**: update JSON Schemas under `schema/tools/**` to match the struct definitions. Use `jsonschema` to validate at runtime and add normative examples to documents. Where legacy payloads exist, the schema should include a `oneOf` or explicit version check.
4. **Migration layer**: provide compatibility loaders (`_load_legacy_payload`) that detect older payload versions and upgrade them to the typed struct. Regression tests must cover both legacy and new formats.
5. **Testing**: add pytest modules that round-trip payloads struct → JSON → struct and assert that invalid payloads raise `ValidationError` or produce Problem Details.
6. **Sitecustomize**: refactor `sitecustomize.py` to use typed config objects, ensuring main interpreter hooks remain deterministic.

#### Commands & checkpoints
- `uv run --no-sync pyrefly check tools/_shared tools/docstring_builder tools/docs tools/navmap sitecustomize.py`
- `uv run --no-sync mypy --config-file mypy.ini tools/_shared tools/docstring_builder tools/docs tools/navmap sitecustomize.py`
- `uv run --no-sync pytest tests/tools/docstring_builder tests/tools/docs tests/tools/navmap tests/tools/shared`
- Update documentation with a table summarising payload structs ↔ schema ↔ helper module.

### 2. LibCST Typing Support

#### Current state
- Codemod modules (`tools/codemods/pathlib_fix.py`, `tools/codemods/blind_except_fix.py`, `tools/docstring_builder/apply.py`) rely on partial stubs; many attributes are typed as `Any`. Tests also manipulate CST nodes without type guidance.
- Developers must manually ignore scores of mypy errors to make changes.

#### Proposed structure
1. **Stub expansion**: extend `stubs/libcst/__init__.pyi` (and supporting modules if needed) so every class/function we use is typed. Follow the upstream LibCST API reference and include docstrings where helpful.
2. **Typed helpers**: create utility functions in `tools/codemods/_utils.py` (new or existing) that wrap common patterns (e.g., building `cst.Attribute`, `cst.Call`, `cst.With`) with strongly typed signatures. Update codemods to use these helpers instead of raw constructors where it improves clarity.
3. **Contributor guide**: document in `tests/tools/README.md` how to run `mypy`/`pyrefly`/pytest for codemods, how to add a new stub entry, and how to interpret common error messages.
4. **Test improvements**: rewrite codemod tests to rely on typed fixtures (e.g., `def apply_transform(code: str, transformer: type[BaseTransformer]) -> str`). Parameterize tests to cover both positive and negative transformations.
5. **Packaging**: include `tools/py.typed` in the wheel and ensure `stubs/libcst` is packaged so downstream users benefit.

#### Commands & checkpoints
- `uv run --no-sync pyrefly check tools/codemods`
- `uv run --no-sync mypy --config-file mypy.ini tools/codemods`
- `uv run --no-sync pytest tests/tools/codemods`
- Validate packaging: `uv build` followed by unpack/install to confirm stubs load.

### 3. Typed Tooling Test Harness

#### Current state
- Tests use implicit typing. Parametrized fixtures return heterogeneous structures, making mypy produce “Function is untyped after decorator transformation” warnings.
- Junior engineers do not have a reference for how to annotate fixtures or helpers.

#### Proposed structure
1. **Documentation**: add a “Typing tests” section to `tests/tools/README.md` with examples of typed fixtures, parametrized tests, context managers, and helper factories. Include command snippets for running mypy on specific directories.
2. **Fixture factories**: introduce helper modules (e.g., `tests/tools/shared/factories.py`) that create typed payloads (`def make_error_report(...) -> ErrorReport`). Encourage tests to import from these fixtures to reduce repeated typing logic.
3. **Decorator annotations**: for `pytest.mark.parametrize`, use typed callables (`Callable[[...], None]`) and `typing.cast` only when narrowing third-party return values.
4. **Iterative cleanup**: work directory by directory (`test_docstring_builder_*`, `test_docs_*`, `test_navmap_*`, `shared`, `codemods`) cleaning up types and confirming `mypy` passes.

#### Commands & checkpoints
- `uv run --no-sync mypy --config-file mypy.ini tests/tools/docstring_builder`
- `uv run --no-sync mypy --config-file mypy.ini tests/tools/docs`
- `uv run --no-sync mypy --config-file mypy.ini tests/tools/navmap`
- `uv run --no-sync mypy --config-file mypy.ini tests/tools/codemods`
- `uv run --no-sync pytest tests/tools`

### 4. Packaging & Public Exports

#### Current state
- Runtime exports (`tools/__init__.py`) exist but stubs were previously removed and not replaced. No `py.typed` marker is shipped.
- Downstream projects cannot rely on typed Problem Details helpers or CLI utilities.

#### Proposed structure
1. **Export contract**: define a canonical list of exported helpers (Problem Details builders, CLI envelope builder, settings helpers, validation utilities). Update `tools/__init__.py` to expose them explicitly and add docstrings.
2. **Stub regeneration**: create `.pyi` files under `stubs/tools/` matching the runtime exports with precise types (no `Any`). Use the new msgspec structs and helper types.
3. **`py.typed`**: add `tools/py.typed` to the repo and ensure packaging metadata includes it.
4. **Smoke test script**: implement `scripts/test-tools-install.sh` that creates a temporary virtual environment, installs `.[tools]`, imports the key helpers, and runs a no-op CLI to confirm entry points behave.
5. **Wheel verification**: use `uv build` / `pip install` to confirm the wheel includes stubs and `py.typed`. Document the command output.

#### Commands & checkpoints
- `uv build`
- `scripts/test-tools-install.sh`
- `uv run --no-sync python - <<'PY'
from tools import build_problem_details, CliEnvelopeBuilder
print(build_problem_details)
print(CliEnvelopeBuilder)
PY`
- `wheel unpack dist/*.whl` (verify `tools/py.typed` and stubs are present).

## Data Contracts & Validation Strategy
- Maintain schemata under `schema/tools/**` with versioned `id` fields (e.g., `https://kgfoundry.dev/schema/tools/docstring-cache-1.1.0.json`).
- Write regression tests validating both current and legacy payloads against their schema.
- Provide a mapping table in the PR linking each struct → schema → tests to aid reviewers.

## Testing Strategy Summary
1. Unit tests for each payload helper (constructors, migrations, validation errors).
2. Pytest suites for codemods, including typed fixtures and negative cases.
3. Full `tests/tools` pytest run to ensure typed fixtures integrate without regressing behaviour.
4. mypy/pyrefly runs on:
   - `tools/_shared`, `tools/docstring_builder`, `tools/docs`, `tools/navmap`, `sitecustomize.py`
   - `tools/codemods`
   - `tests/tools`
5. Packaging smoke test verifying installation and imports.

## Rollout
- Work in phased PRs aligned with the task sections (payloads → LibCST → tests → packaging) to keep reviews manageable.
- After each phase, run the full quality gates: `uv run --no-sync ruff format && uv run --no-sync ruff check --fix`, `uv run --no-sync pyrefly check`, `uv run --no-sync mypy --config-file mypy.ini`, `uv run --no-sync pytest`, `make artifacts`, `openspec validate tools-hardening-phase-3 --strict`.
- Once all phases land, coordinate with release owners to publish the updated wheel and document the migration in the changelog.

## Appendix: Quick Reference for Junior Engineers
- **Shared commands**: after modifying a module, run `uv run --no-sync pyrefly check <module>` and `uv run --no-sync mypy --config-file mypy.ini <module>`.
- **Payload updates**: when adding a field, update the struct, schema, helpers, tests, and docs, then re-run mypy/pyrefly/pytest.
- **Codemods**: if mypy complains about a missing LibCST attribute, add it to the stub and run `uv run --no-sync pytest tests/tools/codemods`.
- **Tests**: follow the patterns in `tests/tools/README.md`; prefer factory helpers over inline dict literals.

