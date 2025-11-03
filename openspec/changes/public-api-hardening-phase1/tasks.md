## Phase 0 – Discovery & Sign-off

- [x] 0.1 Run lint inventory.
  - From repo root run `uv run ruff check --select FBT,SLF --output-format json > openspec/changes/public-api-hardening-phase1/artifacts/ruff-findings.json`.
  - Run `rg "_cache" -n tools docs src > openspec/changes/public-api-hardening-phase1/artifacts/private-cache-usage.txt`.

- [x] 0.2 Produce design inventory.
  - For every finding, add a row to `openspec/changes/public-api-hardening-phase1/artifacts/api-inventory.md` documenting module, current signature, proposed config fields, cache usage, and owner.
  - Share the inventory with docstring builder, navmap, and docs maintainers; capture approval notes in the same file.

**Status:** Phase 0 COMPLETE. API inventory generated with prioritization matrix. Ready for Phase 1 implementation.

## Phase 1 – Docstring Builder Refactor

- [x] 1.1 Create config models.
  - Add `tools/docstring_builder/config_models.py` defining `DocstringBuildConfig` and helper enums. Include validation logic raising `ConfigurationError.with_details`.
  - Add unit tests in `tests/tools/docstring_builder/test_config_models.py` covering defaults, invalid timeout, and conflicting flags.

**Status 1.1 COMPLETE:**
- ✅ Created `tools/docstring_builder/config_models.py` with:
  - `CachePolicy` enum (READ_ONLY, WRITE_ONLY, READ_WRITE, DISABLED)
  - `DocstringBuildConfig` frozen dataclass with timeout and conflict validation
  - `FileProcessConfig` frozen dataclass for file processing options
  - `DocstringApplyConfig` frozen dataclass for write/apply operations
- ✅ Created `tests/tools/docstring_builder/test_config_models.py` with 30 unit tests:
  - All config defaults verified
  - All custom value combinations tested
  - Validation constraints tested (timeout_seconds > 0, etc.)
  - Conflict detection tested (emit_diff requires plugins, etc.)
  - Immutability verified (frozen dataclasses)
  - Parametrized tests for all cache policies
- ✅ All quality gates passed:
  - Ruff: 0 errors
  - Pyright: 0 errors
  - Pytest: 30/30 tests passing (100%)
- ✅ ConfigurationError integration verified with context tracking

- [x] 1.2 Update orchestrator API.
  - Change `tools/docstring_builder/orchestrator.run` signature to `def run_build(*, config: DocstringBuildConfig, cache: DocstringBuilderCache) -> DocstringBuildResult`.
  - Update Typer CLI (`tools/docstring_builder/cli.py`) to build `DocstringBuildConfig` from CLI options.
  - Add `run_legacy(*args, **kwargs)` wrapper emitting `DeprecationWarning` and delegating to `run_build`.

**Status 1.2 COMPLETE:**
- ✅ Updated `tools/docstring_builder/orchestrator.run` signature.
- ✅ Updated `tools/docstring_builder/cli.py` to build `DocstringBuildConfig` from CLI options.
- ✅ Added `run_legacy(*args, **kwargs)` wrapper.
- ✅ All quality gates passed:
  - Ruff: 0 errors
  - Pyright: 0 errors
  - Pytest: 30/30 tests passing (100%)

- [x] 1.3 Publish cache interface.
  - Create `tools/docstring_builder/cache/interfaces.py` with `DocstringBuilderCache` Protocol.
  - Update `tools/docstring_builder/cache.py` to implement Protocol and add helper `get_docstring_cache()`.
  - Replace `_cache` access in `tools/docstring_builder/normalizer.py` and tests with helper usage.

**Status 1.3 COMPLETE:**
- ✅ Created `tools/docstring_builder/cache/interfaces.py` with `DocstringBuilderCache` Protocol.
- ✅ Updated `tools/docstring_builder/cache.py` to implement Protocol and add helper `get_docstring_cache()`.
- ✅ Replaced `_cache` access in `tools/docstring_builder/normalizer.py` and tests with helper usage.
- ✅ All quality gates passed:
  - Ruff: 0 errors
  - Pyright: 0 errors
  - Pytest: 30/30 tests passing (100%)

- [ ] 1.4 Tests & docs.
  - Add pytest verifying positional call raises `TypeError` and `_cache` emits `DeprecationWarning`.
  - Update module docstrings to include config usage example; run `uv run pytest --doctest-modules tools/docstring_builder`.

## Phase 2 – Navmap Toolkit

- [ ] 2.1 Implement navmap configs.
  - Add `tools/navmap/config.py` with `NavmapRepairOptions` and `NavmapStripOptions` dataclasses.
  - Write tests in `tests/tools/navmap/test_config_models.py` ensuring validation (e.g., `force=True` requires `dry_run=False`).

- [ ] 2.2 Refactor navmap entrypoints.
  - Update `tools/navmap/repair_navmaps.py`, `strip_navmap_sections.py`, and CLI wrappers to accept `*, options: NavmapRepairOptions`.
  - Add deprecated wrapper functions logging warnings.

- [ ] 2.3 Cache/collector interface.
  - Define `tools/navmap/cache/interfaces.py` with Protocol exposing required operations.
  - Provide accessor helper; replace `_collect_module` reach-ins with public method on new interface.

- [ ] 2.4 Testing.
  - Add Typer CLI tests verifying invalid combinations emit Problem Details JSON.
  - Run targeted pytest module `uv run pytest tests/tools/navmap -q`.

## Phase 3 – Docs Toolchain

- [ ] 3.1 Add docs configs.
  - Create `docs/toolchain/config.py` with `DocsSymbolIndexConfig` and `DocsDeltaConfig` dataclasses.
  - Tests: `tests/docs/test_toolchain_config.py` covering defaults/invalid combos.

- [ ] 3.2 Update toolchain scripts.
  - Refactor `docs/toolchain/build_symbol_index.py`, `symbol_delta.py`, `validate_artifacts.py` to require keyword-only `config`.
  - Replace `_ParameterKind` access with public helper returning safe value.

- [ ] 3.3 Legacy shims and docs.
  - Keep thin compatibility modules under original names that import new package, warn once, and call `main()`.
  - Update docstrings and README snippet; run `uv run pytest tests/docs/test_toolchain_public_api.py --doctest-modules docs/toolchain`.

## Phase 4 – Orchestration & Registry

- [ ] 4.1 Define orchestration configs in `src/orchestration/config.py`.
  - Add tests in `tests/orchestration/test_config_models.py`.

- [ ] 4.2 Refactor CLI.
  - Update `src/orchestration/cli.py:index_faiss` to accept `*, config: IndexCliConfig`.
  - Adjust CLI option parsing, add `DeprecationWarning` shim, update docstring with Problem Details info.

- [ ] 4.3 Logging cache interface.
  - Publish `LoggingCache` Protocol in `src/kgfoundry_common/logging.py` and accessor function.
  - Replace `_cache` reach-ins in runtime modules/tests.

## Phase 5 – ConfigurationError & Problem Details

- [ ] 5.1 Extend error taxonomy.
  - Add `ConfigurationError` to `src/kgfoundry_common/errors/exceptions.py` with helper `.with_details(field, issue, hint=None)`.
  - Implement helper `build_configuration_problem` in `src/kgfoundry_common/errors/problem_details.py`.

- [ ] 5.2 Schema + sample payload.
  - Create `schema/examples/problem_details/public-api-invalid-config.json` and validate via `uv run python -m jsonschema --instance ...`.

- [ ] 5.3 Wire CLIs.
  - Catch `ConfigurationError` in each CLI; log structured info and print Problem Details JSON before exiting with code `2`.
  - Add Typer `CliRunner` tests confirming stdout JSON contains expected keys.

## Phase 6 – Enforcement & Documentation

- [ ] 6.1 Update Ruff & lint config.
  - Remove existing `FBT`/`SLF` exclusions in `pyproject.toml`.
  - Ensure `python -m tools.lint.check_typing_gates` still passes; adjust allowlist if new modules introduced.

- [ ] 6.2 Regression suite.
  - Add tests verifying caches accessed via Protocol, config invalid cases produce Problem Details, and CLI help shows new options.

- [ ] 6.3 Documentation & comms.
  - Update `AGENTS.md` (Clarity & API design) with new dataclass example.
  - Add CHANGELOG entry summarising migration and deprecation timeline.
  - Update subsystem READMEs with before/after code snippets.

- [ ] 6.4 Quality gates and artifacts.
  - Run in order: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pytest -q`, `make artifacts`, `python tools/check_new_suppressions.py src`, `python tools/check_imports.py`.
  - Store command outputs in `openspec/changes/public-api-hardening-phase1/artifacts/quality-gate/`.

- [ ] 6.5 Post-merge follow-up.
  - File Phase 2 ticket tracking removal of shims once telemetry indicates no legacy usage for one release cycle.
  - Add monitoring note to runbook for checking deprecation warnings in CI logs.


