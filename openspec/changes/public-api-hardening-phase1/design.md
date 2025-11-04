# Design

> **Goal:** Provide an implementation playbook a junior engineer can follow without reverse engineering existing code. Each section lists files, commands, and validation steps.

## 1. System Overview

We touch four subsystems that currently expose ambiguous public APIs:

1. **Docstring builder pipeline** (`tools/docstring_builder/**`).
2. **Navmap toolkit** (`tools/navmap/**`).
3. **Docs toolchain CLIs** (`docs/toolchain/**`).
4. **Orchestration/registry adapters** (`src/orchestration/cli.py`, `src/kgfoundry_common/**`).

Across these subsystems we will:

- Replace boolean positional arguments with keyword-only typed configs.
- Publish cache interfaces so callers stop using private fields.
- Standardise configuration error handling via Problem Details.

## 2. Target Architecture

### 2.1 Config Models

Create new modules housing configuration dataclasses/TypedDicts:

| New File | Type(s) | Consumed By |
| --- | --- | --- |
| `tools/docstring_builder/config_models.py` | `DocstringBuildConfig`, `DocstringReportConfig` | `tools.docstring_builder.orchestrator`, CLI |
| `tools/navmap/config.py` | `NavmapRepairOptions`, `NavmapStripOptions` | `tools.navmap.repair_navmaps`, `tools.navmap.strip_navmap_sections` |
| `docs/toolchain/config.py` | `DocsSymbolIndexConfig`, `DocsDeltaConfig` | `docs.toolchain.build_symbol_index`, `docs.toolchain.symbol_delta` |
| `src/orchestration/config.py` | `IndexCliConfig`, `ArtifactValidationConfig` | `src/orchestration/cli.py`, downstream services |

Each dataclass follows this pattern:

```python
@dataclass(frozen=True, slots=True)
class DocstringBuildConfig:
    cache_policy: CachePolicy = CachePolicy.READ_WRITE
    enable_plugins: bool = True
    emit_diff: bool = False
    timeout_seconds: PositiveInt = PositiveInt(600)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError.with_details(
                field="timeout_seconds",
                issue="Must be > 0",
            )
```

`ConfigurationError.with_details` (added in Section 2.3) builds the Problem Details payload.

### 2.2 Public API Signatures

All public functions become keyword-only. Example for docstring builder:

```python
def run_build(*, config: DocstringBuildConfig, cache: DocstringBuilderCache) -> DocstringBuildResult:
    """Execute docstring build with typed configuration.

    Parameters
    ----------
    config : DocstringBuildConfig
        Typed configuration controlling plugins, diff emission, timeout, etc.
    cache : DocstringBuilderCache
        Cache interface for storing/retrieving docstrings.

    Returns
    -------
    DocstringBuildResult
        Summary plus metrics.

    Raises
    ------
    ConfigurationError
        If config options conflict (e.g., emit_diff requires enable_plugins).
    """
```

CLI commands call this function by constructing `DocstringBuildConfig` from Typer arguments.

### 2.3 Cache Interfaces

Add Protocols describing caches. Example: `tools/docstring_builder/cache/interfaces.py`.

```python
class DocstringBuilderCache(Protocol):
    """Public cache contract used by CLI and orchestrator."""

    def get(self, key: str, /) -> CachedDoc | None: ...
    def put(self, key: str, doc: CachedDoc, /) -> None: ...
    def invalidate(self, key: str, /) -> None: ...
    def stats(self) -> CacheStats: ...
```

Implementation classes (e.g., `BuilderCache`) declare `class BuilderCache(DocstringBuilderCache): ...`. Expose a helper `get_docstring_cache()` returning the interface instance. Accessing `_cache` will raise `DeprecationWarning` with guidance to use the helper.

### 2.4 ConfigurationError & Problem Details

- Add `ConfigurationError` to `src/kgfoundry_common/errors/exceptions.py` with fields `field`, `detail`, `hint`.
- Provide helper `build_configuration_problem(config_error: ConfigurationError) -> ProblemDetails` in `src/kgfoundry_common/errors/problem_details.py`.
- Define schema example `schema/examples/problem_details/public-api-invalid-config.json` with sample response.
- CLI error paths catch `ConfigurationError`, log structured info, print Problem Details JSON, and exit code 2.

## 3. Step-by-Step Implementation Plan

### Phase 0 — Preparation (0.5 day)
1. Create `openspec/changes/public-api-hardening-phase1/artifacts/` folder.
2. Run `uv run ruff check --select FBT,SLF --output-format json > artifacts/ruff-public-api.json`.
3. Run `rg "_cache" -n tools docs src` and save matches to `artifacts/private-cache-usage.txt`.
4. Document each finding in `artifacts/api-inventory.md` (table format: module, current signature, proposed config fields, notes).
5. Review inventory with subsystem maintainers (record meeting notes in the artifact).

### Phase 1 — Docstring Builder (2–3 days)
1. **Introduce config models** in `tools/docstring_builder/config_models.py` with tests in `tests/tools/docstring_builder/test_config_models.py`.
2. Update `tools/docstring_builder/orchestrator.py`:
   - Replace old signature with keyword-only `config` + `cache`.
   - Convert CLI (Typer) arguments to build config object.
   - Add deprecation wrapper `run_legacy(*args, **kwargs)` logging `DeprecationWarning`.
3. Create cache Protocol file and update `tools/docstring_builder/cache.py` to implement interface; add helper `get_builder_cache()`.
4. Update consumers (`tools/docstring_builder/normalizer.py`, tests) to call helper.
5. Add tests verifying positional call raises `TypeError` and `_cache` access triggers warning.
6. Update docstrings/examples and ensure doctests pass via `pytest --doctest-modules tools/docstring_builder`.

### Phase 2 — Navmap Toolkit (2 days)
Repeat steps analogous to Phase 1 for `tools/navmap`:
- New config module with dataclasses.
- Update `repair_navmaps`, `strip_navmap_sections`, and CLI wrappers.
- Publish `NavmapCache` Protocol and accessor.
- Add Typer CLI tests verifying Problem Details on invalid options (e.g., `--dry-run` with `--write` conflict).

### Phase 3 — Docs Toolchain (1.5 days)
- Introduce `docs/toolchain/config.py` with configs for symbol index build and delta.
- Update `docs/toolchain/build_symbol_index.py`, `docs/toolchain/symbol_delta.py`, `docs/toolchain/shared.py` to use configs and helpers.
- Remove direct `_ParameterKind` usage; expose helper function `normalise_parameter_kind()` returning safe interface.
- Add tests under `tests/docs/test_toolchain_public_api.py` verifying new signatures.

### Phase 4 — Orchestration / Registry (1.5 days)
- Add `src/orchestration/config.py` with CLI config dataclass.
- Update `src/orchestration/cli.py` to use config and Problem Details on invalid combos.
- Ensure `src/kgfoundry_common/logging` caches exposed via `LoggingCache` Protocol.
- Refresh tests (`tests/orchestration/test_cli_public_api.py`).

### Phase 5 — Error & Observability Wiring (1 day)
1. Add `ConfigurationError` + helper + schema example.
2. Update CLI error handling across subsystems to call helper.
3. Add integration tests verifying Problem Details JSON matches schema (use `jsonschema` validation in test).

### Phase 6 — Enforcement & Cleanup (1 day)
1. Remove Ruff ignores for `FBT`/`SLF` in `pyproject.toml`.
2. Update `tools/lint/check_typing_gates` allowlist to include new config modules if necessary.
3. Run full quality gates sequentially:
   - `uv run ruff format && uv run ruff check --fix`
   - `uv run pyrefly check`
   - `uv run pyright --warnings --pythonversion=3.13`
   - `uv run pyright --warnings --pythonversion=3.13`
   - `uv run pytest -q`
   - `make artifacts`
   - `python tools/check_new_suppressions.py src`
   - `python tools/check_imports.py`
4. Save outputs to `artifacts/quality-gate/`.
5. Update documentation: `AGENTS.md` Appendix (API clarity examples), `CHANGELOG.md`, and subsystem READMEs.
6. Create follow-up ticket “Remove public API shims (Phase 2)” once telemetry instrumentation ready.

## 4. Testing Plan

| Test | Purpose | Command |
| --- | --- | --- |
| Config unit tests | Validate defaults & error cases | `uv run pytest tests/tools/docstring_builder/test_config_models.py` |
| CLI integration | Ensure CLI uses config + emits Problem Details | `uv run pytest tests/tools/test_docstring_cli.py` |
| Cache protocol tests | Confirm interface contract & warnings | `uv run pytest tests/tools/test_cache_interfaces.py` |
| Docs toolchain tests | Validate new APIs & docstrings | `uv run pytest tests/docs/test_toolchain_public_api.py --doctest-modules docs/toolchain` |
| Schema validation | Problem Details matches schema | `uv run pytest tests/common/test_configuration_problem_details.py` |

All commands must run inside project root (`/home/paul/kgfoundry`).

## 5. Documentation & Developer Guidance

- Update module docstrings with NumPy-style sections referencing new config classes.
- Add Example block in each CLI docstring showing `python -m ... --help` output.
- Extend `docs/toolchain/README.md` and `tools/docstring_builder/README.md` with “Before vs After” usage snippet.
- Update `AGENTS.md` section “Clarity & API design” to show recommended dataclass-based configuration snippet and reference this change ID.

## 6. Risk Register

| Risk | Impact | Plan |
| --- | --- | --- |
| Legacy automation not updated | CLI failures | Ship deprecation warnings + include link to migration docs, monitor logs |
| Cache protocols incomplete | Runtime errors | Inventory prior to refactor, add contract tests, keep shim for one release |
| Problem Details schema drift | Invalid documentation | Validate via jsonschema in CI, include in `make artifacts` |

## 7. Exit Criteria Checklist

- [ ] Inventory + design notes approved by subsystem owners.
- [ ] Config dataclasses & CLI updates merged for docstring builder, navmap, docs toolchain, orchestration.
- [ ] Cache Protocols published and private attribute usage replaced.
- [ ] `ConfigurationError` helper and schema example live; CLI failure prints Problem Details.
- [ ] Ruff/Pyright/Pyrefly/Mypy/Pytest/Make artifacts green with logs stored.
- [ ] AGENTS.md, CHANGELOG, READMEs updated; deprecation timeline noted.
- [ ] Follow-up ticket opened for removing shims after telemetry sign-off.


