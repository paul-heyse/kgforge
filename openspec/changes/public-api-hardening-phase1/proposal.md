## Why

Our docs and tooling surfaces still expose unclear public APIs that violate the AGENTS.md design contract:

- **Boolean positional arguments** are common (`run(..., False, True)` in docstring builder orchestrator, navmap repair, registry helpers). Ruff marks these with `FBT001–FBT003`; the functions are hard to understand and impossible to call safely without reading source.
- **Private attribute reach-ins** (`builder._cache`, `shared._ParameterKind`) leak implementation details to callers and trigger `SLF001` warnings. These patterns make refactors risky and bypass validation/metrics.
- **Configuration error handling** falls back to plain `ValueError` or bare `Exception`, so Problem Details documentation is missing. CLI users receive inconsistent messaging that cannot be observed centrally.

The proposal introduces typed configuration objects, cache interfaces, and structured error handling so junior engineers can extend the system safely without memorising private conventions. This is foundational for the typing-gate and packaging work already planned.

## What Will Change (High-Level)

| Area | Current State | Target State |
| --- | --- | --- |
| Function signatures | Boolean positional flags, optional positional args | Keyword-only parameters receiving frozen dataclasses / TypedDict configs with explicit defaults |
| Cache access | Callers dereference `_cache`, `_ParameterKind` | Public Protocol interfaces + accessor functions returning documented objects |
| Error handling | Raw exceptions, no schema linkage | `ConfigurationError` hierarchy emitting RFC 9457 Problem Details with schema examples |

The change covers docstring builder, navmap tooling, docs toolchain CLIs, and registry adapters—the modules where Ruff and pyright currently report positional Boolean or private attribute usage.

## Deliverables

1. **Capability spec** `code-quality/public-apis` describing mandatory keyword-only configs, cache interfaces, and Problem Details expectations.
2. **Typed configuration objects** (`DocstringBuildConfig`, `NavmapRepairOptions`, `DocsIndexBuildConfig`, etc.) with validation and docstrings.
3. **Cache Protocols** published under `tools.docstring_builder.cache.interfaces` and `tools.navmap.cache.interfaces` plus façade helpers.
4. **Problem Details example** `schema/examples/problem_details/public-api-invalid-config.json` referenced from CLI docstrings and tests.
5. **Regression suite** ensuring legacy positional arguments raise informative `TypeError` and caches are only accessed via interfaces.
6. **Deprecation shims** for existing call sites with metrics/warnings so downstream teams can migrate safely.

## Acceptance Criteria (must all pass)

- Ruff (`uv run ruff format && uv run ruff check --fix`) returns **zero** `FBT00x` and `SLF001` in the touched modules **without ignores**.
- Pyright, Pyrefly, and MyPy run clean after the API changes.
- `pytest -q` includes new parameterized tests that cover:
  - Happy-path usage of each config object.
  - Validation failure raising `ConfigurationError` with Problem Details payload.
  - Cache access solely through the published Protocols.
- CLI smoke tests demonstrate `python -m tools.docstring_builder.orchestrator --help` and failure paths print the documented Problem Details JSON.

## Rollout & Compatibility

1. Phase 1 (this change): introduce new APIs + shims that log `DeprecationWarning` when positional args/private attributes used. Metrics recorded via structured logs.
2. Phase 2 (subsequent change): remove shims once dashboards show no usage for one release cycle.

## Dependencies

- Must land after **docs/tools packaging** (packages provide stable import roots).
- Coordinates with **typing gates Phase 2** so new imports follow façade rules.

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Downstream automation breaks due to signature changes | Ship shims, add telemetry counters, publish migration guide in CHANGELOG and AGENTS.md |
| Cache Protocol misses edge method | Inventory usages before refactor; add contract tests to confirm parity |
| Problem Details duplication | Use shared helper in `kgfoundry_common.errors`, validate via jsonschema |

## Alternatives Considered

1. **Documenting positional args** – rejected because Ruff would still fail and junior engineers would continue to guess semantics.
2. **Accepting `**kwargs` with manual parsing** – rejected because it loses type safety and violates clarity principle.
3. **Hiding caches completely** – rejected; legitimate read-only access exists (e.g., Agent Portal). Instead we expose curated interfaces certified for downstream use.


