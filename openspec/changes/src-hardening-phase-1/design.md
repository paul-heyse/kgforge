## Context
`kgfoundry_common` is the platform layer for every app under `src/`. Today, logging adapters diverge from the structured logging contract, Problem Details helpers return loosely typed dicts, observability shims contain broken Prometheus stub logic, serialization utilities rely on `Any`, and settings constructors accept arbitrary kwargs. These issues generate hundreds of mypy/pyrefly diagnostics, hide telemetry failures, and jeopardize error reporting. Phase 1 establishes typed, schema-backed, observable foundations so later phases (search, embeddings, registry) can build safely.

## Goals / Non-Goals
- **Goals**
  - Provide typed Problem Details & error taxonomy with JSON Schema validation.
  - Unify structured logging via a shared adapter that injects correlation IDs and context fields.
  - Deliver observability helpers with safe Prometheus integrations and OpenTelemetry hooks.
  - Replace dynamic serialization helpers with typed schema validation and explicit exceptions.
  - Adopt `pydantic_settings` models for configuration, ensuring env-only sources and fast failure.
- **Non-Goals**
  - Changing HTTP frameworks or surface APIs (handled in later phases).
  - Redesigning domain-specific error codes outside common taxonomy.
  - Implementing new metrics dashboards (only instrumentation hooks provided now).

## Architecture Overview

| Component | Before | After |
| --- | --- | --- |
| Error taxonomy | Mixed exports, inconsistent `to_problem_details` signatures, loose dicts | `KgFoundryError` hierarchy with typed `ProblemDetails` dataclass + schema validation |
| Logging | Custom adapter returning dicts, no correlation ID | Delegates to shared `StructuredLoggerAdapter` with JSON formatting, `with_fields` helper, NullHandler |
| Observability | Stub metrics lacking `.labels()`, runtime errors if Prometheus unavailable | `MetricsProvider` dataclass encapsulating counters/histograms, stub implementations returning `self`, typed registry handling |
| Serialization | `Any`-typed schema helpers catching broad exceptions | Typed functions using `jsonschema.Draft202012Validator`, raising `SerializationError` subclasses |
| Settings | Manual `BaseSettings` instantiation via `**dict`, `Any` leakage | `@dataclass`-style `pydantic_settings` classes with explicit fields, environment-only configuration |

### Error & Problem Details Design
- Introduce `ProblemDetails` TypedDict and `ProblemDetailsBuilder` helper with defaults (type, title, status, detail, instance, extensions).
- JSON Schema: `schema/common/problem_details.json` (RFC 9457 aligned) with examples under `docs/examples/problem_details/*.json`.
- Error classes subclass `KgFoundryError` and override `code`, `title`, `http_status`. `to_problem_details` accepts optional `title` to comply with base signature.
- Provide mapping utility `as_problem_response(exception, instance, extra)` returning schema-validated JSON + log entry.

Typed API sketch (public):
```python
class ProblemDetails(TypedDict):
    type: str
    title: str
    status: int
    detail: str
    instance: str
    extensions: dict[str, object] | NotRequired[dict[str, object]]

def build_problem_details(*, type: str, title: str, status: int, detail: str, instance: str, extensions: Mapping[str, object] | None = None) -> ProblemDetails: ...
def problem_from_exception(exc: Exception, *, type: str, title: str, status: int, instance: str, extensions: Mapping[str, object] | None = None) -> ProblemDetails: ...
def render_problem(problem: ProblemDetails) -> str: ...
```

### Logging Integration
- `kgfoundry_common.logging` exposes:
  - `get_logger(name: str) -> logging.Logger`
  - `with_fields(logger, **fields)` context manager to attach correlation IDs.
  - Delegation to repo-wide JSON formatter (consistent with `tools/_shared/logging`).
- Provide `CorrelationContext` using `contextvars` to propagate IDs across async tasks.
- Add doctests demonstrating usage.

Minimum structured fields: `correlation_id`, `operation`, `status`, `duration_ms`, `command`.
Libraries MUST add `NullHandler` to module loggers.

### Observability
- `MetricsProvider` dataclass encapsulates counters/histograms. Accepts optional `CollectorRegistry`; defaults to stub class returning `self` for `.labels()`.
- Expose functions to create timers (`observe_duration(operation, status)`).
- Provide OpenTelemetry span helper (if OTEL available) with graceful fallback.
- Tests cover real registry (when dependency available) and stub mode.

Metrics baseline:
- `kgfoundry_runs_total{component,status}` (counter)
- `kgfoundry_operation_duration_seconds{component,operation,status}` (histogram)

### Serialization
- Replace `jsonschema.SchemaError` references with actual `jsonschema.exceptions.SchemaError` import.
- Provide `validate_payload(payload, schema_path)` returning typed result or raising `SchemaValidationError`.
- Add caching for loaded schemas (LRU or global dict) to avoid repeated I/O.
- Ensure functions declare precise return types (e.g., `dict[str, JsonValue]`).

Typed signatures:
```python
JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]

def validate_payload(payload: Mapping[str, object], schema_path: Path) -> None: ...
```

### Configuration & Settings
- Define `JsonPrimitive`, `JsonValue`, `SettingsSource` TypeAlias to reuse across modules.
- Each settings class inherits from `BaseSettings` with explicit `model_config` (env prefix, case sensitivity, validation). Remove arbitrary kwargs; add `SettingsError` for missing env.
- Provide CLI snippet for generating `.env.example` based on `model_fields`.

Typed signatures:
```python
class KgFoundrySettings(BaseSettings): ...

class SettingsError(KgFoundryError):
    def to_problem_details(self, instance: str) -> ProblemDetails: ...

def load_settings() -> KgFoundrySettings: ...
```

### Layering & Import Boundaries
Use `importlinter.cfg` to enforce architecture:
```ini
[contract:common-no-upwards]
name = common must not depend on app layers
type = forbidden
source_modules = src.kgfoundry_common
forbidden_modules =
    src.search_api
    src.registry
    src.embeddings_dense
    src.embeddings_sparse
```

### Packaging & Extras
Define optional extras in `pyproject.toml`:
```toml
[project.optional-dependencies]
obs = ["prometheus-client>=0.20", "opentelemetry-api>=1.26"]
schema = ["jsonschema>=4.23"]
```
CI should verify:
```bash
pip wheel .
python -m venv /tmp/v && /tmp/v/bin/pip install .[obs,schema]
```

### Doctests & Examples
Enable `xdoctest` via `pytest.ini` (already configured). All public modules include short, copy-ready examples that run in CI.

### Micro-benchmarks & Budgets
Add pytest-benchmark for hot paths (validation, logging formatting). Set local budgets (informational at first) and track regressions.
```python
def test_validate_problem_benchmark(benchmark):
    payload = {...}
    schema = Path("schema/common/problem_details.json")
    benchmark(validate_payload, payload, schema)
```

## Detailed Implementation Plan

| Step | Description | Acceptance |
| --- | --- | --- |
| 1 | Introduce `problem_details.py`, schema + fixtures | Tests verifying schema validation + example round-trip |
| 2 | Refactor `errors/__init__.py` & `exceptions.py` | mypy/pyrefly clean; no unused ignores; taxonomy docstrings |
| 3 | Update `logging.py` to use structured adapter & `contextvars` | Integration tests verifying JSON log shape + correlation IDs |
| 4 | Harden `observability.py` metrics provider | Stub + real registry tests; no pyrefly errors |
| 5 | Rewrite serialization helpers with typed schema validation | Table-driven tests (valid, invalid, missing schema) |
| 6 | Replace settings constructors with `pydantic_settings` models | Doctest coverage; environment failures produce Problem Details |
| 7 | Update docs/examples + README, add developer guide snippet | `make artifacts` clean |

Defaults & validation (apply unless overridden):
- Use `yaml.safe_load` only; never `yaml.load`.
- Time sources: `time.monotonic()` for durations; `datetime.now(tz=UTC)` for timestamps.
- Paths via `pathlib.Path`; reject `os.path` in new code.

## Data Contracts & Schemas
- `schema/common/problem_details.json` — canonical Problem Details.
- `schema/common/metrics_envelope.json` — optional schema describing metrics configuration (documented now, used later phases).
- All schemas validated against JSON Schema Draft 2020-12 meta-schema; tests ensure round-trip using new serialization utilities.

## Observability Strategy
- Logs: JSON lines with `timestamp`, `level`, `message`, `correlation_id`, `operation`, `status`, `duration_ms`, `error_code`.
- Metrics: counters/histograms prefixed `kgfoundry_`; stub ensures `.labels()` safe without Prometheus.
- Traces: optional `start_span(operation)` helper writing to OTEL if installed; fallback no-op.
- Provide example configuration in docs referencing metrics/traces dashboards.

## Risks & Mitigations
- **Logging format drift** → snapshot tests + golden fixtures.
- **Increased validation cost** → cache compiled schemas, limit to boundaries.
- **Breaking downstream modules** → feature flags `KGFOUNDRY_LOGGING_V2`, `KGFOUNDRY_PROBLEM_DETAILS_V2`; maintain compatibility wrappers until rollout.
- **Dependency gaps (Prometheus, jsonschema)** → stub fallbacks and optional extras documented.

## Rollout Plan
1. Land new modules + tests hidden behind feature flags (default legacy).
2. Enable `KGFOUNDRY_LOGGING_V2=1` in staging; monitor log ingestion/dashboards.
3. Enable `KGFOUNDRY_PROBLEM_DETAILS_V2=1` in staging; confirm clients parse new envelope.
4. Roll to production after 7-day stability; remove legacy code in Phase 2.
5. Update changelog and developer docs; mark feature flags for removal timeline.

## Migration / Backout
- Backout by toggling feature flags to `0`; legacy helpers remain until Phase 2.
- Schema updates documented with version constants; revert by pinning previous schema version.
- Logging revert documented via golden fixtures to ensure quick diff.

