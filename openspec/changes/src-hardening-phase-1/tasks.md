## 0. Orientation (AI agent pre-flight)
- [ ] 0.1 Confirm runtime matches repo toolchain (`scripts/bootstrap.sh`).
- [ ] 0.2 Cache `openspec/changes/src-hardening-phase-1/` and `openspec/AGENTS.md` locally.
- [ ] 0.3 Read canonical modules listed in `README.md` step 3; keep them open as authoritative contracts.
- [ ] 0.4 Capture baseline diagnostics (attach logs to execution note):
  - `uv run ruff check src/kgfoundry_common`
  - `uv run pyrefly check src/kgfoundry_common`
  - `uv run mypy src/kgfoundry_common`
  - `python tools/check_imports.py`
  - `uv run pip-audit --strict`
- [ ] 0.5 Produce a four-item design note (Summary, API sketch, Data contracts, Test plan) summarizing planned edits.

## 1. Problem Details & Error Taxonomy
- [ ] 1.1 Create `schema/common/problem_details.json` (RFC 9457 aligned) plus fixtures under `docs/examples/problem_details/`.
- [ ] 1.2 Add `src/kgfoundry_common/problem_details.py` exporting `ProblemDetails`, builders (`build_problem_details`, `problem_from_exception`, `render_problem`).
- [ ] 1.3 Refactor `errors/exceptions.py` to:
  - inherit from `KgFoundryError`
  - expose `code`, `title`, `http_status`
  - update `to_problem_details` signature to include optional `title`
  - remove unused `type: ignore`
- [ ] 1.4 Update `errors/__init__.py` and `errors/http.py` exports; ensure modules import from `problem_details` where needed.
- [ ] 1.5 Tests: add `tests/kgfoundry_common/test_problem_details.py` covering success/failure, schema validation, feature-flag fallback.

## 2. Structured Logging
- [ ] 2.1 Refactor `logging.py`:
  - Delegate to shared JSON formatter (compatible with `tools/_shared/logging`).
  - Provide `get_logger`, `with_fields`, `CorrelationContext` using `contextvars`.
  - Ensure library modules install `NullHandler`.
- [ ] 2.2 Add doctests verifying usage and correlation ID propagation; include golden fixture under `tests/kgfoundry_common/golden/logging.json`.
- [ ] 2.3 Update consumers (if any) within `src/kgfoundry_common` to use new helpers.
- [ ] 2.4 Tests: `tests/kgfoundry_common/test_logging.py` table-driven cases (basic log, extra fields, context manager, feature flag off).

## 3. Observability Helpers
- [ ] 3.1 Redesign `observability.py` metrics provider:
  - Accept optional `CollectorRegistry` (real or stub).
  - Provide `Counter`, `Histogram`, `Gauge` wrappers with `.labels()` returning typed helpers.
  - Add OpenTelemetry span helper (`start_span`) with safe fallback.
- [ ] 3.2 Ensure stub implementations satisfy type checkers (methods return `self`).
- [ ] 3.3 Add Prometheus version detection (optional import) with graceful degradation.
- [ ] 3.4 Tests: `tests/kgfoundry_common/test_observability.py` covering stub vs real, label usage, metric increments, span creation.

## 4. Serialization & Schemas
- [ ] 4.1 Replace `jsonschema.SchemaError` references with imports from `jsonschema.exceptions`.
- [ ] 4.2 Implement schema loader cache (LRU or module-level dict) in `serialization.py` using typed aliases (`JsonValue`).
- [ ] 4.3 Add typed exceptions (`SchemaValidationError`, `SerializationError`) raising Problem Details payloads.
- [ ] 4.4 Tests: `tests/kgfoundry_common/test_serialization.py` table-driven (valid payload, invalid payload, missing schema).

## 5. Configuration & Settings
- [ ] 5.1 Define `JsonPrimitive`, `JsonValue` TypeAlias in `config.py` and reuse across modules.
- [ ] 5.2 Refactor `settings.py` to use `pydantic_settings` models with explicit fields and environment prefixes; remove wildcard `**kwargs` instantiation.
- [ ] 5.3 Add typed helper `load_settings()` returning `KgFoundrySettings`, raising `SettingsError` with Problem Details when env missing.
- [ ] 5.4 Doctests demonstrating configuration usage; update developer docs if necessary.
- [ ] 5.5 Tests: `tests/kgfoundry_common/test_settings.py` verifying env overrides, missing env failure, Problem Details conversion.

## 6. Documentation & Rollout
- [ ] 6.1 Update `docs/examples/` with new Problem Details and logging samples.
- [ ] 6.2 Document feature flags (`KGFOUNDRY_LOGGING_V2`, `KGFOUNDRY_PROBLEM_DETAILS_V2`) in developer docs + design appendix.
- [ ] 6.3 Prepare changelog entry, including migration notes for downstream services.
- [ ] 6.4 Add telemetry checklist: dashboards to monitor log ingestion & metric cardinality during rollout.

## 7. Validation & Sign-off
- [ ] 7.1 Run acceptance gates (listed in README) with outputs attached to execution note.
- [ ] 7.2 Ensure `ruff`, `pyrefly`, `mypy` show zero diagnostics under `src/kgfoundry_common`.
- [ ] 7.3 Capture schema validation logs, doctest output, pytest summary.
- [ ] 7.4 Submit execution note summarizing impact, telemetry plan, and flip schedule for feature flags.

## Appendix A — Module-by-module plan (junior-ready)

### problem_details.py
- Create `ProblemDetails` (TypedDict) and helpers:
  - `build_problem_details`, `problem_from_exception`, `render_problem`.
- Validate against `schema/common/problem_details.json` using `jsonschema.Draft202012Validator`.
- Tests: valid payload, invalid payload (raises `ProblemDetailsValidationError`), problem-from-exception preserves cause (`raise ... from e`).

### errors/**
- Refactor exceptions to inherit from `KgFoundryError`.
- Add `code`, `title`, `http_status`.
- Update `to_problem_details(self, *, title: str | None = None) -> ProblemDetails`.
- Ensure all raises use `raise ... from e`.
- Update exports in `errors/__init__.py` and `errors/http.py`.
- Tests: taxonomy import hygiene, `to_problem_details` JSON validates, cause chain.

### logging.py
- Delegate to unified JSON formatter; ensure `NullHandler` is attached in library modules.
- Provide `get_logger`, `with_fields`, `CorrelationContext`.
- Doctests: one-line summaries present; examples run.
- Tests: JSON includes `correlation_id`, `operation`, `status`, `duration_ms`; feature flag toggles legacy path.

### observability.py
- Implement `MetricsProvider` with stub-safe `.labels()`; expose counter/histogram helpers and `observe_duration` context manager.
- Optional OTEL span helper with graceful fallback.
- Tests: stub vs real registry; labels chaining; duration recording increments metrics.

### serialization.py
- Implement `JsonPrimitive`, `JsonValue` TypeAlias; loader cache for schemas.
- Implement `validate_payload(payload, schema_path)` with precise types; import exceptions from `jsonschema.exceptions`.
- Tests: valid/invalid/missing schema; no `Any` in public signatures.

### settings.py
- Define `KgFoundrySettings` (`pydantic_settings`) with explicit `model_config` (env prefix, case sensitivity).
- Implement `load_settings()`; raise `SettingsError` for missing env; emit Problem Details via `to_problem_details`.
- Doctests: happy path + missing env.
- Tests: env overrides, failures produce schema-valid Problem Details.

### Documentation updates
- Add Problem Details examples to `docs/examples/problem_details/`.
- Update developer docs with feature flags (`KGFOUNDRY_LOGGING_V2`, `KGFOUNDRY_PROBLEM_DETAILS_V2`).
- Ensure `make artifacts` regenerates and tree is clean.

## Appendix B — CI & Packaging Checklist
- [ ] Add import-linter contract `common-no-upwards` to `importlinter.cfg` and verify it passes.
- [ ] Enable doctests/xdoctests in CI (already in `pytest.ini`); ensure examples execute.
- [ ] Packaging: `pip wheel .` succeeds; `pip install .[obs,schema]` in a clean venv succeeds.
- [ ] Security: `uv run pip-audit --strict` passes; YAML parsing uses `yaml.safe_load` only.
- [ ] Performance: run pytest-benchmark for validation/logging and record baseline.
- [ ] Artifacts: `make artifacts && git diff --exit-code` remains clean.
## Appendix A — Reference Snippets

### Problem Details Builder Usage
```python
from kgfoundry_common.problem_details import build_problem_details

problem = build_problem_details(
    type="https://kgfoundry.dev/problems/config-missing",
    title="Configuration Missing",
    status=500,
    detail="ENVVAR KGFOUNDRY_API_KEY is required",
    instance="urn:kgfoundry:settings:load",
    extensions={"env": "KGFOUNDRY_API_KEY"},
)
```

### Structured Logging
```python
from kgfoundry_common.logging import get_logger, with_fields

logger = get_logger(__name__)

with with_fields(logger, correlation_id="abc-123", operation="load-settings"):
    logger.info("Loading configuration", extra={"status": "started"})
```

### Metrics Provider
```python
from kgfoundry_common.observability import MetricsProvider

metrics = MetricsProvider.default()

with metrics.observe_duration("load_settings") as obs:
    ...
    obs.success()
```

### Settings Loader
```python
from kgfoundry_common.settings import load_settings, SettingsError

try:
    settings = load_settings()
except SettingsError as exc:
    problem = exc.to_problem_details(instance="urn:kgfoundry:settings")
    # surface Problem Details via HTTP or CLI
```

