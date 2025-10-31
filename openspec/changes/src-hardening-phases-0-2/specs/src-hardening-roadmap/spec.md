## ADDED Requirements
### Requirement: Phase 0 Quality Gates
The hardening initiative SHALL establish deterministic quality gates and collect baseline diagnostics before any implementation work begins.

#### Scenario: Baseline diagnostics captured
- **GIVEN** the repository is synced and `scripts/bootstrap.sh` has been run successfully
- **WHEN** the engineer executes the Phase 0 command set (`uv run ruff format && uv run ruff check src`, `uv run pyrefly check src --show-suppressed`, `uv run mypy --config-file mypy.ini src`, `uv run pytest -q --maxfail=1`, doctest coverage snapshot, `python tools/check_imports.py`, `uv run pip-audit --strict`, `openspec validate src-hardening-phases-0-2`, and docstring coverage tooling) and publishes the four-item design note mapping requirements to tests
- **THEN** the command outputs, coverage metrics, and telemetry dashboard links are attached to the execution note prior to coding

### Requirement: Phase 2 Typed Refactors & Schemas
The hardening initiative SHALL remove `Any` flows, enforce JSON Schema 2020-12/OpenAPI 3.2 contracts, and split high-complexity functions into typed, tested units guarded by feature flags.

#### Scenario: Typed pathways enabled safely
- **GIVEN** Phase 0 requirements are satisfied and feature flags guard new pathways
- **WHEN** engineers deliver typed, documented APIs with Problem Details samples, validate schemas and round-trip tests, expand parametrized/doctor-tested suites (including error-path tables), harden logging/config/security boundaries, enforce layering/import rules, and record performance & idempotency benchmarks
- **THEN** pyrefly/mypy emit zero diagnostics, lint/tests/benchmarks pass within budgets, compatibility shims remain in place, and telemetry shows typed pathways operating without regressions

#### Scenario: Settings missing fails fast
- **GIVEN** typed environment settings are required for search/catalog services
- **WHEN** a required environment variable is absent during startup
- **THEN** the service raises a `SettingsError`, emits an RFC 9457 Problem Details payload, and logs a structured error with correlation ID

#### Scenario: Deprecated flag warns once
- **GIVEN** a CLI or HTTP flag is marked deprecated with a documented removal version
- **WHEN** a client invokes the deprecated flag during Phase 2 rollout
- **THEN** exactly one structured warning is emitted (log + telemetry) with migration guidance and removal version, and tests assert the warning count

#### Scenario: Async context propagated
- **GIVEN** async request handlers span multiple await points
- **WHEN** a request flows through typed pathways
- **THEN** correlation/context IDs persist via `contextvars`, no blocking calls occur in async code (thread pools used when needed), and tests verify propagation

### Requirement: Phase 3 Validation & Rollout
The hardening initiative SHALL execute acceptance gates, monitor telemetry, and archive the rollout once typed pathways are stable in production-like environments.

#### Scenario: Rollout completes with artefacts archived
- **GIVEN** Phase 2 typed pathways are enabled behind feature flags
- **WHEN** engineers rerun the full quality/packaging/docs gates (`pip wheel .`, clean installs, `make artifacts`, Agent Portal checks), verify telemetry dashboards emit log+metric+trace with correlation IDs, update CHANGELOG with SemVer guidance, validate package metadata (PEP 621 fields), enforce Agent Portal deep links, and document rollback steps
- **THEN** feature flags flip to their permanent state after staging burn-in, and the execution note archives command outputs, telemetry snapshots, benchmarks, metadata validation, and residual risks for audit

