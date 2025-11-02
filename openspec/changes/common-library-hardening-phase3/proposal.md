## Why
`kgfoundry_common` underpins logging, error handling, and configuration across the platform, yet the current implementation leaks `Any` types, lacks structured logging helpers, and relies on ad-hoc environment parsing. These gaps break type safety, weaken observability, and allow Problem Details responses to drift away from RFC 9457.

## What Changes
- [x] **ADDED**: Typed structured logging helpers (field-driven `log_success`, `log_failure`, etc.) and module-level `NullHandler` registration for every public logger in `kgfoundry_common`.
- [x] **MODIFIED**: `kgfoundry_common.logging.LoggerAdapter` rewritten with modern generics (`type[...]`), explicit context propagation, and contextvar-aware correlation IDs.
- [x] **MODIFIED**: Configuration surface migrated to `pydantic_settings.BaseSettings` with full docstrings, environment variable documentation, and strict type validation.
- [x] **MODIFIED**: `problem_details`, `errors`, and `serialization` modules updated to raise domain-specific exceptions with `raise ... from e`, avoid f-strings inside exceptions, and capture monotonic timing for duration fields.
- [x] **ADDED**: Regression tests (pytest + xdoctest) ensuring structured logging output, env-config overrides, and Problem Details envelopes stay stable.
- [ ] **REMOVED**: Legacy string-formatting utilities that bypass structured logging and any implicit environment parsing helpers superseded by `pydantic_settings`.

## Impact
- **Affected specs (capabilities):** `kgfoundry-common/core`
- **Affected code paths:** `src/kgfoundry_common/{logging,config,problem_details,errors,serialization}.py`, related tests under `tests/kgfoundry_common/**`, documentation under `docs/reference/common/`
- **Data contracts:** Problem Details JSON examples (`schema/examples/problem_details/**`), any OpenAPI components referencing common error envelopes
- **Rollout plan:** Ship as a coordinated library release; communicate structured logging helper usage and configuration migration; monitor downstream services for logging format changes.

## Acceptance
- [ ] Logger adapter and helpers expose typed APIs, auto-attach `NullHandler`, and produce structured records validated by new tests.
- [ ] Configuration layer uses `pydantic_settings`, documents all environment overrides, and fails fast on invalid values.
- [ ] Problem Details, errors, and serialization modules raise typed exceptions with preserved causes, avoid unsafe formatting, and rely on `time.monotonic()` for durations.
- [ ] Updated Problem Details examples and doc snippets match generated payloads; quality gates (Ruff, pyrefly, mypy, pytest, artifacts) pass without new suppressions.

## Out of Scope
- Refactoring application-specific logging consumers outside `kgfoundry_common`.
- Introducing new telemetry backends (Prometheus/OTel) beyond structured logging fields.
- Broader overhaul of serialization formats beyond ensuring type safety and monotonic timing.

## Risks / Mitigations
- **Risk:** Structured logging format changes may break downstream log parsers.
  - **Mitigation:** Provide adapter functions preserving legacy field names, document migration steps, and test JSON log output.
- **Risk:** Enforcing strict env validation could cause startup failures in misconfigured environments.
  - **Mitigation:** Offer migration guide, include default `.env.example`, and add targeted tests ensuring helpful error messages.
- **Risk:** Raising new exception types may surface differently in existing handlers.
  - **Mitigation:** Maintain backward-compatible subclassing hierarchy and update documentation/examples accordingly.

## Alternatives Considered
- Continue relying on implicit dict payloads for logging — rejected because it perpetuates `Any` usage and inconsistent field names.
- Adopt a third-party logging framework — rejected to minimize dependency footprint and keep stdlib compatibility.
- Keep existing config parser with incremental fixes — rejected because it cannot enforce the required type guarantees or doc generation.

