# Src Hardening Phase 1 — Contributor Guide

Phase 1 targets the shared infrastructure that underpins every service under `src/`: error taxonomy, structured logging, observability helpers, serialization, and typed configuration. This guide augments `proposal.md`, `design.md`, `tasks.md`, and the capability deltas under `specs/src-common-infrastructure/` so autonomous agents can execute the plan deterministically.

## Quick Start Checklist
1. `scripts/bootstrap.sh` — ensure Python 3.13.9 + `uv` toolchain is pinned.
2. Read:
   - `proposal.md` for scope, rollout, acceptance gates
   - `design.md` for architecture, error taxonomy, data contracts, observability strategy
   - `tasks.md` for ordered execution steps and Appendix references
   - `specs/src-common-infrastructure/spec.md` for normative requirements + scenarios
   - `openspec/AGENTS.md` for repo-wide operating protocol
3. Inspect canonical modules referenced in this phase:
   - `src/kgfoundry_common/errors/__init__.py` & `exceptions.py`
   - `src/kgfoundry_common/logging.py`
   - `src/kgfoundry_common/observability.py`
   - `src/kgfoundry_common/serialization.py`
   - `src/kgfoundry_common/config.py` & `settings.py`
   - `schema/common/problem_details.json` (new) and related fixtures
4. Capture baseline diagnostics (attach to execution note):
   ```bash
   uv run ruff check src/kgfoundry_common
   uv run pyrefly check src/kgfoundry_common
   uv run mypy src/kgfoundry_common
   python tools/check_imports.py
   uv run pip-audit --strict
   ```
5. Review existing Problem Details examples and structured log formats to ensure alignment (see `docs/examples/`).

## Deliverables Snapshot
- Unified error taxonomy with Problem Details guarantees and JSON fixtures.
- Structured logging adapters delegating to `kgfoundry_common.logging` with correlation IDs and JSON output.
- Observability helpers exposing typed Prometheus metrics and OpenTelemetry hooks, with stub-safe fallbacks.
- Serialization utilities with typed schema validation and catchable exceptions; removal of `Any` networks.
- Typed configuration via `pydantic_settings` with env-only inputs and fast-fail semantics.
- New JSON Schemas under `schema/common/` validating Problem Details and infrastructure payloads.

## Acceptance Gates (run before submission)
```bash
uv run ruff format && uv run ruff check --fix src/kgfoundry_common
uv run pyrefly check src/kgfoundry_common
uv run mypy src/kgfoundry_common
uv run pytest -q tests/kgfoundry_common
python tools/check_imports.py
uv run pip-audit --strict
make artifacts && git diff --exit-code
openspec validate src-hardening-phase-1 --strict
```

## Junior playbook: executing Phase 1

Follow these steps module-by-module. Aim for small, reviewable edits per commit.

1) Problem Details
- Create `schema/common/problem_details.json` and an example in `docs/examples/problem_details/`.
- Implement `src/kgfoundry_common/problem_details.py` with `ProblemDetails`, `build_problem_details`, `problem_from_exception`, `render_problem`.
- Add tests `tests/kgfoundry_common/test_problem_details.py` covering valid/invalid payloads, cause chain via `raise ... from e`.

2) Logging
- Update `src/kgfoundry_common/logging.py` to delegate to structured logger; ensure `NullHandler` is added and `contextvars` propagation.
- Add doctests for `get_logger`, `with_fields`.
- Add tests `tests/kgfoundry_common/test_logging.py` checking JSON shape, correlation ID, feature flag.

3) Observability
- Implement `MetricsProvider` in `observability.py` with stub-safe `.labels()`.
- Add tests for stub vs real registry.

4) Serialization
- Replace `Any` with `JsonValue` aliases; implement `validate_payload` using `jsonschema.Draft202012Validator`.
- Add a small cache for schema documents.
- Tests in `tests/kgfoundry_common/test_serialization.py` (valid/invalid/missing schema).

5) Settings
- Create `settings.py` with `KgFoundrySettings` (pydantic_settings) and `load_settings()`.
- Raise `SettingsError` (subclass of `KgFoundryError`) for missing env and produce Problem Details.
- Doctests for happy/failure paths; tests verifying env behavior.

6) Verify & ship
- Ensure no `print` in libraries; structured logs only.
- Re-run acceptance gates. Fix any drift and commit.

7) API hygiene & imports
- Add `__all__` to all public modules listing exports explicitly; move internals under leading underscore names.
- Audit PEP 257 docstrings: one-line summaries + full sections where public.
- Run `python tools/check_imports.py` and fix any layering violations; update `importlinter.cfg` if needed to include `common-no-upwards` contract.

8) Packaging & doctests
- Build and install: `pip wheel . && python -m venv /tmp/v && /tmp/v/bin/pip install .[obs,schema]`.
- Ensure doctests/xdoctests pass in CI for examples in public modules.

9) Benchmarks (optional but recommended)
- Add pytest-benchmark tests for schema validation + logging formatting; commit baseline numbers.

## Questions & Support
- Error taxonomy & Problem Details → reach out to platform/API owners.
- Observability & metrics → consult SRE team playbooks referenced in `design.md`.
- Settings/configuration → coordinate with security/compliance for env var catalog.

All contributors must log progress via the execution note template described in `tasks.md`.

