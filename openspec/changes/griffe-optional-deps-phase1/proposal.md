## Why
Docs, navmap, and docstring-builder tooling load Griffe and optional AutoAPI/Sphinx
integration modules at runtime. When those dependencies are absent, the code fails with
raw `ImportError` or `ModuleNotFoundError`, skipping structured logs, metrics, and Problem
Details. Users receive no remediation guidance or pointer to install extras. This violates
the observability and error-handling requirements in `AGENTS.md`, and it leaves our CLI
utilities brittle in minimal environments. We need to gate optional imports behind
defensive guards, emit actionable Problem Details with correlation IDs, document
installation extras, and add CLI smoke tests that verify graceful degradation.

## What Changes
- [ ] **Guarded imports** — Wrap Griffe/AutoAPI/Sphinx imports in try/except blocks that
  raise RFC 9457 Problem Details errors with remediation guidance and correlation IDs.
- [ ] **Extras documentation** — Add explicit install instructions (e.g., `pip install
  kgfoundry[docs]`) to CLI help text, README/docs, and module docstrings.
- [ ] **CLI smoke test** — Introduce a test ensuring tooling scripts detect missing
  dependencies, emit structured Problem Details, and exit cleanly.
- [ ] **Docstring updates** — Refresh scripts/docstrings explaining optional dependencies
  and expected environment setup.

## Impact
- **Capability:** `docs-toolchain`
- **Code paths:** `docs/_types/griffe.py`, `docs/_scripts/*.py`, `tools/docstring_builder/**`,
  CLI entry points (`python tools/...`) that import Griffe or optional plugins, related
  tests.
- **Contracts:** Optional dependency absence must yield structured Problem Details with
  correlation IDs and remediation guidance rather than raw exceptions.
- **Delivery:** Implement on branch `openspec/griffe-optional-deps-phase1`, running the full
  quality-gate suite per `AGENTS.md` (Ruff, Pyright, Pyrefly, MyPy, pytest, make artifacts).

## Acceptance
- [ ] Optional imports are fully guarded with try/except; absence triggers Problem Details
  errors containing remediation instructions.
- [ ] CLI smoke test demonstrates graceful degradation and logs/metrics instrumentation.
- [ ] Documentation (docs site, migration guides, docstrings) clearly outlines install
  extras and environment requirements; doctests/pytest confirm guidance.
- [ ] Quality gates pass without suppressions; observability logging includes correlation
  IDs for dependency failures.

## Out of Scope
- Rewriting Griffe stubs or plugin registry logic (covered by other changes).
- Introducing new CLI features beyond dependency handling.

## Risks / Mitigations
- **Risk:** Guarded imports might hide genuine runtime issues.
  - **Mitigation:** Guard only `ImportError`/`ModuleNotFoundError`; re-raise unexpected
    exceptions with preserved tracebacks.
- **Risk:** Documentation can drift from extras definitions.
  - **Mitigation:** Link docs to `pyproject.toml` extras; add CI assertion verifying extras
    exist.
- **Risk:** Smoke test setup complexity.
  - **Mitigation:** Use isolated environments/mocking to simulate missing dependencies
    without heavy installs.

