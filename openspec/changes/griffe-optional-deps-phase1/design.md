## Context
- Tooling modules (`docs/_types/griffe.py`, `docs/_scripts/*.py`, `tools/docstring_builder`) load
  Griffe and optional AutoAPI/Sphinx plugins to generate docs artifacts. Minimal or fresh
  environments often omit these extras, leading to `ModuleNotFoundError` without structured logs
  or remediation guidance. Users must read stack traces to discover missing dependencies.
- Observability requirements in `AGENTS.md` mandate structured logging, metrics, and Problem
  Details with correlation IDs for failure paths. Current handling violates those standards.
- We already maintain Griffe stubs and typed facades; optional dependency handling must integrate
  with them without regressing type safety or docs generation.

## Goals / Non-Goals
- **Goals**
  1. Gate optional imports (Griffe, AutoAPI, Sphinx components) behind well-defined guards that
     raise RFC 9457 Problem Details encouraging installation via documented extras.
  2. Ensure CLI scripts and docs tooling log failures with operation/correlation metadata and
     increment relevant metrics before exiting.
  3. Document the required extras (`[docs]`, `[docstring-builder]`, etc.) in README, CLI help, and
     `pyproject.toml` to ease installation.
  4. Provide automated tests (unit + CLI smoke) verifying graceful degradation when dependencies
     are absent.
  5. Keep typed facades type-clean, using `typing.TYPE_CHECKING` and safe import patterns without
     reintroducing `Any`.
- **Non-Goals**
  - Rewriting Griffe stubs or typed definitions (handled in `griffe-stubs-hardening-phase1`).
  - Refactoring plugin registry or schema alignment logic.
  - Implementing new CLI features beyond dependency detection and guidance.

## Decisions
- Implement `safe_import_griffe` (and similar helper functions) that attempt to import Griffe,
  returning modules when available or raising `OptionalDependencyError` with Problem Details when
  missing. Log via structured logger and increment metrics counters such as
  `kgfoundry_docs_optional_dependency_failures_total`.
- Use `typing.TYPE_CHECKING` blocks and local imports to satisfy Pyright/Pyrefly/Mypy while keeping
  runtime behavior lazy. Provide typed fallback objects (protocols or dataclasses) to maintain API
  compatibility in tests.
- Extend CLI tools to call these helpers at startup; on failure, print serialized Problem Details to
  stderr and exit non-zero. CLI smoke test runs these commands in an environment simulating missing
  dependencies and verifies output.
- Update documentation (docs site, migration guides) with canonical install commands referencing
  extras defined in `pyproject.toml`. Provide doctest-backed snippets demonstrating the helper
  workflow.

## Alternatives Considered
- **Direct `ImportError` messages** — insufficient for observability and actionable guidance.
- **Bundling dependencies by default** — increases footprint; optional extras remain preferable.
- **Environment variables to skip checks** — unnecessary; failure path should always be explicit and
  actionable.

## Risks & Mitigations
- **Risk:** Guard wrappers may mask unrelated runtime errors.
  - **Mitigation:** Catch only `ImportError`/`ModuleNotFoundError`; re-raise unexpected exceptions.
- **Risk:** Extras list may diverge from docs.
  - **Mitigation:** Add tests verifying extras declared in docs exist in `pyproject.toml`.
- **Risk:** CLI smoke tests add CI time.
  - **Mitigation:** Keep tests lightweight via temporary virtualenv/mocking modules.

## Migration Plan
1. Catalog every optional dependency import across docs/tooling modules.
2. Implement guarded import helpers returning typed modules or raising Problem Details errors with
   observability instrumentation.
3. Update CLI scripts and typed facades to use the helpers; ensure type checkers remain clean.
4. Add CLI smoke tests and unit tests covering missing dependency scenarios, verifying error output
   and metrics logging.
5. Update docstrings, README, and docs site to reference install extras; include doctest examples.
6. Run full quality gates (`ruff`, `pyright`, `pyrefly`, `mypy`, `pytest`, `make artifacts`).

