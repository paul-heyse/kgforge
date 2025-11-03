## Why
Logging and tooling contexts such as `LogContextExtra`, `ErrorReport`, and `CliResult`
currently rely on partially defined `TypedDict` structures that treat optional keys as
required. Tests and runtime code access these keys directly, triggering Pyright
`reportTypedDictNotRequiredAccess` errors and masking missing-field defects. We need to
formalize these contexts with explicit required/optional partitions (or frozen
dataclasses), update call sites to use safe accessors, and guarantee logging APIs emit
Problem Details payloads with correlation IDs per our observability standards.

## What Changes
- [ ] **MODIFIED**: Replace loosely defined `TypedDict` contexts with frozen dataclasses or
  TypedDicts that declare `Required[...]` and `NotRequired[...]` fields.
- [ ] **MODIFIED**: Update logging helpers, docstring-builder utilities, and CLI plumbing
  to use safe accessor helpers (`.get`, wrapper functions) rather than direct indexing.
- [ ] **ADDED**: Ensure logging APIs emit structured Problem Details JSON, including
  correlation IDs and operation metadata, with example payloads checked into
  `schema/examples/problem_details/`.
- [ ] **ADDED**: Expand pytest suites to cover required/optional key handling, accessor
  behavior, and Problem Details emission.
- [ ] **MODIFIED**: Refresh documentation explaining the logging context dataclasses and
  accessor patterns.

## Impact
- **Capability:** `tools-suite`
- **Code paths:** `kgfoundry_common/logging.py`, `tools/docstring_builder/pipeline*.py`,
  `tests/test_logging.py`, `tests/docstring_builder/**`, and any module consuming
  `LogContextExtra`, `ErrorReport`, or `CliResult`.
- **Data contracts:** Problem Details examples augmented to include correlation IDs and
  structured logging metadata; logging context schemas effectively become frozen data
  structures.
- **Delivery:** Implement under branch `openspec/typed-dict-hygiene-phase1` with full
  quality gates (`ruff`, `pyright`, `pyrefly`, `mypy`, `pytest`, `make artifacts`).

## Acceptance
- [ ] `LogContextExtra`, `ErrorReport`, and `CliResult` are represented via frozen
  dataclasses or TypedDicts with explicit required/optional keys; runtime code uses safe
  accessors with zero `reportTypedDictNotRequiredAccess` findings.
- [ ] Logging APIs emit Problem Details JSON examples with correlation IDs; schema examples
  updated and referenced in documentation.
- [ ] Tests cover success/error contexts, legacy behavior, and Problem Details emission,
  running clean under pytest/doctest.
- [ ] Ruff, Pyright, Pyrefly, and MyPy report zero errors without suppressions.

## Out of Scope
- Changing the overall logging API shape or introducing new telemetry backends.
- Refactoring unrelated docstring-builder plugin registries or schema alignment work.
- Implementing new CLI features beyond updating context handling.

## Risks / Mitigations
- **Risk:** Freezing contexts may break callers relying on mutation.
  - **Mitigation:** Provide helper methods that return new instances with updated fields
    and document usage; add tests covering mutation scenarios.
- **Risk:** Problem Details updates may require additional schema maintenance.
  - **Mitigation:** Version examples, validate via `make artifacts`, and coordinate with
    docs owners.
- **Risk:** Legacy call sites may still index optional keys.
  - **Mitigation:** Add accessor helpers and regression tests; use static analysis to ensure
    direct indexing is eliminated.

