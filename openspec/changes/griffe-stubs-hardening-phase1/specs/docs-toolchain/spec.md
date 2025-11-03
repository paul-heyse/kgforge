## ADDED Requirements
### Requirement: Typed Griffe Loader Contracts
The documentation toolchain SHALL expose Griffe loader functionality through precise
stub definitions and typed facades that eliminate `Any`, mirror upstream signatures, and
degrade gracefully when optional plugins are absent. The contract applies to
`stubs/griffe/__init__.pyi`, `stubs/griffe/loader/__init__.pyi`, and
`docs/_types/griffe.py`.

#### Scenario: Loader overload parity
- **GIVEN** the hardened stubs and an environment with Griffe installed
- **WHEN** a parity test introspects `griffe.load`, `load_module`, and related helpers
- **THEN** the exported overloads match the runtime callable signatures (parameters,
  defaults, return annotations) and Pyright/Pyrefly detect no `Any` leakage

#### Scenario: Optional plugin guard
- **GIVEN** a runtime lacking AutoAPI or other optional plugins
- **WHEN** `docs._types.griffe.get_autoapi_loader()` executes
- **THEN** it raises a descriptive `ArtifactDependencyError` (or equivalent), preserves
  the original `ImportError` via `raise ... from e`, and emits an RFC 9457 Problem
  Details payload captured in tests

#### Scenario: Typed facade usage example passes doctest
- **GIVEN** doctest execution over `docs/_types/griffe.py`
- **WHEN** the embedded example constructs a loader via the typed facade and inspects a
  module summary
- **THEN** the doctest passes without additional setup, demonstrating typed return
  values and documenting optional plugin handling

#### Scenario: Upstream alignment check
- **GIVEN** the parity test suite running under continuous integration
- **WHEN** Griffe upstream updates its API surface
- **THEN** the tests assert the stub export list matches runtime attributes, failing fast
  with actionable messaging so maintainers adjust stubs and upstream contribution plan

