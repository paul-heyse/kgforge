## ADDED Requirements
### Requirement: Shared Doc Toolchain Lifecycle Helpers
Doc toolchain commands SHALL use shared lifecycle helpers and context objects to manage configuration, logging, metrics, and Problem Details consistently across symbol index builds, delta generation, and artifact validation.

#### Scenario: Symbol index build uses lifecycle context
- **GIVEN** `docs/toolchain/build_symbol_index.py` runs
- **WHEN** it processes CLI arguments
- **THEN** it constructs a `DocToolContext` via the shared lifecycle module, emits structured start/stop logs, and records Prometheus metrics through `DocMetrics`

#### Scenario: Symbol delta generation emits Problem Details consistently
- **GIVEN** `docs/toolchain/symbol_delta.py` encounters an error (e.g., missing baseline)
- **WHEN** the shared lifecycle captures the failure
- **THEN** it emits a Problem Details payload via `DocLifecycle.run`, logs with `status="error"`, and increments failure metrics

#### Scenario: Artifact validation logs structured telemetry
- **GIVEN** `docs/toolchain/validate_artifacts.py` validates outputs
- **WHEN** validations succeed or fail
- **THEN** the command records structured logs with correlation IDs, updates metrics via `DocMetrics`, and respects lifecycle-provided Problem Details on failure

#### Scenario: Static analysis remains clean
- **GIVEN** Ruff, Pyright, and Pyrefly run over doc toolchain modules
- **WHEN** lifecycle helpers are in place
- **THEN** analyzers report zero errors related to logging, Problem Details, or metrics usage
