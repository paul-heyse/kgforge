## ADDED Requirements

### Requirement: Schema-Validated Typed Docs Pipeline
The documentation toolchain SHALL generate symbol indices, deltas, and API content through typed helpers that
enforce JSON Schema 2020-12 contracts, structured logging, and strict static analysis (Ruff, Pyrefly, mypy)
before artifacts are published.

#### Scenario: Symbol index generation succeeds
- **GIVEN** typed settings from `docs._scripts.shared` and a Griffe loader instantiated via the shared factory
- **WHEN** `docs/_scripts/build_symbol_index.py` runs during `make artifacts`
- **THEN** the script validates the resulting payload against `schema/docs/symbol-index.schema.json`
- **AND** emits structured logs with `operation` and `artifact` metadata
- **AND** Ruff, Pyrefly, and mypy checks for `docs/` report zero errors

#### Scenario: Delta computation emits problem details on failure
- **GIVEN** a missing or corrupt historical snapshot when `docs/_scripts/symbol_delta.py` executes
- **WHEN** the script fails to load or validate the base snapshot
- **THEN** it raises a `ToolExecutionError` (or derivative) containing an RFC 9457 Problem Details payload
- **AND** the failure is logged with structured context, including operation and artifact identifiers

#### Scenario: Artifact validation gate prevents drift
- **GIVEN** freshly generated docs artifacts in `docs/_build`
- **WHEN** `docs/_scripts/validate_artifacts.py` runs as part of `make artifacts`
- **THEN** it validates each payload against the corresponding schema under `schema/docs/`
- **AND** exits non-zero with an RFC 9457 Problem Details payload if any artifact deviates from the contract

#### Scenario: Observability metadata links docs builds across systems
- **GIVEN** docs scripts executed with default logging/metrics configuration
- **WHEN** symbol indexing or delta computation completes (success or failure)
- **THEN** emitted logs include `operation`, `artifact`, and correlation identifiers
- **AND** metrics/tracing hooks publish duration/status data via `tools._shared.metrics.observe_tool_run`

