## ADDED Requirements
### Requirement: Public API Hygiene
Public APIs in `kgfoundry_common` SHALL be explicit and documented: exported via `__all__`, named per PEP 8, fully typed, and documented with PEP 257 one-line summaries.

#### Scenario: Public exports declared via __all__
- **GIVEN** a public module under `kgfoundry_common`
- **WHEN** it is imported
- **THEN** the module defines `__all__` enumerating its public symbols and hides internal helpers

#### Scenario: PEP 257 docstrings present
- **GIVEN** a public function or class
- **WHEN** docs are generated
- **THEN** the first line of the docstring is a one-sentence imperative summary and all parameters/returns are documented

### Requirement: Problem Details Canonicalization
Common infrastructure SHALL emit RFC 9457 Problem Details payloads that validate against the repository’s canonical schema.

#### Scenario: Error converts to schema-compliant payload
- **GIVEN** a `KgFoundryError` raised in common infrastructure
- **WHEN** `to_problem_details()` is called
- **THEN** the returned payload validates against `schema/common/problem_details.json`, includes `type`, `title`, `status`, `detail`, `instance`, and preserves extensions

#### Scenario: Schema validation failure raises typed exception
- **GIVEN** an invalid payload passed to the Problem Details builder
- **WHEN** validation runs
- **THEN** a `ProblemDetailsValidationError` is raised and includes a structured log entry with correlation ID

#### Scenario: Exceptions preserve cause chain
- **GIVEN** an underlying exception `e`
- **WHEN** a higher-level `KgFoundryError` is raised
- **THEN** the code uses `raise ... from e` so `__cause__` is preserved and surfaced in logs

### Requirement: Structured Logging Consistency
All logging emitted from `kgfoundry_common` SHALL use the unified structured logger with correlation ID propagation.

#### Scenario: Logger emits JSON with correlation ID
- **GIVEN** a module obtains a logger via `get_logger(__name__)`
- **WHEN** it logs within `with_fields(..., correlation_id="abc")`
- **THEN** the output JSON contains `correlation_id`, `operation`, and `status`, and no plain `print` occurs

#### Scenario: Libraries install NullHandler
- **GIVEN** a library module under `kgfoundry_common`
- **WHEN** it is imported
- **THEN** it attaches a `logging.NullHandler()` to its module logger to avoid duplicate handlers in applications

### Requirement: Layering & Import Boundaries
Common infrastructure SHALL not import higher-level app layers; import-linter contracts SHALL enforce boundaries.

#### Scenario: Import contracts pass
- **GIVEN** `importlinter.cfg`
- **WHEN** the import-linter check runs
- **THEN** contracts preventing upward imports (e.g., from `src/kgfoundry_common` to app layers) are satisfied

### Requirement: Doctests & Examples Execute
Public examples and inline doctests SHALL execute successfully during CI.

#### Scenario: xdoctest runs
- **GIVEN** inline examples in docstrings
- **WHEN** `pytest -q` runs in CI
- **THEN** doctests execute successfully without skipping

### Requirement: Packaging & Distribution
The package SHALL build wheels, install cleanly in a fresh venv, and expose optional extras.

#### Scenario: Wheel builds and installs
- **GIVEN** a clean virtual environment
- **WHEN** `pip wheel .` and `pip install .[obs,schema]` run
- **THEN** both commands succeed and metadata is correct

### Requirement: Security & Supply-Chain
Infrastructure SHALL avoid dangerous constructs, sanitize inputs, and pass vulnerability scans.

#### Scenario: Safe YAML and path handling
- **GIVEN** YAML or filesystem inputs
- **WHEN** they are parsed or resolved
- **THEN** `yaml.safe_load` is used and `Path.resolve(strict=True)` enforces repository confinement

#### Scenario: Dependency audit passes
- **GIVEN** the dependency set
- **WHEN** `pip-audit --strict` runs
- **THEN** no vulnerabilities are reported

#### Scenario: Legacy logging path disabled via feature flag
- **GIVEN** `KGFOUNDRY_LOGGING_V2=0`
- **WHEN** logs are emitted
- **THEN** the legacy adapter is used while emitting a WARN about deprecation, ensuring rollout safety

### Requirement: Observability Metrics Foundation
Common infrastructure SHALL expose typed metrics helpers with Prometheus-compatible counters and histograms, providing safe fallbacks when Prometheus is unavailable.

#### Scenario: Metrics provider works without Prometheus
- **GIVEN** Prometheus is not installed
- **WHEN** `MetricsProvider.default()` is used and `.labels()` is called
- **THEN** the stub implementation returns `self`, allowing code paths to execute without AttributeError

#### Scenario: Metrics provider records duration
- **GIVEN** a successful operation wrapped in `observe_duration`
- **WHEN** the context exits
- **THEN** the histogram records the elapsed time and increments the success counter with structured logging

### Requirement: Typed Serialization & Schema Validation
Serialization utilities SHALL validate payloads against JSON Schema using typed exceptions and caches to avoid untyped `Any` propagation.

#### Scenario: Valid payload passes validation
- **GIVEN** a payload and schema path
- **WHEN** `validate_payload` executes
- **THEN** the function returns `None` and does not raise, caching the schema for reuse

#### Scenario: Invalid payload surfaces typed error
- **GIVEN** an invalid payload
- **WHEN** validation runs
- **THEN** a `SchemaValidationError` is raised with Problem Details and structured log entry

### Requirement: Typed Configuration & Settings
Settings SHALL use `pydantic_settings` models with explicit environment-only configuration and fast failure semantics.

#### Scenario: Missing env var raises SettingsError
- **GIVEN** required environment variables are absent
- **WHEN** `load_settings()` runs
- **THEN** a `SettingsError` is raised, logs an error with correlation ID, and exposes Problem Details JSON

#### Scenario: Settings populate from environment
- **GIVEN** environment variables are set
- **WHEN** `load_settings()` executes
- **THEN** the returned settings object has fully typed fields and no unexpected kwargs are accepted

