## ADDED Requirements
### Requirement: Typed Docstring Builder Contracts
The docstring builder SHALL operate on typed intermediate representations that validate against the DocFacts JSON Schema and versioned CLI output schemas before emitting artifacts.

#### Scenario: DocFacts schema validation succeeds
- **GIVEN** `docs/_build/schema_docfacts.json` and a builder run with valid harvested symbols
- **WHEN** the builder materializes DocFacts and docstring payloads
- **THEN** each payload validates against the schema and the builder records a success metric

#### Scenario: DocFacts schema mismatch is reported
- **GIVEN** a mismatch between the builder `DOCFACTS_VERSION` constant and the schema metadata
- **WHEN** validation runs
- **THEN** the builder raises `SchemaViolationError` with an RFC 9457 Problem Details payload and aborts the run

#### Scenario: CLI JSON output validates
- **GIVEN** `docstring_builder --json` or `--baseline` execution
- **WHEN** machine-readable output is produced
- **THEN** the payload validates against `schema/tools/docstring_builder_cli.json` and includes the declared schema version

### Requirement: Plugin Protocol Reliability
Docstring builder plugins SHALL implement a typed `Protocol` contract, surface structured errors, and avoid untyped `Any` propagation.

#### Scenario: Bundled plugins conform to Protocol
- **GIVEN** the built-in plugins (`dataclass_fields`, `llm_summary`, `normalize_numpy_params`, others)
- **WHEN** mypy and pyrefly run in strict mode
- **THEN** no plugin contributes unchecked `Any`, and Protocol conformance is enforced via tests

#### Scenario: Plugin failure surfaces structured error
- **GIVEN** a plugin raises an internal error
- **WHEN** the builder executes the plugin hook
- **THEN** the failure is wrapped in `PluginExecutionError`, logged with structured context, and reported in CLI Problem Details output

#### Scenario: Dataclass variance regression is covered
- **GIVEN** a dataclass with `kw_only` fields and custom metadata
- **WHEN** the dataclass_fields plugin runs
- **THEN** the generated parameter descriptions remain stable across runs, and regression tests guard against variance bugs

#### Scenario: Legacy plugin compatibility shim warns
- **GIVEN** a third-party plugin still using the legacy `(symbol, ir)` signature
- **WHEN** the builder executes the plugin with the compatibility flag enabled
- **THEN** the shim adapts the call, emits a single `DeprecationWarning`, and records the plugin result envelope with schema-compliant metadata

### Requirement: CLI Observability and Safety
The docstring builder CLI SHALL emit structured logs/metrics, route subprocess work through shared secure utilities, and enable Jinja autoescape during rendering.

#### Scenario: Structured logging replaces prints
- **GIVEN** any CLI invocation
- **WHEN** the builder runs in normal or error conditions
- **THEN** logs are emitted through `tools._shared.logging.get_logger`, include correlation IDs, and no direct `print` statements remain in library code

#### Scenario: Subprocesses executed safely
- **GIVEN** the CLI must spawn helper commands
- **WHEN** a subprocess runs
- **THEN** it flows through `tools._shared.proc.run_tool`, enforcing absolute executable paths, sanitized environment variables, and timeout handling

#### Scenario: Rendering autoescape enabled
- **GIVEN** the renderer processes docstrings containing HTML-special characters
- **WHEN** templates render output
- **THEN** autoescape is enabled (or configured via safe allowlist), preventing unescaped HTML injection while maintaining expected formatting

#### Scenario: Feature flag toggles typed pipeline safely
- **GIVEN** the environment variable `DOCSTRINGS_TYPED_IR=0`
- **WHEN** the CLI runs
- **THEN** the legacy pipeline executes while emitting a WARN log indicating the fallback, and schema validation runs in dry-run mode without blocking execution

#### Scenario: CLI errors emit Problem Details
- **GIVEN** a builder run encounters a schema violation or plugin failure
- **WHEN** the CLI exits with `--json`
- **THEN** the output includes a top-level `problem` object conforming to RFC 9457 (and per-file problem entries where applicable), matching the example captured in `tools/docstring_builder/models.py`

## MODIFIED Requirements
<!-- None -->

## REMOVED Requirements
<!-- None -->

## RENAMED Requirements
<!-- None -->

