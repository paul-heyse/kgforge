## ADDED Requirements
### Requirement: Modular Docstring Builder Orchestration
The system SHALL execute docstring builder runs via modular components that separate file processing, docfacts reconciliation, diff/manifest management, and CLI reporting, each with bounded complexity and typed interfaces.

#### Scenario: Pipeline runner composes helpers
- **GIVEN** a docstring builder request invoking `run_docstring_builder`
- **WHEN** the pipeline executes with mock helpers for file processing, docfacts coordination, diff management, and manifest writing
- **THEN** the runner delegates to each helper exactly once, aggregates outcomes into a typed `DocstringBuildResult`, and all helper methods remain below Ruff complexity thresholds (C901, PLR091x)

#### Scenario: File processor enforces cache semantics
- **GIVEN** a cached file with matching config hash and a `FileProcessor`
- **WHEN** the processor runs in update mode
- **THEN** it returns a `FileOutcome` marked `skipped=True`, increments cache hit metrics, and avoids calling harvest/render routines

#### Scenario: Policy and plugin hooks remain injectable
- **GIVEN** stubbed `PolicyEngine` and `PluginManager`
- **WHEN** the pipeline processes files
- **THEN** plugin hooks (`apply_harvest`, `apply_transformers`, `apply_formatters`) and policy recording/finalization are invoked via dependency injection without direct imports from the orchestrator module

### Requirement: Reliable Docfacts Coordination
The system SHALL reconcile DocFacts artifacts through a dedicated coordinator that validates payloads, preserves provenance, and emits drift previews deterministically.

#### Scenario: Check mode detects drift
- **GIVEN** existing DocFacts payloads differing from freshly generated entries
- **WHEN** `DocfactsCoordinator.check()` runs
- **THEN** it validates the stored payload, normalizes provenance fields, writes an HTML diff to `DOCFACTS_DIFF_PATH`, logs a structured violation, and returns `ExitStatus.VIOLATION`

#### Scenario: Update mode validates output
- **GIVEN** new DocFacts entries and typed pipeline enabled
- **WHEN** `DocfactsCoordinator.update()` runs
- **THEN** it writes the payload atomically, validates it against the schema, clears stale diff files, and returns `ExitStatus.SUCCESS`

#### Scenario: Baseline reconciliation emits drift entries
- **GIVEN** baseline docfacts text differing from current payload
- **WHEN** the coordinator executes within a baseline comparison run
- **THEN** it triggers `DiffManager` to write an HTML comparison and records the relative diff path in the run manifest

### Requirement: Structured Failure Summaries
The system SHALL render failure summaries using typed run summaries and error envelopes, producing structured logs without manual dictionary coercion.

#### Scenario: Non-success run logs summary
- **GIVEN** a `DocstringBuildResult` with `ExitStatus.VIOLATION`
- **WHEN** `FailureSummaryRenderer.render()` executes
- **THEN** it logs a summary including considered/processed/changed counts, status counts, observability path, and top errors formatted as list entries without raising Ruff complexity warnings

#### Scenario: Success run emits no summary
- **GIVEN** a `DocstringBuildResult` with `ExitStatus.SUCCESS`
- **WHEN** `FailureSummaryRenderer.render()` executes
- **THEN** it performs no logging beyond baseline debug statements, ensuring idempotent behaviour

### Requirement: Observability and Metrics Integrity
The system SHALL emit structured logs and Prometheus metrics via dedicated helpers, ensuring correlation identifiers and status labels propagate consistently.

#### Scenario: Metrics recorder tracks durations
- **GIVEN** a completed pipeline run
- **WHEN** `MetricsRecorder.observe_cli_duration()` executes
- **THEN** it records the duration in the histogram, increments the status counter, and labels include command and status without duplication

#### Scenario: Structured logs include correlation IDs
- **GIVEN** `PipelineRunner` executing with a correlation ID context
- **WHEN** helpers log status transitions
- **THEN** each log entry contains `correlation_id`, `command`, and `subcommand` fields per logging policy while avoiding `print` statements

