# Design

## Context

The doc toolchain scripts (`build_symbol_index.py`, `symbol_delta.py`, `validate_artifacts.py`) grew organically: each script handles CLI args, configuration, logging, metrics, and error reporting differently. Some call `print`, others manually build Problem Details, and metrics hooks are inconsistent. This makes observability fragile and complicates reuse. We want to refactor these entrypoints to share context objects and lifecycle helpers similar to the CLI tooling changes—producing uniform logging, metrics, and Problem Details output while keeping behaviour stable.

> **Note:** Unit tests and documentation updates are intentionally deferred to a later change per scope direction; this plan focuses exclusively on design and implementation.

## Goals

- Introduce small context objects (`DocToolContext`, `DocBuildLifecycle`) that encapsulate input parameters, logging adapters, metrics, and Problem Details helpers.
- Refactor the three doc toolchain scripts to construct contexts and delegate core logic to lifecycle helpers.
- Ensure structured logging (operation, status, artifact info), Problem Details emission, and Prometheus metrics are consistent across tasks.

## Non-Goals

- Adding or expanding unit tests (handled in follow-up work).
- Modifying existing data formats (symbol index JSON, delta payloads, validation outputs) beyond changing logging/observability.

## Decisions

1. **Shared lifecycle module** — add `docs/toolchain/_shared/lifecycle.py` (or similar) exporting:
   - `DocToolSettings` dataclass for CLI inputs (paths, concurrency, flags).
   - `DocToolContext` dataclass bundling settings, logger, metrics hooks, and Problem Details helpers.
   - `DocLifecycle` helper with methods `start(operation)`, `success(summary)`, `failure(problem_details)` to manage logging/metrics.
2. **Structured logging** — use `tools.get_logger` or `kgfoundry_common.logging` APIs to generate structured logs with fields `operation`, `status`, `artifact`, `duration`, and correlation IDs.
3. **Problem Details** — centralize Problem Details building functions (`build_doc_problem_details`) in the shared module; errors in scripts call `context.problem_details(...)`.
4. **Metrics** — register Prometheus counters/histograms (`kgfoundry_docs_operation_total`, `kgfoundry_docs_operation_duration_seconds`) with labels (`operation`, `status`). The shared module provides `metrics.observe_success/failure` hooks.
5. **Configuration** — ensure each script obtains configuration via `DocToolSettings.from_cli(argv)` to avoid per-script logic duplication.

## Detailed Plan

### 1. Shared lifecycle module

1. Create `docs/toolchain/_shared/lifecycle.py` with:
   - `DocToolSettings` capturing CLI inputs (e.g., input path, output path, delta path, concurrency). Provide classmethod `parse_args(argv, *, description, operation)` to standardize CLI parsing.
   - `DocToolContext` dataclass containing settings, `logger: logging.LoggerAdapter`, `metrics: DocMetrics`, and `problem_details_builder`. Include helper methods to log structured messages (`info`, `warning`, `error`) while injecting operation metadata.
   - `DocMetrics` class registering counters/histograms on first use; expose `observe_start()`, `observe_success(duration)`, `observe_failure(status)` methods.
   - `DocLifecycle` context manager method `run(operation, work)` which logs start, measures `time.monotonic()`, executes `work(context)`, and handles exceptions by emitting Problem Details, logging, and re-raising or returning exit codes.
   - Problem Details helpers: `build_doc_problem_details(type, title, detail, instance, status, extra)` returning dict conforming to our taxonomy.
2. Provide references in docstrings for each helper; include example usage and instructions for existing scripts.

### 2. Refactor build_symbol_index.py

1. Replace existing CLI parsing with `DocToolSettings.parse_args(argv, description=..., operation="build_symbol_index")` to obtain settings.
2. Instantiate `DocToolContext` via `create_doc_tool_context(settings)` exported from shared module.
3. Wrap the main workflow in `DocLifecycle.run` to manage logging/metrics.
4. Within the `work` function, keep existing logic for reading docs, building symbol index, and writing output; replace `print` statements with `context.logger.info` and error paths with `context.raise_problem(detail, status=500)`.
5. Ensure success/failure flows call the metrics hooks and Problem Details builder.

### 3. Refactor symbol_delta.py

1. Apply the same pattern: parse CLI args via shared settings, construct context, and wrap core logic in `DocLifecycle.run`.
2. Replace direct logging with context logging methods; ensure metrics capture durations and statuses.
3. Ensure Problem Details emitter handles cases such as missing baseline files, invalid diff payloads, etc.

### 4. Refactor validate_artifacts.py

1. Use shared settings/context/lifecycle; adjust logic to iterate validations while leveraging context logging.
2. Convert validation errors to Problem Details with specific `type` values (e.g., `docs-artifact-validation`) and ensure metrics capture counts.

### 5. Observability consistency

1. Ensure each script exposes correlation ID (generate via `uuid.uuid4()` inside context) so logs can be aggregated.
2. Provide uniform log structure across scripts, e.g., `logger.info("artifact validated", extra={"artifact": settings.output_path})`.
3. Confirm metrics register only once (guard via module-level registry functions).

## Risks & Mitigations

- **Backwards compatibility** — capturing success/failure may need to align with existing CLI exit codes; ensure new lifecycle retains exit codes.
- **Circular imports** — keep shared lifecycle free of script-specific imports; scripts import from shared module but not vice versa.

## Migration

1. Implement shared lifecycle module (with docstrings, Problem Details helpers, metrics registration).
2. Refactor `build_symbol_index.py` to use the new context, ensuring behaviour matches current output.
3. Refactor `symbol_delta.py` similarly; verify deltas remain identical.
4. Refactor `validate_artifacts.py`; ensure validations still output expected results.
5. Execute lint/type checks and CLI smoke runs (`python -m docs.toolchain.build_symbol_index ...`, etc.) to confirm functionality.
