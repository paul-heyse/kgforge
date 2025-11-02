# Design

## Context
`tools/docstring_builder/orchestrator.py` grew organically to handle harvesting, caching, plugin dispatch, policy enforcement, docfacts emission, diff generation, manifest creation, and CLI reporting. Each concern modifies shared mutable dictionaries, causing sprawling functions that trip Ruff complexity (C901, PLR091x) and impede comprehension. Typed-pipeline efforts and observability enhancements require clearer seams between pure logic and side effects.

## Goals
- Modularise orchestration into composable, typed helpers capped at manageable complexity.
- Preserve behavioural parity (CLI payloads, manifest outputs, docfacts handling) while enabling future typed pipeline work.
- Ensure docfacts reconciliation remains schema-first, observable, and idempotent.
- Improve testability via deterministic value objects and injected dependencies.

## Non-Goals
- Changing CLI options, exit codes, or output formats.
- Reworking docfacts schema definitions or introducing new persistence layers.
- Introducing asynchronous execution or multi-process coordination.

## Decisions
1. **Pipeline Runner abstraction**  
   Create `PipelineRunner` coordinating context preparation, plugin/policy loading, file processing, docfacts reconciliation, diff/manifest writing, and CLI result assembly. Each step delegates to dedicated helper classes.

2. **File processor encapsulation**  
   Move `_process_file` responsibilities into `FileProcessor` with injected cache, options, plugins, and metrics. Results are typed (`FileOutcome`) with side effects (cache writes, previews) isolated.

3. **Docfacts coordinator**  
   Extract `_handle_docfacts` into `DocfactsCoordinator` exposing `check()` and `update()` methods. Coordinator manages provenance merging, drift detection, and schema validation, returning structured outcomes.

4. **Diff and manifest managers**  
   Introduce `DiffManager` for docstring/docfacts/schema diff files and `ManifestBuilder` for manifest JSON assembly. Both operate on typed inputs and use safe IO helpers.

5. **Failure summary renderer**  
   Replace `_print_failure_summary` with `FailureSummaryRenderer` consuming typed `RunSummary` and `ErrorEnvelope` objects, removing manual coercion and reducing branching.

6. **Value objects & metrics helpers**  
   Define `CommandContext`, `RunTotals`, `ObservabilityPayload`, `ErrorEnvelope`, and `MetricsRecorder` classes to pass structured data between components instead of ad-hoc dicts.

## Risks / Trade-offs
- **Risk**: Too many helper classes increase mental overhead.  
  Mitigation: Keep class APIs small, align naming with responsibilities, and document with concise docstrings.
- **Risk**: Regression in docfacts drift detection.  
  Mitigation: Add unit tests covering provenance override logic and diff generation; capture before/after payloads in fixtures.
- **Trade-off**: Additional modules vs single-file simplicity.  
  Benefit: Improved clarity, test coverage, and lint compliance outweigh minimal import overhead.

## Migration
1. Land helper modules with unit tests while keeping orchestrator entry point intact.
2. Incrementally replace inline logic in `orchestrator.py` with helper invocations, verifying tests at each step.
3. Remove legacy functions once parity is confirmed; run full quality gates.
4. Update docs/artifacts if generated outputs change; capture CLI regression evidence.
5. Archive change after validation and ensure no open tasks remain.

