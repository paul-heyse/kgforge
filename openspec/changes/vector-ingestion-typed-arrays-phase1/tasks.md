## 1. Implementation

- [ ] 1.1 Capture current ingestion gaps.
  - Inventory all `# type: ignore` directives and numpy usages under `orchestration`, `search_api`, and `kgfoundry_common` related to vector ingestion.
  - Record failing pyrefly/pyright diagnostics as a baseline snapshot.
- [ ] 1.2 Implement shared vector contract module.
  - Add `src/kgfoundry_common/vector_types.py` with `VectorId`, `VectorMatrix`, `VectorBatch`, validation helpers, and `VectorValidationError`.
  - Create matching stubs under `stubs/kgfoundry_common/vector_types.pyi`.
- [ ] 1.3 Replace legacy hints and suppressions.
  - Update `orchestration.cli` and `search_api.vectorstore_factory` to import the shared contracts, remove suppressions, and use helper functions.
  - Ensure imports respect `tools/check_imports.py` boundaries.
- [ ] 1.4 Integrate schema validation.
  - Author `schema/vector-ingestion/vector-batch.v1.schema.json` (Draft 2020-12) with examples.
  - Update loaders to validate JSON payloads prior to coercion and map failures to `VectorValidationError`.
- [ ] 1.5 Add Problem Details support.
  - Create `schema/examples/problem_details/vector-ingestion-invalid-vector.json`.
  - Introduce helper to build ingestion Problem Details, wrap exceptions in `IndexBuildError`, and emit structured logs.
- [ ] 1.6 Enhance factory telemetry.
  - Ensure `FaissVectorstoreFactory` logs vector counts/dimensions and records Prometheus metrics tagged with `stage="ingestion"`.
  - Surface correlation IDs through CLI and factory layers.
- [ ] 1.7 Update documentation and navigation metadata.
  - Add reference page (`docs/reference/vector-ingestion.md`) describing the new contract, schema, and Problem Details.
  - Update relevant `__navmap__` entries to list new exports.
- [ ] 1.8 Remove deprecated helpers.
  - Identify and deprecate any legacy vector parsing utilities, replacing with shims that delegate to the new module while emitting `DeprecationWarning`.

## 2. Testing

- [ ] 2.1 Establish fixtures and validators.
  - Create `tests/vector_ingestion/conftest.py` providing canonical payloads, schema validator fixture, and deterministic correlation IDs.
- [ ] 2.2 Unit test validation helpers.
  - Add table-driven tests for `coerce_vector_batch`, `validate_vector_batch`, and `assert_vector_matrix`, covering success and failure cases.
- [ ] 2.3 Schema round-trip coverage.
  - Validate representative payloads (small, large, edge cases) against the schema and ensure malformed payloads fail with descriptive paths.
- [ ] 2.4 CLI integration tests.
  - Use `typer.testing.CliRunner` to assert success path produces typed batches and failure path emits canonical Problem Details with exit code `1`.
- [ ] 2.5 Telemetry verification.
  - Inspect Prometheus registry state after ingestion success/failure to confirm counters/histograms tagged appropriately.
- [ ] 2.6 Doctest/xdoctest enforcement.
  - Run doctests for new module docstrings and ensure examples in docs execute.
- [ ] 2.7 Regression suite gating.
  - Execute `uv run pytest -q tests/vector_ingestion tests/orchestration/test_index_faiss.py` and capture results for the PR template.

## 3. Docs & Artifacts

- [ ] 3.1 Regenerate schema and doc artifacts.
  - Run `make artifacts` and ensure schema indices, nav maps, and docs catalog include the new entries with clean diff.
- [ ] 3.2 Update operator runbooks.
  - Document remediation guidance for ingestion failures, including Problem Details interpretation.
- [ ] 3.3 Publish migration notes.
  - Add notes to release/migration docs clarifying stricter validation and how to dry-run datasets.

## 4. Rollout

- [ ] 4.1 Validate staging pipelines.
  - Execute ingestion flows against staging datasets with the new contract, monitoring for failures or performance regressions.
- [ ] 4.2 Coordinate downstream adoption.
  - Notify teams consuming vector ingestion outputs, offering validation scripts and timeline.
- [ ] 4.3 Final quality gates.
  - Record outputs of `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pytest -q`, `make artifacts && git diff --exit-code`, `openspec validate vector-ingestion-typed-arrays-phase1 --strict` in the PR checklist.
- [ ] 4.4 Production enablement.
  - Roll out to production ingestion jobs with monitoring on Problem Details count and ingestion latency; define rollback plan in case of regression.


