## 0. Getting Started
- [ ] (0.1) Read `proposal.md`, `design.md` (Implementation Blueprint, phase plan, Testing Matrix), and `specs/quality-hardening/spec.md` end-to-end.
- [ ] (0.2) Bootstrap the environment: `uv sync` → `uv run ruff check src` → `uv run mypy src` to note existing failures.
- [ ] (0.3) Create branch `openspec/lint-type-hardening`; keep commits scoped to individual requirements or automation helpers.
- [ ] (0.4) Configure local tooling (editor/CI hooks) to run `ruff`, `pyrefly`, `mypy`, `pytest`, and doctests on save or pre-commit.
- [ ] (0.5) Review the Clean Baseline definition in `README.md`. Every requirement/PR must leave the tree in that state before requesting review.

---

## Phase 1 — Foundation (Helpers, Taxonomy, Tooling)
> Land shared infrastructure before touching product call sites. Attach codemod logs, helper unit test outputs, and command transcripts to your PR.

### F1 — Pathlib Helpers & Codemod (R1)
- [ ] (F1.1) Implement `src/kgfoundry_common/fs.py` helpers (`ensure_dir`, `safe_join`, `read_text`, `write_text`, `atomic_write`) with docstrings + full typing.
- [ ] (F1.2) Author codemod script `tools/codemods/pathlib_fix.py` (LibCST/Bowler) handling `os.path.join`, `os.makedirs`, bare `open()` patterns.
- [ ] (F1.3) Run codemod: `python -m tools.codemods.pathlib_fix src` and save log (include in PR). Follow with `uv run ruff check --fix` to clean up.
- [ ] (F1.4) Add unit tests `tests/unit/test_fs_helpers.py` covering success/permission/traversal via `pytest.mark.parametrize`.

### F2 — Exception Taxonomy & Problem Details Registry (R2)
- [ ] (F2.1) Create `kgfoundry_common/errors/codes.py` enumerating stable error codes + type URIs.
- [ ] (F2.2) Implement `KgFoundryError` hierarchy referencing the registry and exposing `to_problem_details()` helper with checksum of context.
- [ ] (F2.3) Build codemod `tools/codemods/blind_except_fix.py` to rewrite `except Exception:` blocks into typed scaffolds (leaving TODO where mapping unclear).
- [ ] (F2.4) Run codemod, review diffs, and ensure each change either maps to taxonomy or has a TODO + ticket for follow-up.
- [ ] (F2.5) Implement HTTP/CLI adapters consuming the helper; add tests `tests/unit/test_errors.py` and `tests/search_api/test_problem_details_http.py` verifying payload, `type` URI, `code`, and `raise ... from exc` preservation.

### F3 — Safe Serialization Facade (R3)
- [ ] (F3.1) Introduce `kgfoundry_common/serialization.py` with `serialize_json(obj, schema_path)`/`deserialize_json(data, schema_path)` performing schema validation + SHA256 checksum.
- [ ] (F3.2) Add tests `tests/unit/test_serialization_helpers.py` covering happy/edge/failure cases.
- [ ] (F3.3) Prepare usage guide in docstrings for downstream modules.

### F4 — Schema & Model Scaffold (R4)
- [ ] (F4.1) Create/refresh schemas with `$id`, `x-version`, `x-compatibility-notes`, and examples under `schema/examples/**`.
- [ ] (F4.2) Implement round-trip helper `assert_model_roundtrip(model_cls, example_path)` and use it in unit tests for each model.
- [ ] (F4.3) Wire acceptance commands (`jsonschema validate`, `spectral lint`) into CI locally.

### F5 — Vector Protocols & Stubs (R5)
- [ ] (F5.1) Implement `kgfoundry/search_api/types.py` protocols + TypedDict/dataclass results; provide docstrings referencing performance/concurrency expectations.
- [ ] (F5.2) Add/refresh stubs in `stubs/` for FAISS, libcuvs, pyserini.
- [ ] (F5.3) Ensure namespace consolidation: expose only `kgfoundry.vectorstore_faiss` (deprecate/remove duplicate top-level package).

### F6 — Runtime Settings & Observability Envelope (R7 & R8)
- [ ] (F6.1) Implement `RuntimeSettings(BaseSettings)` with nested models, fail-fast Problem Details, and docstrings listing env vars.
- [ ] (F6.2) Provide `get_logger(__name__)` LoggerAdapter injecting `correlation_id`, `operation`, `status`, `duration_ms`.
- [ ] (F6.3) Add Prometheus metrics/tracing helpers; create tests (`tests/observability/test_logging_adapter.py`, `test_metrics_helpers.py`).

### F7 — Tooling & Automation (R10, R17)
- [ ] (F7.1) Update `mypy.ini` to enable `plugins = pydantic.mypy, sqlalchemy.ext.mypy.plugin`; balance strictness for tests (typed defs, relaxed calls).
- [ ] (F7.2) Update `pyproject.toml` `[tool.ruff.lint.flake8-type-checking]` with `strict = true`; ensure rule families `D, ANN, S, PTH, TRY, ARG, DTZ` enforced.
- [ ] (F7.3) Add import-linter config `importlinter.cfg` capturing domain→adapter→I/O contracts.
- [ ] (F7.4) Implement `tools/check_new_suppressions.py` failing on new `# type: ignore`/`noqa` without `TICKET:`.
- [ ] (F7.5) Create CI step writing `$GITHUB_STEP_SUMMARY` with links to coverage, docs, Agent Portal, schema lint report, build artifacts.

### F8 — Packaging/GPU Extras Prep (R14)
- [ ] (F8.1) Update `pyproject.toml` extras to isolate GPU deps (`[project.optional-dependencies] gpu = [...]`).
- [ ] (F8.2) Document import path consolidation and extras usage in README/CHANGELOG.

> **Phase 1 exit criteria:** All helpers/codemods/tooling merged, tests green, clean baseline achieved, exception codes frozen, namespace consolidated, GPU extras defined.

---

## Phase 2 — Adoption (Requirement-by-Requirement Migration)
> Migrate product surfaces to the new helpers in clusters (search adapters → registry → embeddings → orchestration → remainder). Execute requirements sequentially, running acceptance gates and attaching outputs after each requirement.

### R1 — Pathlib Standardization Across Workflows (Module Adoption)
- [ ] (R1.1) Run pathlib codemod for target cluster, review diff, commit log.
- [ ] (R1.2) Replace remaining filesystem calls manually where codemod cannot infer intent; ensure docstrings + typing added.
- [ ] (R1.3) Update relevant unit/integration tests to assert new helpers are used.

### R2 — Exception Taxonomy & Problem Details Adoption
- [ ] (R2.1) Apply blind-except codemod to target cluster, mapping to taxonomy.
- [ ] (R2.2) Ensure Problem Details responses include `type` URI and `code` from registry; update docs/examples.
- [ ] (R2.3) Write regression tests verifying payload structure, logging, and `raise ... from exc` behavior.

### R3 — Secure Serialization & Persistence Adoption
- [ ] (R3.1) Replace legacy serialization with new helpers; remove direct `pickle` usages.
- [ ] (R3.2) Parameterize SQL queries; add checksum/schema validation to loaders.
- [ ] (R3.3) Extend tests (parametrized) for corrupted payloads and injection attempts.

### R4 — Typed JSON Contracts Adoption
- [ ] (R4.1) Migrate modules to new Pydantic models; remove dict manipulations.
- [ ] (R4.2) Add round-trip tests referencing example JSON; ensure doctests import examples.

### R5 — Vector Protocol Compliance Adoption
- [ ] (R5.1) Update adapters to protocols; remove `Any`/ignores.
- [ ] (R5.2) Add tests verifying protocol satisfaction + result typing; ensure async/gpu paths adhere.

### R6 — Parquet IO Type Safety Adoption
- [ ] (R6.1) Replace ad-hoc Parquet helpers with typed ones; enforce schema validation.
- [ ] (R6.2) Extend tests for mismatched schema/invalid data.

### R7 — Typed Configuration Adoption
- [ ] (R7.1) Swap module-level config dicts for `RuntimeSettings` injection; update docs with env var tables.
- [ ] (R7.2) Extend tests verifying missing env Problem Details + fail-fast behavior.

### R8 — Structured Observability Envelope Adoption
- [ ] (R8.1) Replace logging calls with `get_logger(__name__)`; ensure metrics/traces added.
- [ ] (R8.2) Add tests verifying required log fields and metric increments on failure.

### R9 — Public API Hygiene & Documentation
- [ ] (R9.1) Sort `__all__`, remove dynamic registries, add docstrings referencing schemas/Problem Details.
- [ ] (R9.2) Update doctests/examples; run doctest suite.

### R10 — Tooling Enforcement (Adoption)
- [ ] (R10.1) Ensure no new suppressions; update docs/PR template to list mandatory commands and Problem Details example location.
- [ ] (R10.2) Verify import-linter contracts for each cluster before merge.

### R11 — Concurrency & Context Propagation
- [ ] (R11.1) Introduce `ContextVar` usage in async flows; move blocking IO to thread pools.
- [ ] (R11.2) Add async context invariant tests (two parallel tasks with different IDs).

### R12 — Performance & Scalability Budgets
- [ ] (R12.1) Document budgets in module docstrings/design; implement benchmark test (initially informative).
- [ ] (R12.2) Collect baseline metrics for a week; then flip benchmark to blocking with CI flag.

### R13 — Documentation & Agent Portal
- [ ] (R13.1) Update docs/Agent Portal to reference new schemas/examples; run `make artifacts` and inspect diffs.
- [ ] (R13.2) Ensure cross-links and anchors are valid (open in editor/GitHub).

### R14 — Packaging & Distribution Adoption
- [ ] (R14.1) Build wheels (`uv run python -m build`); run clean-venv install + smoke test.
- [ ] (R14.2) Update CHANGELOG with namespace consolidation + extras info.

### R15 — Security & Supply Chain
- [ ] (R15.1) Run `uv run pip-audit` post-dependency changes; attach report.
- [ ] (R15.2) Confirm no unsafe loaders; add regression tests for sanitized inputs/path whitelist.

### R16 — Idempotency & Retry Semantics
- [ ] (R16.1) Document idempotency/ retry behavior per endpoint; update Problem Details for exhausted retries.
- [ ] (R16.2) Add tests simulating repeated commands ensuring convergence.

### R17 — File/Time/Number Hygiene
- [ ] (R17.1) Replace naive datetimes with timezone-aware versions; ensure durations use `time.monotonic()`.
- [ ] (R17.2) Update tests verifying timestamps include timezone and decimals used where required.

> **Adoption exit criteria:** Every requirement R1–R17 satisfied, acceptance gates green, PR summary posted with links, clean baseline maintained.

---

## 2. Testing & Verification
- [ ] (T1) Map each test to a requirement/scenario via docstring comment for traceability.
- [ ] (T2) Ensure table-driven (`@pytest.mark.parametrize`) coverage for happy/edge/failure cases; avoid single-path tests.
- [ ] (T3) Run doctest/xdoctest suite (`pytest --doctest-modules --doctest-continue-on-failure`).
- [ ] (T4) Execute async suite (`uv run pytest -q --asyncio-mode=auto`).
- [ ] (T5) Run coverage (`uv run pytest -q --cov=src --cov-report=xml:coverage.xml --cov-report=term-missing`) and address low coverage.
- [ ] (T6) Execute performance benchmark (informative first, blocking once stabilized) and archive output.
- [ ] (T7) Run async context invariant tests to ensure correlation IDs do not cross.

## 3. Schemas & Contracts
- [ ] (S1) Validate every schema against the 2020-12 meta-schema (`jsonschema validate -i <schema> -s https://json-schema.org/draft/2020-12/schema`).
- [ ] (S2) Validate each example JSON against its schema (`jsonschema validate -i schema/examples/... -s schema/...`).
- [ ] (S3) Regenerate OpenAPI and lint via `spectral lint schema/search_api/openapi.json`.
- [ ] (S4) Use `assert_model_roundtrip` tests for each model to ensure schema parity.
- [ ] (S5) Document schema version bumps and compatibility notes in PR + CHANGELOG when applicable.

## 4. Observability, Security & Ops
- [ ] (O1) Verify log records include `correlation_id`, `operation`, `status`, `duration_ms` (tests + manual sample).
- [ ] (O2) Validate metrics endpoints (manual `curl` or test harness) and inspect trace spans for error status.
- [ ] (O3) Run `uv run pip-audit` and (if available) secret scan; attach reports.
- [ ] (O4) Run `import-linter --config importlinter.cfg` to enforce layer contracts.
- [ ] (O5) Run suppression guard: `python tools/check_new_suppressions.py src`.
- [ ] (O6) Prepare `$GITHUB_STEP_SUMMARY` in CI with coverage/docs/portal/schema/build links.

## 5. Acceptance Gates (attach outputs to PR)
```bash
uv run ruff format && uv run ruff check --fix
uv run ruff check --select D,ANN,S,PTH,TRY,ARG,DTZ
uv run pyrefly check
uv run mypy src
uv run pytest -q
uv run pytest -q --asyncio-mode=auto
uv run pytest -q --cov=src --cov-report=xml:coverage.xml --cov-report=term-missing
pytest --doctest-modules --doctest-continue-on-failure
uv run pytest -q tests/perf/test_search_latency.py  # informative; flip to blocking when stable
make artifacts && git diff --exit-code
openspec validate lint-type-hardening --strict
uv run pip-audit
spectral lint schema/search_api/openapi.json
jsonschema validate -i schema/agent_catalog/session.v1.json -s https://json-schema.org/draft/2020-12/schema
jsonschema validate -i schema/examples/problem_details/search-missing-index.json -s schema/examples/problem_details/search-missing-index.json
import-linter --config importlinter.cfg
python tools/check_new_suppressions.py src
uv run python -m build
```

## 6. Sign-off
- [ ] (6.1) Domain owner (quality/infra) approval
- [ ] (6.2) Implementation owner (search/catalog) approval
- [ ] (6.3) Security & observability review sign-off
- [ ] (6.4) Product/doc review for schema & documentation updates
- [ ] (6.5) CI green with artifacts uploaded (coverage, docs, Agent Portal, mypy HTML, build artifacts, codemod logs)
- [ ] (6.6) PR description links each requirement R1–R17 to commits/tests and includes all Acceptance Gate outputs + codemod/suppression logs

## 7. Reference & Support
- Review `design.md` for phase guidance, testing matrix, automation notes, and risk mitigations.
- Coordinate with docstring/governance change tracks when edits overlap.
- Add type stubs under `stubs/` when encountering missing typing; never default to `Any`.
- Reach out to observability/security owners for tracing/metric conventions and to CI owners for summary automation.
- Use the “Seven-day first wins” plan (R1, R2, R4, R7, CI automation) to land early momentum before broader migration.

