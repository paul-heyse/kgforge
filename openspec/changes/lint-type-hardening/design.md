## Context
- Ruff reports 709 violations (pathlib usage, blind exceptions, security flags) across search, registry, embeddings, and orchestration modules.
- Mypy emits 706 errors driven by pervasive `Any`, unfollowed imports (pyarrow/duckdb), protocol mismatches, and untyped public APIs.
- Current error handling swallows exceptions and fails to emit Problem Details, blocking dependable observability and client integrations.
- Lax configuration handling allows missing env vars to succeed silently, leading to runtime failures.
- Documentation and schema contracts lag behind code, hampering discoverability and schema-source-of-truth alignment.
- Namespace duplication (`vectorstore_faiss` vs `kgfoundry.vectorstore_faiss`) and optional GPU dependencies make packaging confusing and affect type checking.

## Goals / Non-Goals
- **Goals:**
  - Establish typed, reusable infrastructure for filesystem access, configuration, serialization, vector search adapters, and structured observability.
  - Eliminate blind exception handling and emit RFC 9457 Problem Details with a frozen taxonomy + code registry.
  - Define canonical JSON Schemas/OpenAPI surfaces with examples, version metadata, and automated validation.
  - Guarantee docstring, typing, and Problem Details coverage for every public API touched, with zero unmanaged suppressions.
  - Drive Ruff/mypy/pyrefly baselines to zero, enabling import-linter contracts and “no new suppressions” automation.
  - Ensure concurrency, performance budgets, packaging, security, and idempotency requirements are explicit and test-backed.
- **Non-Goals:**
  - Repo-wide docstring formatting beyond what is necessary for clarity (handled by docstring-builder-hardening).
  - Governance/process policy edits (tracked elsewhere).
  - Introducing net-new product behavior; scope is hardening and parity.

## Phased Execution Strategy
1. **Phase 1 — Foundation (shared helpers/types):**
   - Deliver reusable building blocks: R1 (pathlib helpers + codemod), R2 (exception taxonomy + Problem Details registry), R3 (serialize/deserialize helpers), R4 (schema/OpenAPI scaffolding), R5 (protocols), R7 (RuntimeSettings), R8 (LoggerAdapter/metrics envelope), namespace consolidation and packaging extras (GPU) groundwork, import-linter config, suppression-check script.
   - Land each helper with isolated unit tests and updated tooling configs (mypy plugins, Ruff type-checking settings) before touching module call sites.
   - Freeze exception codes/type URIs (`kgfoundry_common.errors.codes`) prior to adoption to avoid churn.
2. **Phase 2 — Adoption (cluster-by-cluster migrations):**
   - Migrate product surfaces incrementally: search adapters → registry → embeddings → orchestration → remaining packages, executing requirements R6–R17.
   - Apply codemods (`tools/codemods/pathlib_fix.py`, `tools/codemods/blind_except_fix.py`) before manual cleanup in each cluster.
   - Run acceptance gates after each requirement and attach outputs (ruff/mypy/pyrefly, doctests, schema lint, import-linter, suppression check, build, benchmark logs) to the PR or step summary.

## Decisions (with rationale)
- **Codemod-first filesystem migration:** Provide LibCST/Bowler codemods to replace `os.path`/`open` constructs, lowering manual toil and risk.
- **Exception taxonomy registry:** Maintain `kgfoundry_common.errors.codes` enumerating stable codes + type URIs so Problem Details responses and docs remain synchronized.
- **Schema-first + examples:** Enforce `$id`, `x-version`, compatibility notes, and example directories; models use Pydantic with `extra="forbid"` and round-trip helpers.
- **Protocol-based vector interfaces:** Adopt PEP 544 protocols + TypedDicts, enabling mypy to verify adapters; deliver stubs for FAISS/libcuvs/pyserini.
- **Balanced mypy strictness:** Enable `plugins = pydantic.mypy, sqlalchemy.ext.mypy.plugin`, keep tests type-checked at def-level while allowing untyped calls to pytest helpers.
- **Ruff import policy:** Turn on `flake8-type-checking` strict mode to minimize runtime dependency drag.
- **Observability adapter:** Provide `get_logger(__name__)` returning LoggerAdapter with mandatory structured fields and context propagation.
- **Import-linter + suppression guard:** Codify layer boundaries and fail CI when new `# type: ignore`/`noqa` appear without `TICKET:` label.
- **Namespace consolidation & GPU extras:** Expose public APIs via `kgfoundry.*` only and supply optional `gpu` extra to isolate heavy deps.
- **PR summary automation:** After CI, post summary with links to coverage, docs, portal, schema lint output, and build logs for reviewer efficiency.
- **Performance gate evolution:** Start benchmarks as non-blocking reporting for baseline capture, then flip to blocking once stable.

## Implementation Blueprint (per requirement)
> See `tasks.md` for detailed steps, codemod commands, and acceptance gates. Phase 1 covers R1–R8 foundation work; Phase 2 adopts the helpers across modules while fulfilling R6–R17.

### R1 — Pathlib Standardization Across Workflows
- Implement `kgfoundry_common.fs` helpers + codemod (`python -m tools.codemods.pathlib_fix src`). Review codemod output, rerun Ruff autofix, and add missing docstrings/types. Tests cover success/error/traversal.

### R2 — Exception Taxonomy & Problem Details
- Define taxonomy + registry of codes/type URIs; codemod `except Exception:` into typed handlers; implement Problem Details helper returning stable payload; update HTTP/CLI integration; freeze codes before adoption.

### R3 — Secure Serialization & Persistence
- Provide drop-in `serialize_json(obj, schema_path)`, `deserialize_json(data, schema_path)` helpers performing schema validation + checksum; adopt across embeddings/search/registry.

### R4 — Typed JSON Contracts
- Author schemas with `$id`, `x-version`, compatibility notes, and examples; run `jsonschema validate` + `spectral lint`; implement Pydantic models with round-trip helper `assert_model_roundtrip(model_cls, example_path)`.

### R5 — Vector Search Protocol Compliance
- Build protocols + stubs; consolidate public namespace (only `kgfoundry.vectorstore_faiss`); update adapters; optional GPU features under `pip install kgfoundry[gpu]`.

### R6 — Parquet IO Type Safety
- Annotate functions, add stubs for pyarrow/duckdb, enforce schema validation, log typed errors.

### R7 — Typed Configuration & 12-Factor Compliance
- Implement RuntimeSettings with fail-fast behavior, docstrings listing env vars, and structured Problem Details on failure; ensure logs go to stdout/stderr.

### R8 — Structured Observability Envelope
- Provide LoggerAdapter + metrics/tracing utilities; require adoption in each module touched; add tests for required fields.

### R9–R17 — Adoption & Advanced Requirements
- R9: Public API hygiene/docstrings referencing examples.
- R10: Tooling enforcement (mypy plugins, Ruff config, automation).
- R11: Concurrency context with `ContextVar` and async context invariant tests.
- R12: Performance budgets with monotonic timing + staged benchmark gate.
- R13: Documentation/Agent Portal updates with copy-ready examples.
- R14: Packaging extras + clean wheels + smoke installs.
- R15: Security & supply chain scanning, path sanitization.
- R16: Idempotency/retry semantics documented and tested.
- R17: File/time/number hygiene (timezone-aware datetimes, decimals, monotonic durations).

## Architecture Sketch
- Inputs: env vars/settings, schemas/examples, CLI/HTTP requests.
- Processing: codemod-supported helpers, typed models/protocols, structured logging/metrics/tracing, import-linter enforced layering.
- Outputs: deterministic Problem Details, schema-validated artifacts, packaging outputs, PR summary with verification links.

## Data Model & Schemas
- Schema source of truth under `schema/**` with `$id`, `x-version`, compatibility notes, and examples. Example JSON resides in `schema/examples/**` and doctests reference these paths.
- Round-trip helper ensures each Pydantic model matches the schema; tests call `assert_model_roundtrip(Model, example_path)`.

## Invariants & Edge Cases
- Codemods run prior to manual edits; manual adjustments are reviewed for docstring/type compliance.
- Exception codes/type URIs remain stable post-freeze; changes require schema/consumer coordination.
- Async context tests ensure no cross-contamination between tasks.
- Performance benchmarks initially informative-only; once stable, they gate merges.
- “No new suppressions” script enforces zero tolerance for unexplained ignores.

## Observability
- LoggerAdapter ensures `correlation_id`, `operation`, `status`, `duration_ms` present in all library logs.
- Metrics abide by naming conventions; traces include error status and type URIs.
- Async context invariant tests catch propagation regressions.

## Testing Matrix
- Expanded to include codemod verification, logger adapter assertions, async context dual-run tests, benchmark harness, schema round-trip helper, import-linter contract tests, suppression checker, packaging smoke tests, and PR summary script.

## Security / Privacy
- Serialization helpers prevent unsafe loaders; path validations block traversal; logs redact secrets.
- Dependency scans (pip-audit) and suppression guard prevent drift.

## Migration / Compatibility
- Namespace consolidation and GPU extras delivered before adapter refactors to avoid double churn.
- Schema versioning notes accompany each change; compatibility documented in schemas and change log.
- Idempotency ensures repeated operations safe across deployments.

## Junior Developer Checklist
1. Complete Phase 1 helpers before touching call sites; run codemods and attach logs to PR.
2. Freeze exception codes/type URIs and document them in Problem Details example before adoption.
3. Enable mypy plugins, Ruff type-checking, import-linter config, suppression guard script.
4. Follow Phase 2 migration order, executing R6–R17 tasks with acceptance gates after each requirement.
5. Update PR summary (GitHub Step Summary) with coverage/docs/schema/build results; link to scenario test evidence.
6. Escalate any scope that impacts docstring/governance tracks.

## Open Questions
- Confirm ownership and rollout plan for PR summary automation (CI team vs. feature team).
- Decide when to flip performance benchmarks from informative to blocking (after first baseline week?).
- Determine whether import-linter contracts should cover additional layers beyond initial domain→adapter boundaries.

