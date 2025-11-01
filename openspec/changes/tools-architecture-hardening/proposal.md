## Why

The tools stack still depends on implicit namespace bridges, oversized adapters, and dictionary-shaped payloads that defeat strong typing and layering rules. Ruff flags numerous architectural smells—private cross-package imports, excessive complexity, and deprecated typing forms—while pyrefly cannot resolve intra-package modules because exports remain implicit. Without restructuring, we cannot guarantee stable public APIs, enforce layered boundaries, or make pyrefly the canonical guardrail for tooling consumers.

## What Changes

- **Phase 0 — Source-of-Truth & Packaging Hygiene**
  - Ship tooling as a first-class optional extra: update `pyproject.toml` with `[project.optional-dependencies.tools]`, include `tools` (and `tools/py.typed`) in wheel/sdist build targets, and add a packaging smoke test that installs `kgfoundry[tools]` into a clean venv before exercising `tools.run_tool`.
  - Curate package exports and stubs together: align `tools/__init__.py`, `tools/docs/__init__.py`, and `stubs/tools/**` via a shared `PUBLIC_EXPORTS` mapping with NumPy-style docstrings pointing to Problem Details exemplars.
  - Replace ad-hoc namespace bridging by introducing a public adapter around `kgfoundry._namespace_proxy`, updating legacy bridge modules to consume it, and running Ruff/pyrefly guards to prevent private imports from reappearing.

- **Phase 1 — Static Checker Guardrails**
  - Add a shared lint defaults module (`tools._shared.linting`) that normalises union ordering, flags deprecated typing aliases, and provides decorators for static/class method conversions so Ruff (`RUF036`, `UP035`, `UP040`, `PLR6301`) stays green without bespoke ignores.
  - Tighten Ruff settings and pre-commit so `ruff-format`, `ruff --fix`, and `pyrefly check` run automatically over `tools/**`; document the workflow and helper usage in `openspec/AGENTS.md` for contributors.

- **Phase 2 — API Clarity & Module Boundaries**
  - Refactor package initialisers (`tools/__init__.py`, `tools/docs/__init__.py`, etc.) to expose only stable orchestration entry points with fully annotated signatures and PEP 257 docstrings describing their exception contracts.
  - Split oversized adapters such as `tools/docstring_builder/cli.py` into orchestrator, IO, and adapter layers so CLI parsing, filesystem coordination, and pure domain logic live in separate modules with explicit dependency flow.

- **Phase 3 — Security, Reliability, and Context Propagation**
  - Centralise subprocess execution in `tools._shared.proc` with a `ContextVar`-backed operation ID, guaranteed timeout enforcement, structured logging, and `raise ... from e` semantics.
  - Introduce retry-aware wrappers for idempotent tool runs and ensure observability hooks (logging, metrics, tracing) share the same correlation context without leaking secrets.
  - Harden configuration by promoting `tools._shared.settings` to namespaced `BaseSettings` classes that validate environment-driven configuration and raise structured errors for missing or invalid fields.

- **Phase 4 — Modularity & Complexity Reduction**
  - Decompose functions exceeding Ruff complexity thresholds (e.g., BM25/FAISS orchestrators, agent catalog search) into composable strategies and pure helpers, ensuring domain logic stays separable from I/O.
  - Generate import-linter contracts that lock in one-way dependencies between domain modules, adapters, and CLI entry points, preventing regressions.

- **Phase 5 — Data Contracts & Validation**
  - Replace untyped payloads (docstring edits, navmap documents, analytics envelopes) with msgspec structs or frozen dataclasses and expose schema emission helpers from `tools._shared.schema`.
  - Provide legacy conversion helpers so orchestrators validate incoming payloads against the new models before processing, ensuring schema compliance and typed boundaries.

Each phase will conclude by running the existing gate trio (`uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`) before progressing to the next, keeping the architecture refactor incremental and reversible.

## Impact

- **Affected specs:** `openspec/specs/tools-runtime` and `openspec/specs/tools-navmap` will be updated to encode the new layering, typed payloads, and configuration requirements.
- **Affected code:** `tools/__init__.py`, `tools/docs/__init__.py`, `tools/_shared/**`, `tools/docstring_builder/**`, `tools/docs/**`, `tools/navmap/**`, `kgfoundry/_namespace_proxy.py`, and `stubs/tools/**`.
- **Rollout:** Deliver phases sequentially, verifying Ruff, pyrefly, and mypy after each tranche. No additional documentation or testing scope is introduced beyond the pre-commit checks already in place.


Original message describing scope:

Assessment
ruff check src reports 177 violations: missing/incorrect docstrings (D101, D417), unsafe APIs (S403, S113, BLE001), complexity hotspots (C901, PLR091x), None ordering (RUF036), private import bridges (PLC2701), and outdated typing forms (UP0xx).
pyrefly check fails with 139 errors, principally missing modules across the tools.* namespace plus signature mismatches when calling kgfoundry.agent_catalog.search.search_catalog from tooling.
mypy --config-file mypy.ini src is already clean; we need to keep it that way while resolving the other gates.
Phase 0 – Source-of-truth & packaging hygiene (Principles 1, 7, 11)
Promote tools/ into a first-class package: add explicit pyproject.toml optional-extra, ensure __init__.py exports the intended API, and wire it into sys.path via pyproject rather than ad-hoc namespace bridges.
Consolidate namespace-bridging helpers (kgfoundry/_namespace_proxy) into a dedicated adapter module with public wrappers so consumers stop importing private names (PLC2701 errors in namespace_bridge, search_client, vectorstore_faiss).
Document public surfaces with NumPy-style docstrings and typed signatures, then re-export only the supported entry points from package __all__ definitions.
Phase 1 – Static-checker guardrails (Principles 1, 4, 7)
Introduce a shared “lint defaults” module to enforce RUF036 canonical union ordering, forbid deprecated typing aliases (UP035/UP040), and supply helper decorators for @classmethod/@staticmethod conversions to eliminate the PLR6301 family without losing clarity.
Update developer tooling to run uv run ruff format && uv run ruff check --fix and uv run pyrefly check in pre-commit so regressions are caught early; publish guidance in AGENTS.md.
Phase 2 – API clarity & documentation (Principles 1, 3, 5, 13)
Audit every public module in kgfoundry_common, observability, and search_api for complete docstrings; backfill parameter docs flagged by D417.
For kgfoundry_common.errors, define a documented exception taxonomy with RFC 9457 Problem Details exemplars and embed at least one sample JSON under schema/examples/problem_details/.
Ensure module-level loggers expose NullHandler and structured logging helpers; replace print usage with logger calls and add correlation-id support.
Phase 3 – Security & reliability hardening (Principles 5, 8, 9, 10, 15)
Replace insecure pickle usage in sparse indexers with explicit schema-governed serializers (e.g., msgpack or Parquet), supported by validation tests and migration notes.
Wrap network calls (search_client.client) with explicit timeouts/retry policies and document idempotency semantics.
Review subprocess usage (agent_catalog.session) to ensure it’s sandboxed, guards inputs, and propagates context (timeouts, cancellation, structured error logs).
Phase 4 – Modularity & complexity reduction (Principles 3, 7, 12)
Break down large procedural functions (embeddings_sparse.bm25, vectorstore_faiss.gpu, agent_catalog.search) into composable strategies with data classes for config/state, reducing branch count and local variables.
Move shared parsing/validation into pure-domain modules; enforce adapter boundaries via tools/check_imports.py once pyrefly passes.
Establish performance budgets (e.g., BM25 indexing throughput) and add micro-benchmarks to catch regressions after refactors.
Phase 5 – Data contracts & validation (Principles 2, 6, 13)
Align index/search payloads with JSON Schema 2020-12, centralize schemas under schema/, and regenerate Pydantic models from those schemas.
For HTTP layers, emit OpenAPI 3.2 specs and validate them with a linter; ensure Problem Details are referenced from the schema set.
Extend tests to round-trip models ↔ schema (success/failure) and to enforce configuration loading via typed settings objects (pydantic_settings).
Phase 6 – Testing & observability (Principles 3, 9, 12, 15)
Build table-driven pytest suites for each refactored module, covering happy paths, edge cases, and failure modes (e.g., corrupt index files, timeout scenarios).
Add doctest/xdoctest coverage for docstring examples, verifying they align with schemas and logging outputs.
Implement observability helpers to emit structured logs, Prometheus metrics, and OpenTelemetry spans for critical operations; add regression tests ensuring failure paths produce logs + counters + spans.
Phase 7 – Verification & rollout
Execute the full gate: uv run ruff format && uv run ruff check --fix, uv run pyrefly check, uv run mypy --config-file mypy.ini src, uv run pytest -q, make artifacts && git diff --exit-code, python tools/check_new_suppressions.py src, python tools/check_imports.py, uv run pip-audit.
Treat plan execution as a multi-PR initiative; document progress in openspec/changes/tools-hardening-phase-4 to keep spec, schemas, and code in sync.
After stabilization, tighten CI (fail-fast on lint/type regressions) and publish migration notes/CHANGELOG entries for any API changes.
Next Checks
When you start implementation, re-run uv run pyrefly check with reduced scope (e.g., --package tools once packaging is fixed) to confirm missing-import issues are resolved.
Validate docstring completion via pydoclint (already wired through Ruff) and ensure new Problem Details examples pass JSON Schema validation.

