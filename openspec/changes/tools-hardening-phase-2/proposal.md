## Why
Security and quality gates for the `tools/` suite continue to fail because legacy scripts still execute subprocesses with unchecked arguments, swallow broad exceptions, and exceed Ruff complexity budgets. Import-linter integration and Agent Catalog generators are out of sync with their upstream APIs, which blocks automation and raises pyrefly violations. Critical helpers (`tools/docstring_builder`, `tools/docs`, `tools/navmap`) rely on loosely typed dictionaries that spill `Any` across the codebase. Finally, our LibCST codemods ship without local typing support, causing hundreds of mypy errors and discouraging further codemod investment. We need a scoped follow-up to the earlier hardening change to finish securing subprocess usage, bring external integrations back to parity, introduce typed payload models, and provide LibCST stubs so type checkers can reason about codemods.

## What Changes
- [ ] **ADDED**: Secure subprocess & exception discipline across `tools/` by routing executions through `tools._shared.proc.run_tool`, validating command allow-lists, replacing blind `except Exception`, and reducing visitor/handler complexity to Ruff thresholds.
- [ ] **MODIFIED**: External integrations (`tools/check_imports.py`, `tools/docs/build_agent_catalog.py`, related helpers) to match current import-linter and Agent Catalog APIs with typed request objects, structured Problem Details errors, and regression fixtures.
- [ ] **ADDED**: Typed data models (dataclasses/TypedDicts) for docstring builder caches, docs analytics envelopes, navmap records, and CLI results, with JSON Schema validation and generators powered by `msgspec`.
- [ ] **ADDED**: Local `stubs/libcst/` coverage and typed adapter helpers so codemods compile under mypy/pyrefly, plus unit tests covering transformation scenarios.

## Impact
- **Affected specs:** `tools-suite` (new requirements for secure subprocesses, external API parity, typed payloads, and LibCST coverage).
- **Affected code:**
  - `tools/_shared/proc.py`, `tools/_shared/logging.py`, and callers across `tools/docs/*`, `tools/navmap/*`, `tools/gen_readmes.py`, `tools/check_docstrings.py`.
  - Import-linter wrapper (`tools/check_imports.py`) and Agent Catalog generators (`tools/docs/build_agent_catalog.py`, `render_agent_portal.py`).
  - Docstring builder modules (`cache.py`, `models.py`, `harvest.py`, plugins), docs analyzers, navmap utilities, and associated JSON schemas under `schema/tools/`.
  - New or updated type stubs within `stubs/libcst/` plus codemod test fixtures under `tests/tools/codemods`.
- **Rollout:** Deliver in four sequences matching the bullets, each gated by updated Ruff/pyrefly/mypy suites and new pytest parametrized coverage. Secure subprocess changes ship first to unblock Ruff security rules, followed by API parity fixes, typed models, and LibCST coverage. Each phase includes docs/examples and Problem Details samples.
- **Risks:** Widespread refactors risk regressions in documentation pipelines and codemods; mitigated via schema validation, deterministic fixture tests, and dry-run mode verification. Import-linter API drift may differ across versions; we pin supported versions and add contract tests.

## Acceptance
- **Lint & docs**: `uv run ruff format && uv run ruff check --fix` complete with zero violations; public tooling APIs expose PEP 257 docstrings; new Problem Details example `schema/examples/problem_details/tool-exec-timeout.json` ships alongside regenerated docs/artifacts.
- **Type safety**: `uv run pyrefly check` and `uv run mypy --config-file mypy.ini` succeed with no unexplained ignores; public APIs are fully annotated (PEP 695 generics where relevant); LibCST codemods type-check cleanly via local stubs.
- **Schemas & HTTP**: All JSON Schemas validate against the 2020-12 meta-schema with round-trip tests; CLI/documentation payloads include SemVer/version fields and examples; where HTTP surfaces exist, the OpenAPI 3.2 spec lints cleanly via `spectral lint`.
- **Observability**: An induced failure (e.g., forced timeout) produces a structured error log with `correlation_id`, increments `tool_failures_total`, and records an OTEL span with error status and attributes.
- **Security**: `uv run pip-audit --strict` reports no known vulnerabilities; secret scanning passes; all subprocesses execute via allow-listed absolute paths; YAML/pickle usage relies on safe loaders with validation; unsafe `eval/exec` are absent.
- **Configuration**: Typed settings enforce required environment variables, raising `SettingsError` with Problem Details when absent; retry semantics and defaults are documented.
- **Packaging**: `uv run pip wheel .` succeeds; `pip install .[tools]` in a fresh virtual environment runs without runtime import errors and exposes the expected entry points.
- **Idempotency**: Table-driven tests confirm double-run convergence for each CLI, documenting retry semantics.
- **Performance**: Micro-benchmarks demonstrate p95 latency < 2s and RSS < 400 MiB for agreed hot paths, with results captured in test assertions.
- **Quality gates**: `make artifacts`, `openspec validate tools-hardening-phase-2 --strict`, import-linter (`python tools/check_imports.py`), and packaging/docs builds complete without drift; Problem Details builders remain centralized and all `raise ... from e` chains preserve causes.

