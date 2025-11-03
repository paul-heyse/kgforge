## MODIFIED Requirements
### Requirement: Sanctioned Typing Façade
The system SHALL expose canonical typing façade modules (`kgfoundry_common.typing`, `tools.typing`, `docs.typing`) that provide safe access to shared type aliases, protocols, and optional third-party imports exclusively under TYPE_CHECKING guards. All runtime, tooling, and documentation packages SHALL import shared types through these façades, and Phase 1 compatibility shims (`resolve_numpy`, `docs._types`, `_cache`, `_ParameterKind`, etc.) SHALL be removed.

#### Scenario: Runtime packages consume the façade
- **GIVEN** modules in `src/kgfoundry_common`, `src/kgfoundry`, and `src/search_api`
- **WHEN** they reference shared aliases such as `NavMap`, `ProblemDetails`, numpy dtypes, or FAISS adapters
- **THEN** imports originate from the façade modules (verified via import-linter contract `typing-facade-only`), the functions `resolve_numpy`, `resolve_fastapi`, and `resolve_faiss` raise `ImportError` with remediation guidance if invoked, and Ruff reports zero `PLC2701` or `TC00x` violations for those modules

#### Scenario: Docs & tooling adopt façade helpers
- **GIVEN** scripts under `docs/_scripts/`, `docs/scripts/`, and tooling packages like `tools/docstring_builder` and `tools/navmap`
- **WHEN** they annotate functions that require numpy, FastAPI, FAISS, or internal protocols
- **THEN** annotations import via façade helpers or TYPE_CHECKING blocks, doctest/xdoctest executions pass without optional dependencies installed, and no module imports the retired private packages (`docs._types`, `_cache`, `_ParameterKind`)

### Requirement: Automated Typing Gate Enforcement
The system SHALL enforce TYPE_CHECKING hygiene through Ruff policies, an AST-based checker (`python -m tools.lint.check_typing_gates`), and import-linter contracts so that unguarded type-only imports and direct private-module usage are rejected locally and in CI.

#### Scenario: Typing gate checker reports clean tree
- **GIVEN** the repository after migrations
- **WHEN** `python -m tools.lint.check_typing_gates src tools docs` runs
- **THEN** it reports zero violations, and the checker includes targeted rules for numpy, FastAPI, FAISS, pydantic, sqlalchemy, torch, sklearn, and internal module lists enumerated in the Phase 2 codemod

#### Scenario: CI/pre-commit block regressions
- **GIVEN** a pull request introducing an unguarded numpy import or direct import from a retired module
- **WHEN** the pre-commit hook and CI pipeline run Ruff, import-linter (`typing-facade-only` contract), and the typing gate checker
- **THEN** the pipeline fails with actionable guidance pointing to the façade helpers, preventing the change from merging until fixed

### Requirement: Runtime Import Safety Tests
The system SHALL execute representative CLI entry points and docs scripts in environments without optional dependencies to verify that postponed annotations and typing gates preserve runtime determinism.

#### Scenario: Tooling smoke tests succeed without optional extras
- **GIVEN** a CI job that removes optional dependencies (`uv pip uninstall faiss-cpu fastapi numpy -y`)
- **WHEN** the job executes `tools/navmap/build_navmap.py`, `docs/_scripts/build_symbol_index.py`, and `docs/scripts/validate_artifacts.py`
- **THEN** each CLI completes successfully or emits documented Problem Details without raising import errors, and the job records pass/fail status for observability

#### Scenario: Runtime surfaces clear Problem Details for missing deps
- **GIVEN** the smoke suite intentionally invokes `orchestration/cli.py:index_faiss` without FAISS installed
- **WHEN** the CLI runs
- **THEN** it emits the canonical Problem Details payload describing missing FAISS, exits gracefully, and the smoke test asserts the payload matches `schema/examples/problem_details/faiss-index-build-timeout.json`

## ADDED Requirements
### Requirement: Typing Compliance Reporting
The system SHALL publish typing-gate compliance metrics (Ruff violation counts, typing gate checker findings, import-linter status, smoke test outcomes) as structured logs and Prometheus counters so maintainers can monitor drift.

#### Scenario: CI emits structured compliance metrics
- **GIVEN** the CI workflow running lint, typing, and smoke jobs
- **WHEN** each job completes
- **THEN** it logs JSON records (via `kgfoundry_common.logging`) with fields `job`, `status`, `violations`, and `facade_regressions`, and increments Prometheus counters `kgfoundry_typing_gate_violations_total` and `kgfoundry_typing_gate_checks_total`

#### Scenario: Maintainers review compliance dashboard
- **GIVEN** the metrics exported to the observability stack
- **WHEN** maintainers inspect the dashboard during release readiness review
- **THEN** they can drill down by package (docs, tools, runtime) to identify regressions, and compliance thresholds are documented in release notes


