## ADDED Requirements
### Requirement: Postponed Annotations Baseline
The system SHALL enable postponed evaluation of annotations for every module within `src/`, `tools/`, `docs/_scripts/`, `docs/scripts/`, and `tests/` that participates in runtime execution or lint/type analysis, ensuring type hints no longer require runtime imports of optional dependencies.

#### Scenario: Ruff enforces postponed annotations
- **GIVEN** the repository with the updated Ruff configuration
- **WHEN** `uv run ruff check --fix` executes over the entire tree
- **THEN** no module under the targeted directories reports missing postponed-annotation directives, and newly added modules fail the gate until the directive (or Python 3.13 equivalent) is present

#### Scenario: Type checkers run without optional dependencies
- **GIVEN** a virtual environment lacking FAISS, FastAPI, and other optional extras
- **WHEN** `uv run pyright --warnings --pythonversion=3.13` and `uv run pyright --warnings --pythonversion=3.13` execute
- **THEN** both commands succeed without import errors attributable to type-hint evaluation, demonstrating postponed annotations remove eager imports

### Requirement: Sanctioned Typing Façade
The system SHALL expose canonical typing façade modules (`kgfoundry_common.typing`, `tools.typing`, `docs.typing`) that provide safe access to shared type aliases, protocols, and optional third-party imports exclusively under TYPE_CHECKING guards, replacing direct imports from private or heavy dependencies.

#### Scenario: Runtime modules consume the façade
- **GIVEN** runtime packages such as `kgfoundry_common`, `kgfoundry`, and `search_api`
- **WHEN** they reference shared aliases (e.g., `NavMap`, `ProblemDetails`, numpy dtypes)
- **THEN** imports originate from the façade modules, no private package (`_types`, `_cache`) is imported, and runtime execution succeeds even when optional dependencies are absent

#### Scenario: Tooling adopts façade helpers
- **GIVEN** docs and tooling scripts under `docs/_scripts/` and `tools/`
- **WHEN** they annotate functions requiring heavy dependencies (FastAPI, FAISS, numpy)
- **THEN** annotations import via façade helpers or TYPE_CHECKING blocks, doctest/xdoctest executions pass, and Ruff emits no `TC00x` or `PLC2701` warnings for those modules

### Requirement: Typing Gate Enforcement
The system SHALL provide automated enforcement—via Ruff policies, an AST-based checker (`python -m tools.lint.check_typing_gates`), and CI integration—that prevents unguarded type-only imports from entering the codebase and asserts CLI entry points remain import-clean.

#### Scenario: Typing gate checker passes clean tree
- **GIVEN** the repository after façade adoption
- **WHEN** `python -m tools.lint.check_typing_gates` runs without arguments
- **THEN** it reports zero violations, demonstrating all third-party and internal heavy imports used solely for type hints are safely guarded

#### Scenario: CI blocks regressions
- **GIVEN** a pull request introducing an unguarded numpy import for typing purposes
- **WHEN** the CI pipeline runs Ruff and the typing gate checker
- **THEN** the pipeline fails with actionable error messages, preventing the change from merging until the import is moved behind a TYPE_CHECKING gate or façade helper

### Requirement: Runtime Import Safety Tests
The system SHALL execute representative CLI entry points and docs scripts in environments without optional dependencies to verify that postponed annotations and typing gates preserve runtime determinism.

#### Scenario: Tooling smoke tests succeed without optional extras
- **GIVEN** a CI job that uninstalls optional dependencies (`uv pip uninstall faiss-cpu fastapi -y`)
- **WHEN** the job executes `typer`-based CLIs (`tools/navmap/build_navmap.py`, `docs/_scripts/build_symbol_index.py`) and captures exit codes
- **THEN** each CLI completes successfully (or emits documented Problem Details for legitimate runtime requirements), proving type-only imports no longer break tooling when extras are missing


