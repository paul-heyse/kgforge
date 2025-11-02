## ADDED Requirements
### Requirement: Eliminate Pyrefly Suppressions
The codebase SHALL remove existing `pyrefly` suppressions and replace them with typed interfaces, stubs, or helpers that keep optional dependencies safe without sacrificing static guarantees.

#### Scenario: Observability metrics use shared typed helpers
- **GIVEN** tooling observability modules (`tools/_shared/metrics.py`, `tools/**/observability.py`)
- **WHEN** `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini tools/_shared tools/docstring_builder tools/docs tools/navmap` execute
- **THEN** no `type: ignore[...]` suppressions remain around Prometheus metrics, and optional dependency fallbacks continue to operate via the shared typed facade

#### Scenario: FAISS adapters compile without suppressions
- **GIVEN** `src/search_api/faiss_adapter.py` and `src/vectorstore_faiss/gpu.py`
- **WHEN** `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini` cover `src/search_api/faiss_adapter.py src/vectorstore_faiss/gpu.py` in both CPU-only and GPU-extra environments
- **THEN** adapters rely solely on typed protocols (no runtime `Any` fallbacks), and cloning/normalisation helpers remain type-safe without suppressions

#### Scenario: FastAPI wiring retains type guarantees
- **GIVEN** `src/search_api/app.py` and `kgfoundry_common/errors/http.py`
- **WHEN** `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check src/search_api/app.py kgfoundry_common/errors/http.py`, and `uv run mypy --config-file mypy.ini src/search_api/app.py kgfoundry_common/errors/http.py` execute
- **THEN** middleware/exception handlers compile without suppressions because FastAPI stubs or typed wrappers expose precise signatures

#### Scenario: Suppression manifest stays empty
- **GIVEN** `scripts/check_pyrefly_suppressions.py`
- **WHEN** the script runs in CI/pre-commit
- **THEN** it reports zero unmanaged `type: ignore` entries, failing the build if new suppressions are added without justification

