## 1. Migration Blueprint

- [ ] 1.1 Generate import inventory and ownership matrix.
  - Run `python -m tools.lint.check_typing_gates --list --json` to capture every module importing `_types`, `_cache`, `_ParameterKind`, `resolve_numpy`, `resolve_fastapi`, or heavy third-party packages outside TYPE_CHECKING.
  - Export results to `openspec/changes/typing-gates-holistic-phase2/artifacts/import-inventory.json` with ownership tags (docs toolchain, tooling, runtime, tests) to drive codemod batches.

- [ ] 1.2 Codemod docs toolchain to façade-only imports.
  - Author LibCST codemods that rewrite imports for `docs/_scripts/*.py`, `docs/scripts/*.py`, and `docs/types/**/*.py` to consume `docs.typing` / `kgfoundry_common.typing` symbols.
  - Update module docstrings and `__all__` declarations where façade re-exports change public surface.
  - Add regression tests in `tests/docs/test_typing_imports.py` verifying key scripts import-clean in isolation.

- [ ] 1.3 Codemod tooling packages.
  - Apply codemods to `tools/docstring_builder/**`, `tools/navmap/**`, `tools/lint/**`, and `tools/shared/__init__.py` to replace private imports (`docs._types`, `_cache`, `_ParameterKind`) with façade utilities.
  - Ensure stateful caches expose public APIs so consumers stop touching `_cache` attributes; add helper adaptor functions where necessary.
  - Update stubs under `stubs/tools/**` to mirror new import paths.

- [ ] 1.4 Codemod runtime packages.
  - Rewrite imports in `src/kgfoundry_common`, `src/kgfoundry`, `src/search_api`, `src/orchestration`, and `src/vectorstore_faiss` to remove `resolve_numpy/fastapi/faiss` helpers in favour of façade or `gate_import` usage.
  - Replace namespace bridge references (`kgfoundry/vectorstore_faiss/gpu.py`, `kgfoundry/search_api/faiss_adapter.py`) with façade aliases and update doctest examples accordingly.
  - Synchronise stubs (`stubs/kgfoundry_common/**`, `stubs/search_api/**`) to expose identical `__all__` collections.

- [ ] 1.5 Remove compatibility shims and dead paths.
  - Delete `resolve_numpy`, `resolve_fastapi`, `resolve_faiss`, and any remaining shim functions from `kgfoundry_common.typing`; replace with raising stubs that point to façade usage if legacy code remains.
  - Remove `docs/_types/**` and `_cache` exports; add explicit `ImportError`-raising modules with deprecation guidance and unit tests covering the error message.
  - Clean per-file Ruff ignores tied to shims (update `pyproject.toml` accordingly).

- [ ] 1.6 Update configuration for strict enforcement.
  - Add import-linter contract `typing-facade-only` preventing targeted packages from importing retired modules.
  - Tighten Ruff configuration: enable repository-specific rule banning `resolve_*` usage, drop temporary ignores, and ensure `TC00x` applies to all migrated directories.
  - Extend pre-commit hook to run the new import-linter contract and `python -m tools.lint.check_typing_gates --diff`.

## 2. Validation & Enforcement

- [ ] 2.1 Extend typing gate checker coverage.
  - Enhance `tools.lint.check_typing_gates` to understand new façade patterns, emit actionable autofix suggestions, and support `--list` output consumed by codemods.
  - Add unit tests in `tests/tools/test_check_typing_gates.py` covering failed cases (direct numpy import, private module import) and expected messaging.

- [ ] 2.2 Harden import-linter & Ruff gates.
  - Introduce `importlinter` configuration under `tools/lint/importlinter.typing.ini` enforcing façade-only boundaries and integrate execution into CI.
  - Write regression tests (pytest) that intentionally violate the contract inside a temporary module to ensure the check fails with the expected message.

- [ ] 2.3 Implement runtime smoke suite.
  - Create `tests/tools/test_typing_gate_smoke.py` that uninstalls optional dependencies within a subprocess (`uv run python -m tools.tests.typing_gate_smoke`) and runs `tools/navmap/build_navmap.py`, `docs/_scripts/build_symbol_index.py`, `docs/scripts/validate_artifacts.py`, and `orchestration/cli.py:index_faiss`.
  - Validate that success paths complete and failure paths emit canonical Problem Details payloads; capture structured logs using `kgfoundry_common.logging`.

- [ ] 2.4 Expand static analysis matrix.
  - After each codemod batch, run `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`, and `uv run pyright --warnings --pythonversion=3.13` scoped to touched packages; log results in change artifacts.
  - Ensure doctest/xdoctest suites covering migrated modules continue to pass.

- [ ] 2.5 Surface compliance metrics.
  - Instrument CI jobs to emit structured log lines and Prometheus counters (`kgfoundry_typing_gate_checks_total`, `kgfoundry_typing_gate_violations_total`).
  - Add pytest assertions that the smoke suite writes metric snapshots to `site/_build/agent/typing-gate.json` for observability ingestion.


