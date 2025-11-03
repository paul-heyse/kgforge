## 1. Implementation

- [x] 1.1 Capture baseline lint violations.
  - Export current Ruff `TC00x`, `INP001`, `EXE00x`, and `PLC2701` findings with paths for regression tracking.
  - Record modules lacking `from __future__ import annotations` or relying on private `_types` imports.
- [x] 1.2 Introduce typing façade packages.
  - Create `src/kgfoundry_common/typing` with helper APIs (`gate_import`, `safe_imports`) and shared aliases.
  - Add tooling/doc counterparts (`tools/typing`, `docs/typing`) re-exporting helpers.
  - Provide compatibility shims with deprecation warnings for private module access.
- [x] 1.3 Automate postponed annotations adoption.
  - Implement fixer script to insert `from __future__ import annotations` while preserving module headers.
  - Run fixer across `src/`, `docs/_scripts/`, `docs/scripts/`, `tools/`, and `tests/` batches; rerun Ruff formatter afterwards.
- [x] 1.4 Refactor type-only imports.
  - Replace heavy imports with façade access in runtime, docs, and tooling modules; ensure runtime code uses lazy helpers.
  - Update stubs to mirror new import paths and remove `# type: ignore` pragmas.
- [x] 1.5 Enhance Ruff enforcement.
  - Configure Ruff to require postponed annotations and to escalate `TC00x`, `INP001`, `EXE00x`, `PLC2701` to errors.
  - Add repository rule banning private-module imports outside designated compatibility shims.
- [x] 1.6 Implement typing gate checker.
  - Build `tools/lint/check_typing_gates.py` that analyses ASTs for unguarded type-only imports.
  - Wire the checker into CI and developer tooling (pre-commit hook / scripts).

## 2. Testing

- [x] 2.1 Add pytest coverage for façade helpers.
  - Validate `gate_import` lazily loads modules, memoises results, and raises informative errors when dependencies are absent.
  - Confirm compatibility shims emit warnings and guide developers to new APIs.
- [x] 2.2 Verify runtime determinism without optional deps.
  - Execute representative CLIs (`docs/_scripts/build_symbol_index.py`, `tools/navmap/build_navmap.py`, `orchestration/cli.py`) in an environment missing FAISS/FastAPI, asserting graceful behaviour.
- [x] 2.3 Expand lint/typing test matrix.
  - Run `uv run ruff check`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini` across updated modules, ensuring zero regressions.
  - Execute `python -m tools.lint.check_typing_gates --diff` within CI to prove no unguarded imports remain.
- [ ] 2.4 Doctest/xdoctest validation.
  - Ensure doctest examples continue to execute with postponed annotations and façade imports, updating examples where necessary.

## 3. Docs & Rollout

- [x] 3.1 Update AGENTS.md and developer onboarding docs describing postponed annotations and façade usage.
- [x] 3.2 Document migration path for private-module imports, including deadlines and troubleshooting tips.
- [x] 3.3 Regenerate docs/artifacts (`make artifacts`) once module paths change; verify clean git diff.
- [x] 3.4 Announce new CI gate (`check_typing_gates`) in release notes and internal communication channels.
- [ ] 3.5 Monitor CI for two release cycles, then remove compatibility shims and enforce façade-only imports.


