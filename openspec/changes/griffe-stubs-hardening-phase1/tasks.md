## 1. Implementation
- [x] 1.1 Capture current Griffe API surface
  - [x] Record the pinned Griffe version and enumerate loader/plugin signatures used by the docs pipeline
  - [x] Document gaps between runtime exports and existing stubs
- [x] 1.2 Rewrite `stubs/griffe/__init__.pyi`
  - [x] Define precise overloads for `load`, `load_module`, `load_package`, and plugin registration helpers
  - [x] Replace `Any` usage with typed Protocols and generics representing loader results
  - [x] Add docstrings/comments linking overloads to upstream source references
- [x] 1.3 Update `stubs/griffe/loader/__init__.pyi`
  - [x] Annotate loader factories, registries, and plugin hooks with concrete parameter/return types
  - [x] Model optional dependency hooks via Protocols that capture accepted callables
- [x] 1.4 Harden docs Griffe facades
  - [x] Refactor `docs/_types/griffe.py` to use the new typed stubs and manage optional imports via `typing.TYPE_CHECKING`
  - [x] Ensure runtime fallbacks raise descriptive errors when dependencies are unavailable
- [x] 1.5 Regression coverage
  - [x] Add parity tests comparing stub exports and runtime attributes (skip if Griffe unavailable)
  - [x] Add Pyright/Pyrefly smoke tests (via existing test harness) ensuring no `Any` leakage in docs scripts
  - [x] Provide doctest snippet showing typed loader usage with graceful fallback handling
- [x] 1.6 Upstream contribution prep
  - [x] Draft summary of stub changes for potential PR to the Griffe project
  - [x] Capture maintenance doc covering update workflow and version pinning

## 2. Testing
- [x] 2.1 `uv run ruff format && uv run ruff check --fix`
- [x] 2.2 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.3 `uv run pyrefly check`
- [x] 2.4 `uv run pyright --warnings --pythonversion=3.13`
- [x] 2.5 `uv run pytest -q tests/docs tests/tools/docstring_builder`

## 3. Docs & Artifacts
- [x] 3.1 Update `docs/contributing/docs-pipeline.md` with stub maintenance guidance
- [x] 3.2 Document optional dependency behavior and error messages in the docs reference section
- [x] 3.3 Run `make artifacts && git diff --exit-code`

## 4. Rollout
- [x] 4.1 Coordinate timing with docs owners to avoid breaking artifact builds
- [x] 4.2 Notify downstream consumers about tighter typing expectations
- [x] 4.3 Plan upstream Griffe PR submission after local validation

