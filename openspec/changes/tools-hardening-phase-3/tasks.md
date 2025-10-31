## 1. Implementation

> **How to read this checklist:** each subsection corresponds to a cluster of failures currently reported by `mypy --config-file mypy.ini tests/tools` (and the matching `pyrefly` runs). Work through them in order—the goal is **zero errors on the test tree**.

### 1.1 Typed payload surfaces
Goal: replace every `dict[str, Any]` payload with a msgspec struct plus helpers so mypy understands the shape of docstring caches, analytics docs, navmap documents, CLI envelopes, and sitecustomize payloads.
- [ ] 1.1.1 Create a payload inventory table (docstring cache, docfacts, CLI envelopes, analytics, navmap, sitecustomize) in the PR description, listing module ↔ schema ↔ producer/consumer.
- [ ] 1.1.2 Introduce/updated msgspec structs for each payload (`tools/_shared/cli.py`, `tools/docstring_builder/models.py`, `tools/docs/analytics_models.py`, `tools/navmap/document_models.py`, `sitecustomize.py`).
- [ ] 1.1.3 Wire helpers (`from_payload`, `to_payload`, validators) that wrap struct usage and raise Problem Details on failure.
- [ ] 1.1.4 Update JSON Schemas under `schema/tools/**`, add normative examples, and regenerate artifacts.
- [ ] 1.1.5 Replace `dict[str, Any]` / `cast(Any, …)` in payload modules and their tests.
- [ ] 1.1.6 Add pytest coverage that round-trips each payload and asserts schema validation rejects malformed input.

### 1.2 LibCST typing support
Goal: eliminate `Any` usage inside codemods so `tests/tools/codemods/**` type-check cleanly.
- [ ] 1.2.1 Extend `stubs/libcst/**` with the nodes/visitors used by our codemods; ensure stubs live alongside a `py.typed` marker.
- [ ] 1.2.2 Refactor codemod helpers to construct CST nodes via typed builders (no dynamic attribute access).
- [ ] 1.2.3 Add typed pytest fixtures/utilities for parsing, applying transformers, and asserting results.
- [ ] 1.2.4 Run `uv run --no-sync pyrefly check tools/codemods` and `uv run --no-sync mypy --config-file mypy.ini tools/codemods` and record outputs.

### 1.3 Typed tooling test harness
Goal: ensure every fixture, decorator, and helper in `tests/tools/**` has explicit types so the test suite passes under strict mypy.
- [ ] 1.3.1 Draft/update `tests/tools/README.md` with fixture/decorator typing guidelines for contributors.
- [ ] 1.3.2 Annotate fixtures, parametrized tests, and helper utilities across `tests/tools/**`, eliminating `Any` escapes.
- [ ] 1.3.3 Introduce helper factories (e.g., `make_error_report()`) to simplify typed construction in tests.
- [ ] 1.3.4 Execute `uv run --no-sync mypy --config-file mypy.ini tests/tools` after each directory sweep and attach logs to the PR.

### 1.4 Exports referenced by tests
Goal: ensure everything imported as `from tools import ...` inside `tests/tools/**` actually exists and is typed.
- [ ] 1.4.1 Audit runtime exports and update `tools/__init__.py` (plus relevant subpackages) to expose the helpers used across the test suite.
- [ ] 1.4.2 Author stubs under `stubs/tools/**` (no `Any` fallbacks) matching those exports so mypy stops reporting “module does not explicitly export attribute …”.
- [ ] 1.4.3 Re-run `uv run --no-sync mypy --config-file mypy.ini tests/tools` to verify the export-related errors disappear before moving on.

## 2. Testing
These checks prove that the fixes above really drove the test tree to zero errors.
- [ ] 2.1 Extend or add pytest coverage for payload round-trips and codemod transformations referenced by `tests/tools/**`.
- [ ] 2.2 Run `uv run --no-sync pyrefly check` for each touched module group (`tools/_shared`, `tools/docstring_builder`, `tools/docs`, `tools/navmap`, `tools/codemods`).
- [ ] 2.3 Run `uv run --no-sync mypy --config-file mypy.ini tests/tools` and `uv run --no-sync mypy --config-file mypy.ini tools/codemods` (must pass with no ignores).
- [ ] 2.4 Execute targeted pytest suites (`tests/tools/docstring_builder`, `tests/tools/docs`, `tests/tools/navmap`, `tests/tools/shared`, `tests/tools/codemods`).

## 3. Docs & Artifacts
Document the new contracts and make sure generated artifacts stay in sync.
- [ ] 3.1 Regenerate JSON Schemas / Problem Details examples impacted by typed payload changes.
- [ ] 3.2 Run `make artifacts` and review generated docs/portal outputs for regressions.
- [ ] 3.3 Update `docs/tooling.md` (or equivalent) with a “typed payload checklist” and the commands required to keep `mypy --config-file mypy.ini tests/tools` green.
- [ ] 3.4 Refresh changelog / README snippets referencing tooling exports or installation guidance.

## 4. Rollout
Tie the work off with verifications and communication.
- [ ] 4.1 Capture performance/diagnostic metrics after typed payload conversion (ensure no regression against budgets) using existing benchmark helpers.
- [ ] 4.2 Update the PR summary with before/after `mypy --config-file mypy.ini tests/tools` outputs (should be zero after the change).

