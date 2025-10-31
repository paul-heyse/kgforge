## 1. Implementation

### 1.1 Typed payload surfaces
- [ ] 1.1.1 Create a payload inventory table (docstring cache, docfacts, CLI envelopes, analytics, navmap, sitecustomize) in the PR description, listing module ↔ schema ↔ producer/consumer.
- [ ] 1.1.2 Introduce/updated msgspec structs for each payload (`tools/_shared/cli.py`, `tools/docstring_builder/models.py`, `tools/docs/analytics_models.py`, `tools/navmap/document_models.py`, `sitecustomize.py`).
- [ ] 1.1.3 Wire helpers (`from_payload`, `to_payload`, validators) that wrap struct usage and raise Problem Details on failure.
- [ ] 1.1.4 Update JSON Schemas under `schema/tools/**`, add normative examples, and regenerate artifacts.
- [ ] 1.1.5 Replace `dict[str, Any]` / `cast(Any, …)` in payload modules and their tests.
- [ ] 1.1.6 Add pytest coverage that round-trips each payload and asserts schema validation rejects malformed input.

### 1.2 LibCST typing support
- [ ] 1.2.1 Extend `stubs/libcst/**` with the nodes/visitors used by our codemods; ensure stubs live alongside a `py.typed` marker.
- [ ] 1.2.2 Refactor codemod helpers to construct CST nodes via typed builders (no dynamic attribute access).
- [ ] 1.2.3 Add typed pytest fixtures/utilities for parsing, applying transformers, and asserting results.
- [ ] 1.2.4 Run `uv run --no-sync pyrefly check tools/codemods` and `uv run --no-sync mypy --config-file mypy.ini tools/codemods` and record outputs.

### 1.3 Typed tooling test harness
- [ ] 1.3.1 Draft/update `tests/tools/README.md` with fixture/decorator typing guidelines for contributors.
- [ ] 1.3.2 Annotate fixtures, parametrized tests, and helper utilities across `tests/tools/**`, eliminating `Any` escapes.
- [ ] 1.3.3 Introduce helper factories (e.g., `make_error_report()`) to simplify typed construction in tests.
- [ ] 1.3.4 Execute `uv run --no-sync mypy --config-file mypy.ini tests/tools` after each directory sweep and attach logs to the PR.

### 1.4 Packaging & public exports
- [ ] 1.4.1 Audit runtime exports and ensure `tools/__init__.py` plus subpackages expose the canonical helpers.
- [ ] 1.4.2 Author stubs under `stubs/tools/**` (no `Any` fallbacks) and include `tools/py.typed` in the build.
- [ ] 1.4.3 Script a smoke test (`scripts/test-tools-install.sh`) that creates a temp venv, runs `pip install .[tools]`, imports the helpers, and runs a no-op CLI command.
- [ ] 1.4.4 Build wheels via `uv build`, unpack them, and verify `py.typed`/stubs are present.

## 2. Testing
- [ ] 2.1 Extend or add pytest coverage for payload round-trips, codemod transformations, and packaging smoke tests.
- [ ] 2.2 Run `uv run --no-sync pyrefly check` for each touched module group (`tools/_shared`, `tools/docstring_builder`, `tools/docs`, `tools/navmap`, `tools/codemods`).
- [ ] 2.3 Run `uv run --no-sync mypy --config-file mypy.ini tests/tools` and `uv run --no-sync mypy --config-file mypy.ini tools/codemods` (must pass with no ignores).
- [ ] 2.4 Execute targeted pytest suites (`tests/tools/docstring_builder`, `tests/tools/docs`, `tests/tools/navmap`, `tests/tools/shared`, `tests/tools/codemods`).

## 3. Docs & Artifacts
- [ ] 3.1 Regenerate JSON Schemas / Problem Details examples impacted by typed payload changes.
- [ ] 3.2 Run `make artifacts` and review generated docs/portal outputs for regressions.
- [ ] 3.3 Update `docs/tooling.md` (or equivalent) with a “typed payload checklist” and the install smoke-test instructions.
- [ ] 3.4 Refresh changelog / README snippets referencing tooling exports or installation guidance.

## 4. Rollout
- [ ] 4.1 Validate packaging via `scripts/test-tools-install.sh` (clean venv) and attach output to the PR.
- [ ] 4.2 Capture performance/diagnostic metrics after typed payload conversion (ensure no regression against budgets) using existing benchmark helpers.
- [ ] 4.3 Coordinate with release owners to publish the updated wheel once CI is green.

