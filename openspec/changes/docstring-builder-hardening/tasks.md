## 0. Orientation (AI agent submission pre-flight)
- [x] 0.1 Confirm the ephemeral worker image already matches the repo’s pinned toolchain (Python 3.13.9 + `uv`).
- [x] 0.2 Download and review `openspec/changes/docstring-builder-hardening/` (proposal, design, spec) plus `openspec/AGENTS.md`; these instructions must be cached locally for reference during execution.
- [x] 0.3 Read `tools/docstring_builder/models.py` and `schema/tools/docstring_builder_cli.json` to internalise the typed contracts and schema identifiers prior to coding.
- [x] 0.4 Review the capability scenarios in `specs/docstring-tooling/spec.md`, especially the new Problem Details requirement, so implementation steps map back to acceptance.
- [x] 0.5 Record baseline `uv run ruff check tools/docstring_builder` and `uv run mypy tools/docstring_builder` outputs to prove progress after implementation.

## 1. Planning & Alignment (automated context sync)
- [x] 1.1 Cache supporting specs locally:`docs/_build/schema_docfacts.json` and the new CLI schema. These files should be mounted into the agent workspace before execution.
- [x] 1.2 Ingest `design.md` sections “Detailed Implementation Plan”, “Typed Models Outline”, and “CLI Schema Definition”; persist key action items into the agent’s prompt/context store.
- [x] 1.3 Register the feature flag strategy (`DOCSTRINGS_TYPED_IR`) and rollout expectations in the job metadata so later stages (tests, toggles) can reference them without re-reading the docs.
- [x] 1.4 Prepare a concise execution note (Summary, API sketch using the typed contracts, Data contracts touched, Test plan) to submit alongside the job; no human approval loop is expected, so the note acts as self-audit before coding.

## 2. Adopt Typed Models & Enforce Schemas
- [x] 2.1 Introduce adapter functions that convert legacy harvest/policy/render structures into `DocstringIR*` dataclasses (`tools/docstring_builder/models.py`).
- [x] 2.2 Replace `dict[str, Any]` payloads across builder modules with the typed dataclasses/TypedDicts; remove legacy helper signatures that return `Any`.
- [x] 2.3 Implement `validate_docfacts_payload(payload: DocfactsDocumentPayload) -> None` and `validate_cli_output(payload: CliResult) -> None`, raising `SchemaViolationError` with Problem Details context.
- [x] 2.4 Wire the validators into the builder pipeline behind `DOCSTRINGS_TYPED_IR=1`, ensuring dry-run mode logs warnings rather than aborting.
- [x] 2.5 Create pytest suites (`tests/tools/docstring_builder/test_schemas.py`) covering round-trip serialization, negative cases, and Problem Details formatting.
- [x] 2.6 Update docs to record schema versions and bump `CLI_SCHEMA_VERSION` when fields change; regenerate schema artifacts (`docstring_builder schema`). _(No schema changes required in this patch; version remains 1.0.0.)_

## 3. Refactor Core Modules
- [ ] 3.1 Extract helper modules (`normalizer_signature.py`, `normalizer_annotations.py`) or internal functions as outlined in design doc Section “Detailed Implementation Plan”.
- [ ] 3.2 Refactor `policy.py` `_apply_mapping` into pure helper functions with deterministic ordering; add unit tests covering new branches.
- [ ] 3.3 Split `render.py` `_build_signature` into composable units and enable Jinja `autoescape=True` (or `select_autoescape`).
- [ ] 3.4 Replace broad `except Exception` blocks across builder modules with targeted exceptions (e.g., `ImportError`, `TypeError`); rethrow as custom `DocstringBuilderError` hierarchy with `raise ... from e`.
- [ ] 3.5 Update module-level docstrings and type hints to meet NumPy docstring + strict typing requirements while returning typed models.

## 4. Plugin Architecture
- [ ] 4.1 Define `DocstringBuilderPlugin` `Protocol` in `plugins/base.py` (or similar) with explicit method signatures and typed payloads.
- [ ] 4.2 Update bundled plugins (`dataclass_fields.py`, `llm_summary.py`, `normalize_numpy_params.py`, others) to satisfy the Protocol, removing `Any` and blind exceptions.
- [ ] 4.3 Add regression test (`tests/tools/docstring_builder/test_plugins.py`) covering the dataclass variance bug, plugin opt-in/out behaviors, and error wrapping.
- [ ] 4.4 Provide compatibility shim that adapts legacy call signatures, emits `DeprecationWarning`, and document removal timeline in changelog.
- [ ] 4.5 Update developer documentation (`docs/contributing/quality.md`) with a plugin authoring guide referencing the new Protocol and schemas.

## 5. CLI & Shared Infrastructure
- [ ] 5.1 Add `tools/_shared/logging.py` with `get_logger(name: str) -> logging.Logger` (NullHandler, structured logging helpers).
- [ ] 5.2 Add `tools/_shared/proc.py` with `run_tool(...)` enforcing absolute executables, sanitized env, timeouts, and Problem Details error translation.
- [ ] 5.3 Update `docstring_builder/cli.py` to construct responses via `build_cli_result_skeleton`, populate typed fields, and emit Problem Details JSON on failure.
- [ ] 5.4 Emit typed machine outputs for `--json` (current) and placeholder structure for `--baseline`; validate against `schema/tools/docstring_builder_cli.json` before writing.
- [ ] 5.5 Add CLI integration tests under `tests/tools/docstring_builder/test_cli.py` (parametrized for success/failure and feature-flag combinations) asserting schema validation.
- [ ] 5.6 Provide a `docs/examples/docstring_builder_problem_details.json` fixture used in doctests and developer guides.

## 6. Observability & Performance
- [ ] 6.1 Instrument builder with operation timers, counters, and trace spans (OTel); update design doc with metric names and trace spans.
- [ ] 6.2 Add structured logging fields (`operation`, `symbol_id`, `duration_ms`, `schema_version`) and verify via unit tests or captured logs.
- [ ] 6.3 Provide doctest verifying sample Problem Details JSON (see `design.md` appendix) and ensure doctest executes via pytest.
- [ ] 6.4 Run micro-benchmark (`tests/perf/test_docstring_builder_bench.py`) to confirm refactors stay within runtime budget; document results in PR.

## 7. Docs & Communication
- [ ] 7.1 Update `docs/contributing/quality.md` with new builder workflow, feature flag usage, and schema validation steps.
- [ ] 7.2 Add `docs/howto/docstring_builder_migration.md` describing typed IR changes, plugin Protocol, and troubleshooting tips.
- [ ] 7.3 Regenerate Agent Portal artifacts (`make artifacts`) to ensure new metadata fields surface correctly; attach screenshots/links to PR.
- [ ] 7.4 Provide migration guide for third-party plugins (README update + changelog entry) and announce via internal comms channel.

## 8. Acceptance Gates (attach outputs in PR)
```bash
uv run ruff format && uv run ruff check --fix
uv run pyrefly check
uv run mypy --config-file mypy.ini
uv run pytest -q
make artifacts && git diff --exit-code
openspec validate docstring-builder-hardening --strict
```

## 9. Sign-off
- [ ] Domain owner review (Doc Tooling)
- [ ] Implementation owner review
- [ ] CI green; docs/portal artifacts uploaded (coverage, JUnit, docs, portal, schema validation logs)
- [ ] Feature flag default flipped and compatibility shim timeline documented
