## Why

The tools package still experiences recursive imports, implicit public APIs, and dynamically-typed entry points that undermine clarity, testability, and OpenSpec compliance. Prior tooling hardening (phase 3) focused on surface linting and typed factories for tests, but the runtime modules retain `Any`-heavy code paths, oversized orchestration scripts, and uneven observability. A holistic refactor is required to align with principles 1–16, provide predictable schemas, and unblock strict pyrefly/mypy gates for both library and CLI consumers.

## What Changes

- **Governance & Baseline (Principles 1, 7, 11, 14)**
  - Restructure `tools/__init__.py` and `tools/docs/__init__.py` to replace recursive imports with explicit module references, build curated `__all__` exports, and provide PEP 257 docstrings that cite the shared exception taxonomy and the representative Problem Details payload at `schema/examples/tools/problem_details/tool-execution-error.json`.
  - Split monolithic adapters such as `tools/docstring_builder/cli.py` into orchestrator modules (`tools/docstring_builder/orchestrator.py`, `tools/docstring_builder/io.py`, `tools/docstring_builder/paths.py`) with thin CLI shims, and update `tools/make_importlinter.py` to emit layered contracts for docstring builder, docs, and navmap packages enforced via `python tools/make_importlinter.py --check`.
- **Type Safety & Data Contracts (Principles 1, 2, 4, 16)**
  - Introduce msgspec/dataclass domain models for CLI envelopes, docstring edits, navmap documents, and analytics payloads (including new `tools/docs/catalog_models.py`), replacing `dict[str, Any]` usage and providing migration helpers for legacy payloads.
  - Enhance `tools/_shared/schema.py` with `render_schema` / `write_schema` utilities that emit JSON Schema 2020-12 definitions straight from the models, and centralise validation via `validate_struct_payload` leveraging `kgfoundry_common.serialization.validate_payload`.
  - Provide typed facades and stubs for optional dependencies (`pydot`, `libcst`, etc.) under `stubs/tools/**`, eliminating redundant `# type: ignore` pragmas.
  - Normalise time/path usage on `pathlib.Path`, timezone-aware `datetime`, and `decimal.Decimal` throughout tooling manifests and analytics outputs.
- **Error Handling, Logging, Observability, Idempotency (Principles 5, 8, 9, 15)**
  - Harden `tools._shared.proc.run_tool` with a shared `ContextVar`-backed operation ID, structured logging, and Problem Details-rich exceptions; introduce `run_tool_with_retry` for idempotent operations.
  - Extend `tools._shared.metrics.observe_tool_run` with injectable Prometheus/OpenTelemetry factories so every tool invocation records counters/histograms/spans annotated with operation and correlation metadata.
  - Consolidate the exception taxonomy in `tools/_shared/exceptions.py`, ensuring orchestrators and adapters emit RFC 9457 Problem Details (including retry hints) for all failure paths.
- **Security, Configuration, & Supply Chain (Principles 6, 10)**
  - Promote `tools._shared.settings` to namespace-aware `pydantic_settings` classes (e.g., `DocbuilderSettings`, `DocsSettings`) with eager validation and structured `SettingsError` payloads.
  - Centralise input sanitisation through helpers like `require_workspace_file` / `validate_allowed_url`, rejecting traversal or unsafe schemes before orchestration occurs.
  - Introduce `tools._shared.security` wrappers for `uv run pip-audit` and secret scans, replacing unsafe YAML/JSON loading with schema-backed helpers in `tools._shared.serialization`.
- **Verification Loop**
  - After each refactor tranche, execute the canonical gate sequence: run `uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini` and confirm that all code blocks you made edits to are error free. 

## Impact

- **Affected specs:** `openspec/specs/tools-runtime`, `openspec/specs/tools-navmap`, and related schema entries under `schema/tools/**` will require updates to reflect the new typed models and observability guarantees.
- **Affected code:** `tools/__init__.py`, `tools/docs/__init__.py`, `tools/docstring_builder/**`, `tools/_shared/**`, `tools/navmap/**`, `tools/docs/**`, `stubs/tools/**`, and documentation under `docs/tools/**`.
- **Rollout:** Perform changes incrementally behind the verification loop; introduce migration notes and deprecation warnings where public APIs evolve (notably the new orchestrator modules and curated exports). No feature flags are required, but publish SemVer bumps for the tools package when exported symbols change.

