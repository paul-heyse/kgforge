## Why

The documentation toolchain (`docs/conf.py` plus the generators in `docs/_scripts/`) still relies on
dynamic typing, ad-hoc JSON construction, and permissive logging that bypass our production quality
gates. Outstanding Ruff, Pyrefly, and pyright violations illustrate broader issues: configuration is
scattered across environment lookups, Griffe loaders are instantiated with `Any`, data payloads lack
schemas, and failure paths do not emit the RFC 9457 envelopes mandated elsewhere in the stack. The
resulting drift blocks strict type enforcement, complicates observability, and risks breakage for the
Agent Catalog and Portal builds.

## What Changes

- **Baseline & Configuration Governance**
  - Centralise environment handling, path wiring, and logging adapters inside
    `docs/_scripts/shared.py`, making all downstream scripts depend on typed settings objects rather
    than raw `os.environ` access.
  - Document and harden optional integrations (Pydantic, auto-docstrings, Griffe) via protocols and
    guarded imports so missing dependencies fail fast and stay type-safe.
  - Cache environment/settings discovery, expose navmap/GitHub metadata via shared helpers, and supply
    pre-configured `LoggerAdapter` factories so scripts consistently include operation/artifact context.
- **Type Safety & Modularity**
  - Replace residual `Any` usage in `docs/_scripts/build_symbol_index.py`, `docs/_scripts/mkdocs_gen_api.py`,
    and `docs/_scripts/symbol_delta.py` with typed protocols, dataclasses, and helper functions that wrap
    Griffe objects, JSON parsing, and file IO.
  - Extract shared helpers for member traversal, signature rendering, and permalink assembly to enforce
    single-responsibility and enable reuse across scripts.
  - Model MkDocs generation as pure `RenderedPage` objects, encapsulate delta/index artifacts in dataclasses,
    and standardise CLI/build entry points around structured `ToolExecutionError` handling.
- **Data Contracts & Validation**
  - Introduce JSON Schema 2020-12 definitions for the symbol index and delta payloads, publish canonical
    examples under `schema/docs/`, and validate generated artifacts before writing to disk.
  - Ensure docs build tasks serialize structured data exclusively through schema-aware helpers and record
    schema versions for drift detection.
  - Ship a reusable schema validation helper plus a `docs/_scripts/validate_artifacts.py` command that hooks
    into `make artifacts`, blocking drift across generated payloads.
- **Logging, Errors, and Observability**
  - Route subprocess and git interactions through `tools._shared.proc.run_tool`, attaching correlation
    metadata via `with_fields` and emitting RFC 9457 Problem Details on failure.
  - Replace broad `except Exception` blocks in Sphinx hooks with targeted handling that surfaces actionable
    error details and fallback behaviour.
  - Record metrics/tracing via shared adapters (contextvars-driven correlation IDs, `observe_tool_run`) so
    docs builds are observable across automation pipelines.
- **Verification Loop**
  - Mandate Ruff, Pyrefly, pyright, schema validation

## Impact

- **Affected specs:** new capability specification `docs-toolchain` capturing schema-backed, type-safe docs
  generation obligations.
- **Affected code:** `docs/conf.py`, `docs/_scripts/shared.py`, `docs/_scripts/build_symbol_index.py`,
  `docs/_scripts/mkdocs_gen_api.py`, `docs/_scripts/symbol_delta.py`, and supporting schema directories.
- **Rollout:** Ship iteratively behind the shared helper and schema validation work. No feature flags are
  required, but all scripts must pass strict Ruff/Pyrefly/pyright gates

