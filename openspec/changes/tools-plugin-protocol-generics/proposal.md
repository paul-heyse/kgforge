## Why
The docstring-builder plugin stack currently relies on pyrefly suppressions where stage-specific Protocols narrow `DocstringBuilderPlugin.apply`. That mismatch prevents static checkers from proving soundness and hides regressions in third-party plugins. Tightening the typing surface unlocks higher assurance for the tooling suite and completes the static-checker hardening workstream the user requested.

## What Changes
- Redesign `DocstringBuilderPlugin` and the stage-specific Protocols as fully generic interfaces so harvester, transformer, and formatter plugins advertise the payload types they accept and return.
- Update the plugin manager, legacy adapter, and built-in plugins to propagate those generics, removing `pyrefly: ignore` suppressions and eliminating `Any` flows in orchestration code.
- Refresh stubs and contributor guidance so external plugins can adopt the stronger contract without runtime surprises.
- Extend lint/type coverage to assert that `uv run pyrefly check` and `uv run pyright --warnings --pythonversion=3.13` pass with no suppressions for `tools/docstring_builder/plugins/**`.


## Impact
- **Affected specs:** `tools-suite`
- **Affected code:** `tools/docstring_builder/plugins/**`, `tools/docstring_builder/orchestrator.py`, `tools/docstring_builder/legacy.py`, `tools/docstring_builder/__init__.py`, stubs under `stubs/tools/docstring_builder/**`
- **Rollout:** Type-only refactor; no new runtime behaviour. Requires coordinated update to plugin documentation so downstream contributors can adjust implementations before the next release.

