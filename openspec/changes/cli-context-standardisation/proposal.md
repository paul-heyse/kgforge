## Why
Every CLI suite currently maintains bespoke context helpers that replicate path discovery, version detection, and metadata wiring. The divergence causes docstring drift, inconsistent typing coverage, and unreliable façade behaviour (metrics, envelopes, OpenAPI). Consolidating these helpers into a single registry is required to complete the CLI integration plan and uphold the zero-error mandate in `AGENTS.md`.

## What Changes
- [ ] **ADDED**: `cli-context-standardisation` capability describing the shared registry contract and validation requirements.
- [ ] **ADDED**: `tools/_shared/cli_context_registry.py` implementing:
  - `CLIContextDefinition` with validation hooks and version resolver support.
  - `CLIContextRegistry` providing cached accessors for settings, contexts, metadata, and overrides.
- [ ] **ADDED**: registration table covering all existing CLIs (download, orchestration, codeintel, navmap, docstrings, docs tooling) plus helper utilities for new registrations.
- [ ] **MODIFIED**: each `cli_context.py` delegates to the registry while preserving constants, public helper signatures, doctest examples, and `__all__`.
- [ ] **ADDED**: regression tests (`tests/tools/test_cli_context_registry.py`, extended façade smoke tests) validating registry caching, error handling, and per-module integration.
- [ ] **ADDED**: contributor documentation describing registry usage and onboarding steps for new CLIs.

## Impact
- **Specs**: new capability; existing CLI specs reference the registry as the canonical metadata source.
- **Code**: shared tooling (`tools/_shared`), six CLI context modules, contributor docs, and associated tests.
- **Data contracts**: envelope schema unchanged; documentation updated to reflect registry-driven metadata.
- **Rollout**: implemented as a coordinated change—registry scaffolding lands first, followed by per-module adoption and tests. Façade smoke tests and registry unit tests must pass before merge.

## Rollout / Dependencies
- Depends on existing façade modules (`cli_runtime`, `cli_integration`, `cli_tooling`) already present in the repository.
- Must precede façade adoption scanners to ensure consistent metadata is available across CLIs.
- Requires documentation updates and OpenSpec validation prior to requesting review (`openspec validate cli-context-standardisation --strict`).

