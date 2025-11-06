## 1. Implementation

- [ ] 1.1 Create facade module.
  - Add `tools/_shared/augment_registry.py` with dataclasses (`AugmentData`, `RegistryData`, `ToolingMetadata`, `AugmentRegistryError`) and helper functions (`load_augment`, `load_registry`, `load_tooling_metadata`, `clear_cache`, `render_problem_details`).
  - Integrate safe JSON IO wrappers and Problem Details payloads.
- [ ] 1.2 Update shared CLI tooling.
  - Refactor `tools/_shared/cli_tooling.py` to consume the facade and drop duplicate JSON parsing.
  - Ensure `load_cli_tooling_context` returns `CLIConfig` derived from `ToolingMetadata`.
- [ ] 1.3 Refactor OpenAPI generator.
  - Replace augment/registry access in `tools/typer_to_openapi_cli.py` with facade calls.
  - Remove obsolete helpers, adjust imports, and verify docstrings reflect new flow.
- [ ] 1.4 Align MkDocs CLI scripts.
  - Update both the internal script and public façade to fetch operations via the facade-driven shared context.
  - Ensure tag groups and overrides originate from `ToolingMetadata`.
- [ ] 1.5 Docstring builder integration.
  - Identify augment/registry consumers in `tools/docstring_builder` (e.g., pipeline helpers) and swap to the shared facade.
  - Clean up redundant JSON loaders.
- [ ] 1.6 Documentation & linting.
  - Add module-level documentation with examples and error handling guidance.
  - Run `uv run ruff format && uv run ruff check --fix`, `uv run pyright --warnings --pythonversion=3.13`, and `uv run pyrefly check` on affected modules.

## 2. Testing

- [ ] 2.1 Unit tests for facade.
  - Add `tests/tools/test_augment_registry.py` covering success paths, missing files, malformed payloads, caching behaviour, and Problem Details emission.
- [ ] 2.2 Update existing tooling tests.
  - Adjust CLI generator, MkDocs diagram, and docstring builder tests to patch the facade or verify shared behaviour.
  - Ensure diagram tests assert tag groups/operations remain consistent.
- [ ] 2.3 Regression commands.
  - Run `pytest tests/tools -q` (including new tests) and capture outputs for PR checklist.
  - Validate Problem Details example output manually or via snapshot tests.
