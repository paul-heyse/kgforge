## 1. Implementation

- [ ] 1.1 Scaffold shared tooling module.
  - Create `tools/_shared/cli_tooling.py` with dataclasses (`AugmentConfig`, `RegistryContext`, `CLIToolingContext`, `CLIToolSettings`) and custom exception `CLIConfigError`.
  - Implement `load_augment_config`, `load_registry_context`, `build_cli_config`, and `load_cli_tooling_context`, each emitting structured logs and raising Problem Details on failure.
- [ ] 1.2 Harden IO helpers.
  - Wrap JSON reads via existing safe loader (`docs._scripts.shared.safe_json_deserialize`) and provide dependency injection hooks for tests.
  - Memoise successful loads (LRU keyed by `(augment_path, registry_path)`).
- [ ] 1.3 Refactor CLI generator.
  - Replace inline augment/registry logic in `tools/typer_to_openapi_cli.py` with calls to `load_cli_tooling_context`.
  - Update argument parsing to populate `CLIToolSettings` and pass the resulting `CLIConfig` into `make_openapi`.
  - Remove obsolete helpers (`_load_registry`, `_ensure_str_list` clones) now provided by the shared module.
- [ ] 1.4 Align MkDocs CLI diagram tooling.
  - Update `tools/mkdocs_suite/docs/_scripts/gen_cli_diagram.py` and `tools/mkdocs_suite/docs/cli_diagram.py` to consume the shared context (no direct imports from internal generator modules).
  - Ensure tag groups and `x-cli` metadata derive from `CLIToolingContext`.
- [ ] 1.5 Documentation & linting.
  - Add NumPy-style docstrings with runnable examples to the shared module.
  - Update developer docs (if applicable) pointing to the new shared helpers.
  - Run `uv run ruff format && uv run ruff check --fix` and `uv run pyright tools` to guarantee cleanliness after refactor.

## 2. Testing

- [ ] 2.1 Unit tests for shared module.
  - Add `tests/tools/test_cli_tooling.py` covering success, missing files, malformed augment, registry cache hits, and dependency injection behaviour.
- [ ] 2.2 Update MkDocs diagram tests.
  - Adjust `tests/tools/mkdocs_suite/test_gen_cli_diagram.py` to patch `load_cli_tooling_context`, ensuring diagram operations come from the shared context.
  - Add regression asserting tag deduplication and optional `operationId` anchor handling still behave as expected.
- [ ] 2.3 CLI generator smoke tests.
  - If existing tests cover `make_openapi`, enhance them to call through the shared loader (mocking file reads) and verify consistent operations.
- [ ] 2.4 Tools quality gates.
  - Run `uv run pyrefly check tools` (or scoped to affected modules) and `uv run pyright --warnings --pythonversion=3.13 tools` to enforce typing.
  - Execute targeted pytest suite `pytest tests/tools -q` and capture outputs for the PR checklist.
