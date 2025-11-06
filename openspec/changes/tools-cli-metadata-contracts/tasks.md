## 1. Implementation

- [ ] 1.1 Define Pydantic models.
  - Implement Pydantic `BaseModel` classes (`OperationOverrideModel`, `TagGroupModel`, `AugmentMetadataModel`, `RegistryInterfaceModel`, `RegistryMetadataModel`, `ToolingMetadataModel`) with `ConfigDict(frozen=True)`.
  - Add validators/serializers for tuple conversion, key normalization, and extra field handling.
- [ ] 1.2 Update facade loaders.
  - Modify `load_augment`, `load_registry`, `load_tooling_metadata` to call `model_validate` on the new models and raise `AugmentRegistryValidationError` with Problem Details on failure.
  - Adjust caching to store model instances and expose `clear_cache()` for tests.
- [ ] 1.3 Adapt shared CLI tooling.
  - Refactor `tools/_shared/cli_tooling.py` to consume the Pydantic models instead of raw dicts; update helpers accordingly.
- [ ] 1.4 Refactor OpenAPI generator.
  - Replace dictionary indexing in `tools/typer_to_openapi_cli.py` with model attribute access (`override.tags`, `augment.tag_groups`).
  - Update docstrings/examples to demonstrate typed usage.
- [ ] 1.5 Update MkDocs CLI tooling.
  - Adjust script + façade to read operations/tag groups via the new models; remove legacy dict manipulation.
- [ ] 1.6 Integrate docstring builder tooling.
  - Migrate registry access patterns in docstring pipeline modules to use `RegistryMetadataModel`.
- [ ] 1.7 Documentation & linting.
  - Document model fields in module docstrings; provide usage examples calling `.model_dump()` and handling validation errors.
  - Run `uv run ruff format && uv run ruff check --fix`, `uv run pyright --warnings --pythonversion=3.13`, and `uv run pyrefly check` on affected modules.

## 2. Testing

- [ ] 2.1 Unit tests for models.
  - Add `tests/tools/test_cli_metadata_models.py` covering valid validation, default handling, tuple conversion, `model_dump`, and Problem Details.
- [ ] 2.2 Regression tests for consumers.
  - Update CLI generator, MkDocs diagrams, and docstring tooling tests to assert outputs unchanged while using model attributes.
- [ ] 2.3 Error handling coverage.
  - Add tests ensuring missing/malformed augment and registry files raise `AugmentRegistryValidationError` with expected Problem Details content.
- [ ] 2.4 Command suite.
  - Run `pytest tests/tools -q`, capture outputs for PR checklist.
  - Perform targeted snapshot comparisons (if necessary) to confirm CLI diagrams/OpenAPI remain stable.
