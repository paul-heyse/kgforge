## 1. Implementation

- [x] 1.1 **Shared registry scaffolding**
  - [x] 1.1.1 Create `tools/_shared/cli_context_registry.py` and add module docstring summarising intent plus façade references.
  - [x] 1.1.2 Implement `CLIContextDefinition` (frozen dataclass) with validation (`__post_init__` ensures non-empty command/interface_id, normalises operation IDs).
  - [x] 1.1.3 Implement `default_version_resolver(*package_names: str) -> Callable[[], str]`.
  - [x] 1.1.4 Implement `CLIContextRegistry` with methods: `register`, `settings_for`, `context_for`, `augment_for`, `registry_for`, `interface_for`, `operation_override_for`.
  - [x] 1.1.5 Add caching helpers (decorated inner functions) so repeated lookups reuse objects; unit test coverage planned in §2.
  - [x] 1.1.6 Expose module-level singleton `REGISTRY`, `register_cli`, and thin function wrappers for convenience.

- [x] 1.2 **Register existing CLIs**
  - [x] 1.2.1 Populate registry entries for:
    - `download` (bin: `kgf`, package fallback: `kgfoundry`)
    - `orchestration` (bin: `kgf`, interface `orchestration-cli`, operation aliases for `index-bm25`, `index_bm25`, etc.)
    - `codeintel` (bin: `kgf-codeintel`, package fallback order `kgfoundry-codeintel`, `kgfoundry`)
    - `navmap` (bin: `tools-navmap`, package fallback `kgfoundry-tools`, `kgfoundry`)
    - `docstrings` (bin: `docstring-builder`, package fallback `kgfoundry-tools`, `kgfoundry`)
    - documentation CLIs (`docs-validate-artifacts`, `docs-build-symbol-index`, `docs-build-graphs`) with explicit bin names.
  - [x] 1.2.2 Document registrations by co-locating definitions alongside each CLI module so traceability is preserved.

- [x] 1.3 **Refactor CLI modules**
  - [x] 1.3.1 `src/download/cli_context.py`
    - Replace manual `get_cli_settings`/`get_cli_context` implementations with registry wrappers.
    - Ensure `CLI_COMMAND`, `CLI_TITLE`, `CLI_INTERFACE_ID`, `CLI_OPERATION_IDS` derive from registry definition.
    - Update docstring to reference registry usage; keep doctest example (call `get_cli_settings()` and assert `bin_name`).
  - [x] 1.3.2 `src/orchestration/cli_context.py`
    - Mirror download changes; ensure operation aliases preserved and docstring outlines alias behaviour.
  - [x] 1.3.3 `codeintel/indexer/cli_context.py`
    - Replace version resolution logic with `default_version_resolver("kgfoundry-codeintel", "kgfoundry")`.
    - Keep behaviour for `tokens: Sequence[str] | None` in `get_operation_override`; delegate to registry.
  - [x] 1.3.4 `tools/navmap/cli_context.py`
    - Delegate functions/constants to registry; ensure docstring emphasises tooling usage and path discovery.
  - [x] 1.3.5 `tools/docstring_builder/cli_context.py`
    - Register large operation mapping; update docstrings referencing registry.
  - [x] 1.3.6 `docs/_scripts/cli_context.py`
    - Remove `_CLIDefinition` dataclass in favour of registry definitions.
    - Implement helper functions accepting `command: str`, validate via registry (`KeyError` -> helpful message).
    - Update docstring example to show retrieving two different CLI settings.

- [x] 1.4 **Clean up imports & module metadata**
  - [x] 1.4.1 Ensure each module retains `from __future__ import annotations` as first import.
  - [x] 1.4.2 Remove redundant imports (`importlib.metadata`, `lru_cache`, etc.) replaced by registry.
  - [x] 1.4.3 Confirm `__all__` lists stay identical (sorted and containing same symbols).
  - [x] 1.4.4 Update module-level constants to remain public for docs/typer consumers.

- [ ] 1.5 **Developer documentation**
  - [ ] 1.5.1 Update `docs/contrib/howto_cli.md` (or create if missing) with section “Registering CLIs via the shared context registry”.
  - [ ] 1.5.2 Reference the registry in `openspec/high-level-plans/CLI_Integration_Implementation_Plan.md` if required.

## 2. Testing & Validation

- [x] 2.1 **Unit tests for registry**
  - [x] 2.1.1 Add `tests/tools/test_cli_context_registry.py`.
  - [x] 2.1.2 Cover registration success and duplicate-key error.
  - [x] 2.1.3 Verify `settings_for`/`context_for` return cached instances (use `monkeypatch` to count load calls).
  - [x] 2.1.4 Test `operation_override_for` with and without overrides.
  - [x] 2.1.5 Test `KeyError` messaging for unknown CLI key.

- [x] 2.2 **Integration smoke tests**
  - [x] 2.2.1 Extend CLI façade smoke test (`tests/test_cli_runtime.py` or new `tests/test_cli_context_modules.py`) to import each CLI module and assert wrappers return expected types (e.g., `CLIToolSettings.bin_name` equals expected).
  - [x] 2.2.2 Validate docs CLI module’s multi-command behaviour via parametrised tests.

- [ ] 2.3 **Doctest coverage**
  - [ ] 2.3.1 Ensure doctest/xdoctest runs include updated modules; add targeted doctest files if necessary.
  - [ ] 2.3.2 Confirm examples execute without external dependencies.

- [x] 2.4 **Static analysis & formatting**
  - [x] 2.4.1 Run `uv run ruff format && uv run ruff check --fix`.
  - [x] 2.4.2 Run `uv run pyright --warnings --pythonversion=3.13`.
  - [x] 2.4.3 Run `uv run pyrefly check`.
  - [x] 2.4.4 Address any findings before continuing.

- [x] 2.5 **Test suite & CLI smoke**
  - [x] 2.5.1 Execute `uv run pytest -q`.
  - [ ] 2.5.2 (Optional) Run representative CLI commands (e.g., `python -m src.download.cli --help`) to confirm contexts still load without runtime errors.

## 3. Docs, Spec, & Validation

- [ ] 3.1 Regenerate relevant documentation / nav artifacts (`make artifacts && git diff --exit-code`) if contributor docs updated.
- [ ] 3.2 Validate OpenSpec change (`openspec validate cli-context-standardisation --strict`).
- [ ] 3.3 Prepare PR checklist evidence: outputs of `ruff`, `pyright`, `pyrefly`, `pytest`, `make artifacts`, and `openspec validate`.

