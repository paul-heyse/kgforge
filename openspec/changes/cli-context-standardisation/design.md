## Design

### Context

Six separate CLI suites currently define bespoke `cli_context.py` helper modules:

| Module | Unique behaviour today |
| --- | --- |
| `src/download/cli_context.py` | Declares constants (`CLI_COMMAND`, `CLI_TITLE`, etc.), caches a single `CLIToolSettings`, and wires `CLI_OPERATION_IDS` for a small Typer tree. |
| `src/orchestration/cli_context.py` | Mirrors the download helper but includes multiple operation-id aliases (`index-bm25`, `index_bm25`). |
| `codeintel/indexer/cli_context.py` | Uses a different package prefix (`kgfoundry-codeintel`) when resolving the CLI version and exposes sequence-typed `tokens` for overrides. |
| `tools/navmap/cli_context.py` | Same pattern as download/orchestration but lives under `tools/`, uses multiple fallback distributions for version lookup, and is consumed by MkDocs tooling. |
| `tools/docstring_builder/cli_context.py` | Registers more than ten operations; docstrings emphasise the docstring tooling pipeline. |
| `docs/_scripts/cli_context.py` | Maintains a dictionary of CLI definitions (`docs-validate-artifacts`, `docs-build-symbol-index`, etc.), returning helper functions that accept a `command` parameter. |

Each module manually:

1. Resolves the repo root paths (`openapi/_augment_cli.yaml`, `tools/mkdocs_suite/api_registry.yaml`).
2. Performs package version detection via `importlib.metadata.version`, often with duplicate fallback logic.
3. Allocates and caches `CLIToolSettings` and `CLIToolingContext`.
4. Exposes nearly identical helper functions (`get_cli_settings`, `get_cli_context`, `get_cli_config`, `get_operation_context`, `get_tooling_metadata`, `get_augment_metadata`, `get_registry_metadata`, `get_interface_metadata`, `get_operation_override`).

The duplication makes it difficult to:

- Guarantee alignment with the CLI façade lifecycle (`tools/_shared/cli_runtime.py`, `tools/_shared/cli_integration.py`).
- Maintain docstrings and doctest coverage; small edits require touching each module.
- Enforce typing standards; any divergence may reintroduce Pyright/Pyrefly suppressions.

### Goals

1. Provide a single, typed registry that stores CLI metadata (command label, title, interface ID, operation IDs, optional bin name and version resolver) with deterministic caching.
2. Refactor every existing `cli_context.py` to consume the registry but retain public APIs and constants to avoid breaking Typer entry points, documentation scripts, or OpenAPI tooling.
3. Document how contributors register new CLIs through the shared registry, ensuring doctest-friendly examples and NumPy-style docstrings.
4. Deliver regression tests that cover registry behaviour, module exports, and error handling while keeping Ruff, Pyright, and Pyrefly error-free.

### Non-Goals

- Changing CLI command implementations, CLI argument surfaces, or the envelope schema.
- Introducing new CLI operations or altering Typer application structures.
- Replacing the `tools` metadata pipeline (`tools/_shared/cli_tooling.py`); the registry delegates to existing tooling functions.

### Decisions

1. **Shared registry module**
   - Create `tools/_shared/cli_context_registry.py`.
   - Define:
     - `CLIContextDefinition`: frozen `@dataclass` with fields:
       - `command: str` – logical command name used in envelopes (`download`, `orchestration`, etc.).
       - `title: str` – human-readable title displayed in docs/help text.
       - `interface_id: str` – registry interface identifier.
       - `operation_ids: Mapping[str, str]` – Typer subcommand name → canonical operation ID mapping.
       - `bin_name: str | None` – optional CLI binary label (defaults to `command` when omitted).
       - `version_resolver: Callable[[], str] | None` – lazy resolver for version discovery; defaults to a helper that checks `kgfoundry-tools`, `kgfoundry`, or CLI-specific packages.
       - `augment_path: Path | None` / `registry_path: Path | None` for overrides (defaults read from repo root).
     - `CLIContextRegistry`: responsible for registering definitions under a symbolic key (e.g., `"download"`) and exposing typed helpers:
       - `settings_for(key: str) -> CLIToolSettings`
       - `context_for(key: str) -> CLIToolingContext`
       - `augment_for(key: str) -> AugmentMetadataModel`
       - `registry_for(key: str) -> RegistryMetadataModel`
       - `interface_for(key: str) -> RegistryInterfaceModel`
       - `operation_override_for(key: str, *, subcommand: str, tokens: Sequence[str] | None = None)`
     - All helpers use caching (`functools.lru_cache` or `functools.cache`) keyed by `key` (and optional `command` parameter for overrides) to avoid repeated filesystem or metadata reads.
   - Provide `discover_repo_paths(start: Path | None = None) -> Paths` to centralise path discovery (mirrors `tools/_shared/paths.Paths.discover` but returns typed dataclass or simply reuses it).
   - Expose a module-level singleton `REGISTRY` and convenience wrappers (`register_cli`, `settings_for`, etc.) for import ergonomics.

2. **Module delegations**
   - Update each `cli_context.py` to register its definition(s) in `REGISTRY` at import time (module-level constants).
   - Replace bespoke helper implementations with thin wrappers around `settings_for`, `context_for`, etc.
   - Maintain existing exports by:
     - Keeping constants (`CLI_COMMAND`, `CLI_TITLE`, `CLI_INTERFACE_ID`, `CLI_OPERATION_IDS`) derived from the registered definition.
     - Preserving `__all__` lists to avoid breaking star-import consumers.
     - Retaining NumPy-style docstrings but referencing the shared registry in explanatory text.
   - For modules supporting multiple CLIs (`docs/_scripts/cli_context.py`):
     - Register each CLI under a stable key (e.g., `"docs-validate-artifacts"`, `"docs-build-symbol-index"`, `"docs-build-graphs"`).
     - Implement helper functions (`get_cli_settings(command: str = DEFAULT)`) that delegate to `settings_for(command)` and validate allowed values.

3. **Version resolution**
   - Provide utility functions within the registry:
     - `default_version_resolver(packages: Sequence[str]) -> Callable[[], str]` that attempts `importlib.metadata.version` for each package name, returning `"0.0.0"` when all lookups fail.
     - CLI-specific modules supply package preference orders (e.g., `("kgfoundry-tools", "kgfoundry")`, or `("kgfoundry-codeintel", "kgfoundry")`).

4. **Error handling and messaging**
   - Registry lookups raise `KeyError` with descriptive messages (include the unknown key, a sorted list of known keys, and remediation guidance).
   - Operation overrides return `None` when not present rather than raising (mirrors current behaviour).
   - Docstrings describe error semantics and reference façade requirements (e.g., structured logging, deterministic envelopes).

5. **Testing strategy**
   - New test module `tests/tools/test_cli_context_registry.py` covers:
     - Successful registration and retrieval of definitions with mocked augment/registry paths (use `tmp_path` fixtures).
     - Caching behaviour (calling `context_for` twice returns same object id).
     - Error path for unknown CLI key.
     - Operation override resolution for CLIs with overrides.
     - Multi-definition modules (docs script) verifying default selection and keyed lookups.
   - Update existing smoke test `tests/test_cli_runtime.py` (or create `tests/test_cli_contexts.py`) to import each CLI module and assert helper functions return expected metadata types/values.
   - Ensure doctest/xdoctest runs for CLI modules by adding targeted tests if necessary (`pytest --doctest-glob` configuration already exists).

### Detailed Plan

1. **Inventory current helpers**
   - Document field values (command labels, interface IDs, operation mappings) for each CLI context to ensure registration table accuracy.
   - Note differences in version resolution logic and tokens parameters (e.g., `docs/_scripts` uses `_CLIDefinition` dataclass; capture that structure for migration).

2. **Implement shared registry**
   - Create `CLIContextDefinition` with dataclass-level validation (strip/normalise command labels, ensure operation IDs map to non-empty strings).
   - Implement `CLIContextRegistry` with:
     - Internal dictionary storing definitions keyed by string.
     - `_cache_settings`, `_cache_context`, etc., using `functools.lru_cache(maxsize=None)` decorated inner functions.
     - Methods that leverage existing tooling utilities:
       - `CLIToolSettings` construction delegates to `tools.load_cli_tooling_context`.
       - Use `Paths.discover()` to resolve repo root for default augment/registry paths unless overrides provided.
   - Provide simple instrumentation (structured log message) when `register_cli` overwrites an existing key (should raise to avoid silent overrides).

3. **Refactor CLI modules**
   - For each module:
     1. Replace manual constants with values pulled from `CLIContextDefinition`.
     2. Register definition(s) at import time.
     3. Rewrite helper functions to delegate, keeping docstrings updated to describe registry usage and referencing façade obligations (logs, metrics, envelopes).
     4. Ensure `__all__` remains sorted and contains the same exports as pre-refactor.
     5. Remove now-unused imports (`importlib.metadata`, caching decorators) and re-run Ruff formatting.
   - For `docs/_scripts/cli_context.py`, adapt existing `_CLIDefinition` dataclass into registry definitions and tighten parameter validation (raise `KeyError` or `ValueError` for unknown CLI names).

4. **Testing & validation**
   - Add new regression tests for the registry (parametrised by CLI key).
   - Update existing smoke tests to assert wrapper functions return `CLIToolingContext`/`CLIToolSettings` and maintain caching semantics.
   - Ensure doctests still execute (update docstrings to keep runnable examples minimal; e.g., demonstrate retrieving settings via registry).
   - Run lint/type/test suite locally; capture outputs for PR checklist.

5. **Documentation**
   - Update or create contributor doc (`docs/contrib/howto_cli.md` or new doc) explaining how to register a CLI with `register_cli`, choose package names for version detection, and expose wrappers.
   - Mention registry in CLI integration high-level plan if cross-referenced.

### Risks & Mitigations

- **Registry misconfiguration** (incorrect command key or file path) could break CLI tooling.
  - *Mitigation*: Provide strong validation in `register_cli` (assert key uniqueness, non-empty command, existing files), add regression tests per CLI, and ensure `KeyError` messages point to contributor docs.
- **Backwards-incompatible exports** might slip in during refactor.
  - *Mitigation*: Compare `__all__` before/after, add tests importing old names, and include review checklist verifying module-level API parity.
- **Performance regressions** due to repeated metadata loading.
  - *Mitigation*: Use caching for settings/context retrieval; incorporate tests verifying caches are used (e.g., measure call counts with monkeypatch).
- **Docstring or doctest drift**.
  - *Mitigation*: Keep examples short, rely on `pytest --doctest` to catch regressions, and document required imports.

### Migration Plan

1. Implement and test the registry module in isolation (unit tests + type checks).
2. Sequentially refactor each CLI context module, running `uv run ruff check` and targeted tests after each to maintain a clean state.
3. Add/extend tests to cover registry usage and CLI module wrappers.
4. Update contributor documentation and high-level plans to reference the registry.
5. Execute full validation suite (`ruff`, `pyright`, `pyrefly`, `pytest`, `make artifacts`, `openspec validate cli-context-standardisation --strict`).
6. Archive the change once CI passes and documentation has been updated.

