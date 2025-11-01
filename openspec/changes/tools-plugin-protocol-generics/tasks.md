## 1. Implementation
- [ ] **1.1 Formalize the generic plugin contract**
  - [ ] Introduce `TypeVar` pairs (e.g., `InputT`, `OutputT`) in `tools/docstring_builder/plugins/base.py`, redefine `DocstringBuilderPlugin` as `Protocol[InputT, OutputT]`, and specialise `HarvesterPlugin`, `TransformerPlugin`, and `FormatterPlugin` with `HarvestResult`, `SemanticResult`, and `DocstringEdit` respectively.
  - [ ] Ensure every public Protocol/class in `plugins/base.py` carries a PEP 257 docstring with a single-sentence summary, explicit `__all__`, and fully annotated signatures (PEP 695 generics where possible).
  - [ ] Document contract-level exceptions, confirming that plugin failures ultimately surface `PluginExecutionError` consistent with `schema/examples/tools/problem_details/tool-execution-error.json`.

- [ ] **1.2 Rebuild the plugin manager around generics**
  - [ ] Refactor `PluginManager` fields and methods in `tools/docstring_builder/plugins/__init__.py` to use the new generic Protocols, eliminating `cast(...)` and `pyrefly` suppressions in `_invoke_apply`, `_ensure_plugin_instance`, and `_register_plugin`.
  - [ ] Introduce helper type aliases or generic functions so stage pipelines remain type-sealed (harvesters return `HarvestResult`, etc.) and keep iteration logic clear and side-effect free.
  - [ ] Hook the module into `tools._shared.logging.get_logger` to emit structured error logs (with plugin name, stage, and file path) when raising `PluginExecutionError`, and ensure a `NullHandler` is attached at import time.

- [ ] **1.3 Harden legacy adapter and compatibility shims**
  - [ ] Update `LegacyPluginAdapter` to implement `DocstringBuilderPlugin[DocstringPayload, DocstringPayload]`, provide overloads for stage-specific wrapping, and retain the one-time deprecation warning (using `warnings.warn(..., stacklevel=3)`).
  - [ ] Add typed guards/TypeGuards so `_ensure_plugin_instance` recognises legacy plugins without falling back to `Any`, and verify all branches raise typed configuration errors with preserved exception causes (`raise ... from e`).

- [ ] **1.4 Migrate built-in plugins to the new contract**
  - [ ] Update `DataclassFieldDocPlugin`, `LLMSummaryRewritePlugin`, `NormalizeNumpyParamsPlugin`, and any other shipped plugins under `tools/docstring_builder/plugins/` to subclass or register against the new generics, adding explicit annotations for `apply`, `on_start`, and `on_finish`.
  - [ ] Review each module’s public surface (`__all__`, docstrings, logger usage) for PEP 8 naming, type completeness, and structured logging alignment; ensure no module leaks mutable globals.
  - [ ] Where plugins emit data crossing boundaries (e.g., docstring edits), reconfirm the payload conforms to existing msgspec models and documented Problem Details examples.

- [ ] **1.5 Align orchestration and legacy pipelines**
  - [ ] Update `tools/docstring_builder/orchestrator.py`, `tools/docstring_builder/legacy.py`, and `tools/docstring_builder/__init__.py` to consume the generic `PluginManager` without casts, ensuring CLI/status reporting continues to use typed models and RFC 9457 envelopes.
  - [ ] Verify retry/idempotency semantics in orchestrator flows remain intact and documented, leveraging existing `tools._shared.proc` helpers for subprocess execution.
  - [ ] Remove any lingering `pyrefly` or `mypy` ignores introduced for plugin payloads, replacing them with precise typing or helper utilities.

- [ ] **1.6 Refresh stubs and type-checker surfaces**
  - [ ] Regenerate or hand-edit `stubs/tools/docstring_builder/plugins/*.pyi` (and related stub files) to mirror the new generic signatures so external consumers receive accurate typing information.
  - [ ] Update `stubs/tools/docstring_builder/__init__.pyi` and top-level exports to reflect the refined API, keeping annotations in sync with the runtime modules.
  - [ ] Review `mypy.ini` / `pyrefly.toml` to drop suppressions for plugin modules and ensure the strict baseline enforces the new contract.

## 2. Testing
- [ ] 2.1 Run `uv run ruff check --fix`.
- [ ] 2.2 Run `uv run pyrefly check` and confirm suppressions for `tools/docstring_builder/plugins/**` are removed.
- [ ] 2.3 Run `uv run mypy --config-file mypy.ini` with focus on `tools/docstring_builder`.


