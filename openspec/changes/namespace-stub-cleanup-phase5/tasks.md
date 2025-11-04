## 1. Implementation
- [ ] 1.1 Capture current namespace exports.
  - Run `python -m kgfoundry` (or a dedicated script) to list `_namespace_proxy` exports; export them to a scratch file for comparison.
  - Use `rg "__all__" src/kgfoundry` to document runtime exports in key modules.
- [ ] 1.2 Design typed namespace registry.
  - Create `NamespaceRegistry` dataclass with fields `name: str`, `loader: Callable[[], object]`, `module: str` (if needed).
  - Implement typed `register(name: str, loader: Callable[[], T]) -> None` and `resolve(name: str) -> T` functions with generics.
  - Replace `Any` sentinels with typed registries; ensure lazy loading semantics remain via cached loader results.
- [ ] 1.3 Update `_namespace_proxy.py` implementation.
  - Refactor `__getattr__` to consult the registry and raise `AttributeError` for missing symbols.
  - Ensure module docstring documents behavior and references contributing guide.
  - Add unit tests (`tests/kgfoundry/test_namespace_proxy.py`) verifying resolution, caching, and missing symbol behavior.
- [ ] 1.4 Align stubs with runtime exports.
  - For each affected stub (starting with `stubs/kgfoundry/agent_catalog/search.pyi`), list exports present in runtime module.
  - Replace `Any` types with precise `type[...]` aliases or Protocol definitions.
  - Remove redundant `# type: ignore` directives; ensure stub docstrings (if any) reference runtime modules.
- [ ] 1.5 Purge redundant casts/ignores in search modules.
  - Audit `src/kgfoundry/agent_catalog/search.py` for `cast(...)` or `# type: ignore`; adjust function signatures/return types to eliminate them.
  - Update dependent modules (`cli.py`, `client.py`, docs builders) if they relied on casted types.
  - Ensure tests cover the updated typing by running Pyrefly/Mypy.
- [ ] 1.6 Validate stub/runtime parity.
  - Write a small script (`tools/check_stub_parity.py`) or unit test verifying `dir(runtime_module)` matches stub exports (allowing controlled differences).
  - Integrate parity check into CI or developer checklist if possible.
- [ ] 1.7 Update documentation.
  - Amend `docs/contributing/typing.md` (or create it) describing namespace registry usage and stub alignment workflow.
  - Provide step-by-step instructions for adding new exports and updating stubs.
- [ ] 1.8 Execute targeted quality gates.
  - Run `uv run ruff check src/kgfoundry/_namespace_proxy.py` and `uv run ruff check stubs/kgfoundry/agent_catalog/search.pyi`.
  - Execute `uv run pyrefly check src/kgfoundry/_namespace_proxy.py stubs/kgfoundry/agent_catalog/search.pyi` and `uv run pyright --warnings --pythonversion=3.13 src/kgfoundry/agent_catalog` to ensure clean results.
  - After cleanup, run the full test suite (`uv run pytest -q`) focusing on namespace-dependent tests.

## 2. Verification & Artifact Updates
- [ ] 2.1 Regenerate docs if namespace changes affect examples (`make artifacts`).
- [ ] 2.2 Update release notes / changelog to mention namespace/stub alignment, including guidance for downstream tooling.
- [ ] 2.3 Final validation run: `uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run pyright --warnings --pythonversion=3.13`, `uv run pytest -q`, confirm no new suppressions.

