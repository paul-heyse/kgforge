## Why
`tools/docstring_builder/plugins/__init__.py` currently instantiates Protocol classes
directly, causing Pyrefly “bad-instantiation” errors and MyPy unreachable code paths. The
registry relies on runtime duck typing and `Any`, undermining the strict typing posture
for docstring-builder pipelines. We need to refactor the registry to register concrete
factory callables, enforce stage Protocol compliance, and supply runnable examples so
plugin authors have a deterministic contract.

## What Changes
- [ ] **MODIFIED**: Refactor the plugin registry to store callables returning concrete
  plugin instances, avoiding direct Protocol instantiation and removing dead code.
- [ ] **ADDED**: Introduce `PluginFactory[T_Plugin]` Protocols and typed helpers that
  validate factories at registration time, raising structured errors for abstract or
  mis-typed plugins.
- [ ] **ADDED**: Expand tests covering registry registration, duplicate detection,
  rejection of Protocol types, and execution of built-in plugin factories.
- [ ] **ADDED**: Documentation with runnable examples illustrating how to register and
  invoke plugins via factories and how legacy adapters behave.
- [ ] **MODIFIED**: Update docstring-builder CLI/tests to consume the new registry API
  while keeping backwards compatibility for legacy adapters.

## Impact
- **Affected specs:** `tools-suite`
- **Affected code:** `tools/docstring_builder/plugins/__init__.py`,
  `tools/docstring_builder/plugins/base.py`, associated tests under
  `tests/tools/docstring_builder`, documentation under `docs/contributing/`.
- **Data contracts:** None (internal tooling change), but CLI docs must reflect updated
  plugin registration API.
- **Rollout:** Implement on branch `openspec/docbuilder-plugin-registry-hardening-phase1`
  with full lint/type/test gates; coordinate with tooling owners before merging.

## Acceptance
- [ ] Ruff, Pyright, Pyrefly, and MyPy pass without suppressions for the plugin registry
  modules and tests.
- [ ] Registry refuses to register Protocol classes or abstract factories, raising
  `PluginRegistryError` (or equivalent) with RFC 9457 Problem Details.
- [ ] Built-in plugins register via factories and execute through the registry in tests.
- [ ] Documentation provides runnable example code showing factory registration and
  invocation, validated via doctest.

## Out of Scope
- Revisiting the broader docstring-builder pipeline orchestration or CLI UX.
- Changing plugin payload schemas (covered by “tools-plugin-protocol-generics”).
- Introducing new plugin stages beyond formatter/harvester/transformer.

## Risks / Mitigations
- **Risk:** Factory-based API could break existing custom plugins.
  - **Mitigation:** Provide compatibility adapter accepting legacy plugin classes and
    document migration steps with tests.
- **Risk:** Additional validation might surface latent bugs in built-in plugins.
  - **Mitigation:** Expand tests to cover built-in plugin factories and ensure CLI smoke
    tests run through the registry.
- **Risk:** Documentation examples may drift.
  - **Mitigation:** Use doctest/xdoctest for the new examples and include them in CI.

