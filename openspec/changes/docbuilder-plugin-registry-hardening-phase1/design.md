## Context
- The plugin registry currently instantiates Protocols (`FormatterPlugin`, `HarvesterPlugin`,
  etc.), violating typing rules and causing Pyrefly “bad-instantiation” errors and MyPy
  unreachable code. This architecture also allows abstract plugin classes to slip through
  without validation.
- Plugins are key to the docstring-builder pipeline; they must remain discoverable and
  configurable without sacrificing strict typing or observability.
- Legacy plugins may still rely on direct class registration, so the refactor must provide
  adapters or migration tooling to keep existing behavior operational while nudging
  adopters toward factory-based registration.

## Goals / Non-Goals
- **Goals**
  - Store plugin factories (callables) in the registry, ensuring registered objects are
    fully constructed instances.
  - Introduce `PluginFactory[T_Plugin]` Protocol defining a `__call__` contract for
    factory functions, enabling static validation.
  - Validate factories at registration time, rejecting Protocol classes, abstract base
    classes, or callables with incompatible signatures.
  - Provide migration helpers for legacy plugins and document the new workflow via
    runnable examples.
  - Strengthen regression coverage (unit tests + doctests) to assert registry invariants.
- **Non-Goals**
  - Redesign plugin discovery (entry points, config files) beyond the factory interface.
  - Alter plugin payload schemas or stage generics (already covered by
    `tools-plugin-protocol-generics`).
  - Introduce asynchronous plugin stages (tracked separately if needed).

## Decisions
- Define `PluginFactory[T_Plugin]` as a Protocol with `__call__(**kwargs) -> T_Plugin`,
  parameterizing over plugin stage types.
- Maintain separate registries per stage (`FormatterPlugin`, `HarvesterPlugin`,
  `TransformerPlugin`) storing `PluginFactory` callables and metadata (name, legacy flag).
- Provide helper `register_plugin(name, factory, stage)` that validates the factory and
  ensures uniqueness, raising `PluginRegistryError` with Problem Details payloads on
  failure.
- Implement `LegacyPluginFactory` adapter wrapping legacy plugin classes or instances,
  emitting deprecation warnings while satisfying the factory Protocol.
- Update CLI/tests to obtain plugin instances via the registry’s factory invocation instead
  of direct class instantiation.
- Add doctest-backed documentation demonstrating registration, validation failure, and
  execution of a formatter plugin.

## Alternatives
- **Retain class registration with runtime checks** – Rejected due to persistent typing
  errors and less explicit control over instantiation side effects.
- **Use dataclasses to describe plugins** – Deferred; factories provide adequate flexibility
  without heavier abstractions.
- **Adopt dependency injection container** – Overkill for current scope; factories offer a
  simpler path aligned with existing architecture.

## Risks / Trade-offs
- Factory enforcement may require plugin authors to refactor code. Mitigated by providing a
  legacy adapter and clear migration guide.
- Additional validation could introduce runtime overhead. Mitigated by caching validation
  results and keeping factory invocation lightweight.
- More complex registry state increases maintenance. Mitigated by comprehensive tests and
  documentation of invariants.

## Migration
1. Introduce `PluginFactory` Protocols and registry data structures, alongside a new
   `PluginRegistryError` (or extend existing error taxonomy).
2. Refactor registration functions to accept factories, validating uniqueness and
   instantiation results.
3. Update built-in plugin registrations to use factories; add `LegacyPluginFactory` for
   older patterns.
4. Adjust CLI/tests to obtain plugin instances via factory invocation; update fixtures as
   needed.
5. Add regression tests verifying valid registration, rejection of Protocol classes, and
   legacy compatibility; integrate doctest examples.
6. Document new workflow and run full lint/type/test gates before merging.

