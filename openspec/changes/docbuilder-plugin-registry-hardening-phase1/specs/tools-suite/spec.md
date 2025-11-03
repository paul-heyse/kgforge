## ADDED Requirements
### Requirement: Plugin registry factory contracts
The docstring-builder plugin registry SHALL manage stage-specific registries of
factory callables rather than Protocol classes, validating factories at registration
time and ensuring only concrete plugin instances execute during pipeline runs.

#### Scenario: Factory registration succeeds
- **GIVEN** a formatter plugin factory conforming to `PluginFactory[FormatterPlugin]`
- **WHEN** it registers via `register_plugin("markdown-formatter", factory, stage="formatter")`
- **THEN** the registry stores the callable, later invocation returns a concrete
  formatter instance, and Pyrefly/Mypy report no protocol-instantiation errors

#### Scenario: Abstract plugin is rejected
- **GIVEN** a plugin registry registration attempt passing a Protocol class or abstract
  base class
- **WHEN** `register_plugin` validates the input
- **THEN** it raises `PluginRegistryError` with an RFC 9457 Problem Details payload
  documenting the stage, plugin name, and reason (abstract/protocol), and logs the
  failure with `status="error"`

#### Scenario: Legacy adapter preserves compatibility
- **GIVEN** a legacy plugin class using the pre-factory API
- **WHEN** the migration adapter wraps it and registers via the new factory interface
- **THEN** the plugin executes successfully, emits a one-time deprecation warning, and
  tests confirm payload types remain strict throughout the pipeline

#### Scenario: Documentation example remains runnable
- **GIVEN** doctest execution over the plugin registry documentation or helper module
- **WHEN** the example registers a toy plugin factory and runs it through the pipeline
- **THEN** the doctest passes without additional setup, demonstrating the new factory
  contract and error handling semantics

