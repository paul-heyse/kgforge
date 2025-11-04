## ADDED Requirements
### Requirement: Typed Docstring Builder Plugin Interfaces
Docstring-builder plugins SHALL declare generic input/output payload types so static checkers validate stage boundaries without suppressions.

#### Scenario: Stage protocols advertise payload generics
- **GIVEN** `tools/docstring_builder/plugins/base.py`
- **WHEN** `uv run pyrefly check` runs over `tools/docstring_builder`
- **THEN** harvester, transformer, and formatter Protocols each expose typed `apply` signatures (`HarvestResult`, `SemanticResult`, `DocstringEdit` respectively) and no `pyrefly: ignore` suppressions remain for those modules

#### Scenario: Plugin manager preserves typing and observability
- **GIVEN** the plugin manager helpers in `tools/docstring_builder/plugins/__init__.py`
- **WHEN** built-in plugins execute via the manager during `uv run pytest -q tools/docstring_builder`
- **THEN** payloads remain strongly typed throughout invocation, structured error logs capture `plugin`, `stage`, and `file_path`, and no `cast` operations are required to satisfy pyrefly or pyright

#### Scenario: Legacy plugins remain compatible
- **GIVEN** a legacy plugin using the pre-generic `run` API wrapped by `LegacyPluginAdapter`
- **WHEN** the adapter executes that plugin during docstring generation
- **THEN** the payload types are bridged without `Any` escapes, a deprecation warning is emitted once, and the run completes without type-checker suppressions


