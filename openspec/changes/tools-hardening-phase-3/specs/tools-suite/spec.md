## MODIFIED Requirements
### Requirement: Typed Tooling Payload Models
Documentation and navmap tooling SHALL operate on explicit dataclasses/TypedDicts validated against JSON Schemas, eliminating `Any` leakage.

#### Scenario: Docstring cache round-trips through schema
- **GIVEN** docstring builder cache persistence
- **WHEN** cached payloads are serialized
- **THEN** they instantiate typed models, validate against `schema/tools/docstring_cache.json`, and tests assert parity for happy/edge/error cases

#### Scenario: Docs analytics emit typed envelopes
- **GIVEN** `tools/docs/build_agent_analytics.py`
- **WHEN** it produces analytics JSON
- **THEN** the result matches a typed model, validates against `schema/tools/doc_analytics.json`, and returns Problem Details on validation failure

#### Scenario: Msgspec structs type-check without Any
- **GIVEN** msgspec-backed helpers for CLI envelopes, navmap documents, docstring caches, analytics, and `sitecustomize`
- **WHEN** `mypy --config-file mypy.ini` runs against `tools/_shared`, `tools/docstring_builder`, `tools/docs`, `tools/navmap`, and `sitecustomize.py`
- **THEN** no `Any`-based diagnostics remain because helpers expose typed constructors, converters, and schema validators

#### Scenario: Legacy payloads migrate safely
- **GIVEN** previously stored cache or navmap payloads that follow the legacy structure
- **WHEN** the updated tooling deserializes them
- **THEN** migration helpers accept the old version, upgrade it to the new struct, and record a regression test demonstrating the behaviour

### Requirement: LibCST Codemod Typing Guarantees
Codemod utilities SHALL rely on typed LibCST interfaces backed by local stubs so mypy/pyrefly report no missing attributes or `Any` leakage.

#### Scenario: LibCST stubs enable typing
- **GIVEN** codemod modules under `tools/codemods/`
- **WHEN** mypy and pyrefly run
- **THEN** LibCST node references resolve via local stubs (or bundled `py.typed`), yielding zero `attr-defined`/`name-defined` violations

#### Scenario: Codemod tests exercise typed transformations
- **GIVEN** pytest codemod suites
- **WHEN** they execute
- **THEN** they construct typed LibCST trees, apply transformers, and assert output without disabling type checking via `Any`

#### Scenario: Codemod package ships typed stubs
- **GIVEN** the tooling distribution
- **WHEN** `pip install .[tools]` is performed in a clean environment
- **THEN** the installed package exposes `py.typed` for `libcst` shims and codemod helpers import without mypy or runtime stub errors

#### Scenario: Codemod contributor guide exists
- **GIVEN** the repository documentation
- **WHEN** a developer reads the codemod section
- **THEN** they find a short guide describing how to add typed transformers, run `mypy`/`pyrefly`, and extend the `stubs/libcst` package

### Requirement: Packaged Tools
The tools suite SHALL build distributable artifacts with optional extras and run cleanly after installation.

#### Scenario: Clean install
- **WHEN** `pip install .[tools]` executes inside a fresh virtual environment
- **THEN** the installation succeeds, entry points resolve, and CLIs run without import errors or missing metadata

#### Scenario: Distribution exposes typed exports
- **GIVEN** the built wheel or sdist
- **WHEN** introspected
- **THEN** it contains `tools/py.typed`, accurate stub packages for shared helpers, and the `tools` package re-exports CLI/Problem Details utilities consistent with runtime behavior

#### Scenario: Smoke test verifies installation
- **GIVEN** the packaging smoke test script
- **WHEN** it provisions a fresh virtual environment, installs `. [tools]`, imports the canonical helpers, and runs a no-op CLI
- **THEN** the script exits with status 0 and logs successful imports and CLI execution

## ADDED Requirements
### Requirement: Typed Tooling Test Suite
Tooling test modules SHALL declare precise types and execute under the strict mypy configuration without lossy `Any` casts or untyped decorators.

#### Scenario: Mypy validates tests/tools
- **GIVEN** the strict mypy configuration
- **WHEN** `mypy --config-file mypy.ini tests/tools` runs
- **THEN** it succeeds with zero errors because parametrized fixtures, decorators, and helper functions expose typed signatures and avoid broad `Any`

#### Scenario: Fixtures enforce typed payloads
- **GIVEN** shared fixtures under `tests/tools/**`
- **WHEN** they build docstring builder, docs analytics, navmap, or sitecustomize payloads
- **THEN** the payloads are instantiated via typed models or helpers and validated against their schemas within the tests

#### Scenario: Contributor guide documents typing rules
- **GIVEN** `tests/tools/README.md`
- **WHEN** a new engineer opens the file
- **THEN** it explains how to type-annotate fixtures, parametrized tests, and helper factories, and references the commands required to verify `mypy`

