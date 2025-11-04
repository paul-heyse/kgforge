## Summary

Rebuild the documentation artifact pipeline on top of authoritative typed models
and shared loader facades. Landing this change reinstates strict type safety for
`build_symbol_index`, `symbol_delta`, and MkDocs generation, guarantees every
artifact is emitted through schema-backed helpers, and hardens `docs/conf.py`
against optional dependency drift. The effort replaces ad-hoc dictionary logic
and scattered casts with reusable modules under `docs/_types/`, restoring clean
Ruff/Pyrefly/pyright gates while preserving the observability and configuration
enhancements shipped previously.

This remediation emphasises three pillars:

1. **Authoritative data models** – typed structs and helper utilities own the JSON
   contract (index, delta, reverse lookups) and encapsulate transformations.
2. **Typed integration surfaces** – Griffe, MkDocs, and Sphinx integrations are
   mediated by runtime-checkable protocols that match only the attributes we use.
3. **Enforced validation loop** – schema validation, problem details, and tests are
   first-class, ensuring regressions are caught before artifacts hit disk.

The implementation strategy is incremental: introduce the new `_types` modules,
refactor writers to those primitives, then complete the downstream CLI and config
work so every touchpoint runs through typed facades.

## Public API Sketch

```python
"""docs/_types/artifacts.py"""
from __future__ import annotations

import msgspec
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonPayload: TypeAlias = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue

@dataclass(frozen=True, slots=True)
class LineSpan:
    start: int | None
    end: int | None

@msgspec.struct(omit_defaults=True)
class SymbolIndexRow:
    path: str
    canonical_path: str | None = None
    kind: str | None = None
    module: str | None = None
    package: str | None = None
    file: str | None = None
    span: LineSpan | None = None
    signature: str | None = None
    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    section: str | None = None
    tested_by: tuple[str, ...] = ()
    is_async: bool = False
    is_property: bool = False

@msgspec.struct
class SymbolIndexArtifacts:
    rows: tuple[SymbolIndexRow, ...]
    by_file: dict[str, tuple[str, ...]]
    by_module: dict[str, tuple[str, ...]]

@msgspec.struct
class SymbolDeltaChange:
    path: str
    before: dict[str, JsonValue]
    after: dict[str, JsonValue]
    reasons: tuple[str, ...]

@msgspec.struct
class SymbolDeltaPayload:
    base_sha: str | None
    head_sha: str | None
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[SymbolDeltaChange, ...]

class ArtifactCodec(msgspec.json.Codec):
    """Typed codec with repository-specific encoder/decoder defaults."""

def symbol_index_from_json(raw: JsonPayload) -> SymbolIndexArtifacts: ...
def symbol_index_to_payload(model: SymbolIndexArtifacts) -> JsonPayload: ...
def symbol_delta_from_json(raw: JsonPayload) -> SymbolDeltaPayload: ...
def symbol_delta_to_payload(model: SymbolDeltaPayload) -> JsonPayload: ...
def load_symbol_index(path: Path) -> SymbolIndexArtifacts: ...
def dump_symbol_index(path: Path, model: SymbolIndexArtifacts) -> None: ...
```

```python
"""docs/_types/griffe.py"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

@runtime_checkable
class GriffeNode(Protocol):
    path: str
    members: Mapping[str, "GriffeNode"]
    is_package: bool
    is_module: bool
    kind: str | None
    file: str | None
    lineno: int | None
    endlineno: int | None
    signature: object | None

class LoaderFacade(Protocol):
    def load(self, package: str) -> GriffeNode: ...

class MemberIterator(Protocol):
    def iter_members(self, node: GriffeNode) -> Iterator[GriffeNode]: ...

class GriffeFacade(Protocol):
    loader: LoaderFacade
    member_iterator: MemberIterator

def build_facade(env: BuildEnvironment) -> GriffeFacade: ...
```

```python
"""docs/_types/sphinx_optional.py"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

@runtime_checkable
class AutoapiParserFacade(Protocol):
    def parse(self) -> None: ...

@runtime_checkable
class AstroidManagerFacade(Protocol):
    def build_from_file(self, path: str) -> object: ...

class OptionalDependencies(Protocol):
    autoapi_parser: AutoapiParserFacade
    astroid_manager: AstroidManagerFacade

def load_optional_dependencies() -> OptionalDependencies: ...
```

```python
"""docs/_scripts/validate_artifacts.py"""
class ArtifactValidationError(RuntimeError):
    problem: ProblemDetailsDict

@dataclass(slots=True)
class ArtifactCheck:
    name: str
    path: Path
    schema: Path
    loader: Callable[[Path], JsonPayload]
    codec: Callable[[JsonPayload], object]

def validate_symbol_index(path: Path) -> SymbolIndexArtifacts: ...
def validate_symbol_delta(path: Path) -> SymbolDeltaPayload: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

## Component Breakdown

### Typed artifact module (`docs/_types/artifacts.py`)
- Owns `JsonValue` aliases, shared codec configuration, and `msgspec.Struct`
  definitions for every artifact (`SymbolIndexRow`, `SymbolIndexArtifacts`,
  `SymbolDeltaPayload`, future reverse lookup models).
- Provides helper functions to:
  - Construct models from raw JSON (`*_from_json`) with deterministic validation.
  - Emit JSON-compatible payloads (`*_to_payload`) ensuring schema alignment.
  - Load/dump files from disk with stable ordering and UTF-8 handling.
- Exposes utilities (`merge_reverse_lookup`, `sort_rows`) for writers, keeping
  mutation outside script code.

### Griffe facade (`docs/_types/griffe.py`)
- Wraps dynamic Griffe objects in runtime-checkable protocols and iterators.
- Provides a builder that configures the loader search path based on
  `BuildEnvironment`, returning a facade struct with a `loader` and
  `member_iterator` implementation.
- Includes helpers (`get_signature`, `iter_public_members`) producing typed
  outputs so downstream scripts never touch `Any`.

### Sphinx optional dependencies (`docs/_types/sphinx_optional.py`)
- Houses protocols for AutoAPI, Astroid, and docstring overrides with precise
  method signatures.
- Supplies a `load_optional_dependencies()` helper that performs guarded imports
  and raises `MissingDependencyError` with RFC 9457 Problem Details if modules
  are absent.
- Used by `docs/conf.py` to keep configuration logic declarative while being
  fully typed.

### Validation CLI enhancements
- `ArtifactValidationError` encapsulates typed Problem Details and the failing
  artifact name.
- `validate_artifacts.py` defines a table of `ArtifactCheck` instances so new
  artifacts can be registered declaratively.
- Schema validation uses the shared codec and typed models, removing parallel
  validation logic and ensuring consistent error messaging.

### docs/conf.py integration
- Consumes the typed optional dependency facades and the refined `WarningLogger`
  helper from `docs/_scripts/shared.py`.
- Maintains existing observability (structured logs, problem details) while
  eliminating `Any` by routing through the new helpers.

## Data / Schema Contracts

- JSON Schema 2020-12 documents remain the source of truth:
  - `schema/docs/symbol-index.schema.json`
  - `schema/docs/symbol-delta.schema.json`
  - (New) `schema/docs/symbol-reverse-lookup.schema.json` to formalise
    `by_file.json` / `by_module.json` (including required ordering and map
    semantics).
- Typed models in `docs/_types/artifacts.py` MUST serialize exactly to payloads
  that validate against these schemas. Every conversion helper enforces this via
  `jsonschema.Draft202012Validator` during tests and `validate_artifacts.py` at
  runtime.
- Problem Details emitted by the validation CLI follow the existing tools
  taxonomy (RFC 9457) with extensions describing:
  - `artifact`: logical artifact name (`symbols.json`, `symbols.delta.json`, etc.).
  - `schema_version`: semantic version extracted from the schema `$id`.
  - `json_pointer`: pointer to the failing location when available.
- Schema evolution will bump the `$id` and include backward/forward compatibility
  notes in `schema/docs/README.md`.

## Test Plan

- **Schema validation** – parametrised pytest suite covering
  `symbol_index_from_json`, `symbol_index_to_payload`, `symbol_delta_from_json`,
  `symbol_delta_to_payload`, and future reverse lookup helpers. Each test asserts
  round-trip fidelity and schema compliance.
- **Negative scenarios** – tests for malformed inputs (missing required fields,
  wrong types, unexpected properties) expect `ArtifactValidationError` with the
  correct Problem Details extensions.
- **Integration** – run `make artifacts` in CI (and locally) ensuring writers use
  the new helpers and the validation step reports `status=validated` logs.
- **Type gates** – enforce clean `uv run pyrefly check` and
  `uv run pyright --warnings --pythonversion=3.13` results for `docs/**`, `docs/_types/**`,
  and `tests/docs/**`.
- **Optional dependency coverage** – targeted tests (or doctest snippets) in
  `tests/docs/test_conf_optional.py` verifying `docs/conf.py` pathways when
  optional dependencies are present vs. missing, ensuring typed facades raise
  descriptive errors.
- **Documentation** – update `docs/contributing/quality.md` with copy-ready
  examples demonstrating invocation of the validation CLI and sample Problem
  Details payloads; include doctested snippets when feasible.

