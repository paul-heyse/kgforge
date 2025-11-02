# Task 1 – Typed artifact core

- [ ] **1.1** Add `docs/_types/artifacts.py` with authoritative `msgspec.Struct` models for the
      symbol index rows, deltas, and reverse-lookups. Include helper functions for
      JSON conversion (`from_json`, `to_payload`).
  - [ ] **1.1.a** Scaffold the module with a top-level NumPy-style docstring describing purpose and
        relationship to schemas; ensure `__all__` enumerates exported symbols.
  - [ ] **1.1.b** Declare `JsonPrimitive`, `JsonValue`, and `JsonPayload` type aliases; document
        invariants (e.g., tuples for deterministic ordering, UTF-8 serialization rules).
  - [ ] **1.1.c** Implement `LineSpan`, `SymbolIndexRow`, `SymbolIndexArtifacts`, `SymbolDeltaChange`,
        `SymbolDeltaPayload`, and reverse-lookup structs using `msgspec.struct` with
        `omit_defaults=True` where applicable; ensure each field is annotated and has a default.
  - [ ] **1.1.d** Implement conversion helpers (`*_from_json`, `*_to_payload`, `load_*`, `dump_*`) that
        call a shared `ArtifactCodec`; include defensive validation (type checks, tuple conversion)
        and raise `ArtifactValidationError` on invalid data.
  - [ ] **1.1.e** Add unit tests under `tests/docs/test_artifact_models.py` covering round-trip and
        field coercion behaviours, mapping to schema examples.

- [ ] **1.2** Refactor `docs/_scripts/build_symbol_index.py` to construct
      `SymbolIndexArtifacts` instances and emit payloads exclusively through the
      helper functions.
  - [ ] **1.2.a** Replace local dataclasses with imports from `docs/_types/artifacts.py`; remove redundant
        JSON assembly logic.
  - [ ] **1.2.b** Adapt row collection functions to build `SymbolIndexRow` instances directly, leveraging
        helper utilities for span, metadata, and reverse lookups.
  - [ ] **1.2.c** Update writing logic to call `symbol_index_to_payload` before persisting; ensure schema
        validation hooks receive typed payloads.
  - [ ] **1.2.d** Adjust structured logging metadata (`status`, `artifact`, counts) to reflect new flow;
        confirm `observe_tool_run` context remains intact.
  - [ ] **1.2.e** Extend existing tests (or add new ones) to verify deterministic ordering and payload
        equality against schema examples.

- [ ] **1.3** Refactor `docs/_scripts/symbol_delta.py` to use the new models and remove all
      residual `Any` flows / hand-built dictionaries.
  - [ ] **1.3.a** Replace manual dict assembly with `SymbolDeltaChange` and `SymbolDeltaPayload` instances;
        drop intermediate `Mapping[str, object]` constructs.
  - [ ] **1.3.b** Ensure row coercion functions return typed `SymbolIndexRow` instances via shared helpers,
        handling legacy compatibility quirks (e.g., floats vs. ints) within the codecs.
  - [ ] **1.3.c** Update diffing logic to produce sorted tuples and reason strings; guarantee all
        operations return typed collections (`tuple[str, ...]`, etc.).
  - [ ] **1.3.d** Route writing/validation through `symbol_delta_to_payload` with schema validation prior to
        disk writes.
  - [ ] **1.3.e** Add regression tests covering added/removed/changed scenarios, ensuring payloads match
        schema expectations and Problem Details surface on errors.

# Task 2 – Loader & configuration facades

- [ ] **2.1** Create `docs/_types/griffe.py` exposing runtime-checkable protocols and a
      typed loader facade; update `docs/_scripts/shared.py` and
      `docs/_scripts/mkdocs_gen_api.py` to consume it.
  - [ ] **2.1.a** Define `GriffeNode`, `LoaderFacade`, `MemberIterator`, and `GriffeFacade` protocols with
        the exact attribute/method subset used downstream.
  - [ ] **2.1.b** Implement `build_facade(env)` that configures the Griffe search path, returns a facade
        containing the loader and a typed member iterator helper.
  - [ ] **2.1.c** Replace direct Griffe usage in `shared.py` and `mkdocs_gen_api.py` with imports from the
        facade; adjust `_documentable_members` and traversal routines to accept typed nodes only.
  - [ ] **2.1.d** Add unit tests (or property-based tests) to ensure the facade rejects objects missing
        required attributes and correctly iterates over packages/modules.

- [ ] **2.2** Replace ad-hoc logger adapters with a helper that satisfies the
      `WarningLogger` protocol so `resolve_git_sha` and related helpers remain
      type-safe.
  - [ ] **2.2.a** Introduce `docs/_scripts/logging.py` (or extend `shared.py`) with a `build_warning_logger`
        helper returning an adapter whose `warning` signature matches the protocol.
  - [ ] **2.2.b** Update all call sites (`resolve_git_sha`, artifact writers, validation CLI) to use the
        helper; ensure structured context (`operation`, `artifact`, `status`) is preserved.
  - [ ] **2.2.c** Verify via mypy that the helper satisfies the protocol and remove redundant casts.

- [ ] **2.3** Extract optional dependency shims for Sphinx (Astroid, AutoAPI, docstring
      overrides) into `docs/_types/sphinx_optional.py` and update `docs/conf.py` to
      rely on the typed interfaces.
  - [ ] **2.3.a** Define protocols for each optional component (`AutoapiParserFacade`, `AstroidManagerFacade`,
        docstring override hooks) with precise method signatures.
  - [ ] **2.3.b** Implement a loader function that performs guarded imports, surfaces
        `ProblemDetailsDict` on failure, and documents configuration fallbacks.
  - [ ] **2.3.c** Refactor `docs/conf.py` to consume the typed shims, removing residual `Any` and redundant
        casts; ensure docstrings and logging remain intact.
  - [ ] **2.3.d** Add tests covering both “dependency present” and “dependency missing” pathways.

# Task 3 – Validation, CLI, and tests

- [ ] **3.1** Introduce `ArtifactValidationError`, update `docs/_scripts/validation.py` and
      `docs/_scripts/validate_artifacts.py` to use the typed models, and ensure
      Problem Details envelopes include schema metadata.
  - [ ] **3.1.a** Implement `ArtifactValidationError` with fields for `artifact`, `schema`, and
        `ProblemDetailsDict`; include a helper to render CLI-safe output.
  - [ ] **3.1.b** Update validation helpers to accept typed payloads, invoke the shared codec, and raise
        the new exception with RFC 9457-compliant details.
  - [ ] **3.1.c** Refactor the CLI to iterate over declarative `ArtifactCheck` entries, collect results,
        and emit structured logs/metrics for successes and failures.
  - [ ] **3.1.d** Add targeted tests for success, missing file, schema violation, and invalid JSON cases.

- [ ] **3.2** Expand `tests/docs/test_doc_artifacts.py` (and new tests as needed) with
      parametrised round-trip, schema failure, and missing-artifact cases.
  - [ ] **3.2.a** Use pytest parametrisation to cover each artifact type with valid payloads loaded via the
        new codec helpers.
  - [ ] **3.2.b** Add fixtures or factories for constructing malformed payloads (missing fields, wrong
        types) and assert that `ArtifactValidationError` or `ToolExecutionError` surfaces with the
        expected Problem Details.
  - [ ] **3.2.c** Validate that round-tripped payloads are byte-identical to the original JSON examples
        stored under `schema/examples/docs/`.
  - [ ] **3.2.d** Integrate doctest/xdoctest snippets demonstrating CLI usage in contributor docs.

- [ ] **3.3** Update contributor documentation to describe the typed artifact workflow and
      confirm `make artifacts` validates all payloads. Verify Ruff, Pyrefly, mypy,
      pytest, and schema validations are clean before marking the change complete.
  - [ ] **3.3.a** Amend `docs/contributing/quality.md` (and related pages) with sections covering the new
        `_types` modules, validation CLI, and sample Problem Details output; ensure examples are
        copy-ready.
  - [ ] **3.3.b** Run the full quality gate sequence (`ruff check --fix`, `pyrefly`, `mypy`, `pytest`,
        `make artifacts`) and capture command outputs for the PR template.
  - [ ] **3.3.c** Execute `openspec validate docs-typed-artifacts-remediation --strict` prior to submission
        and archive outputs for reviewers.

