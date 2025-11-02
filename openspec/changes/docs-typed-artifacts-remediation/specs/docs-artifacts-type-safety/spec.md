# Capability: Typed Docs Artifact Pipeline

## Purpose

Ensure every documentation artifact (symbol index, delta, reverse lookups) is
produced and validated through typed models that align with canonical JSON
Schemas. The capability formalises the guarantees needed for observability,
schema drift detection, and strict type gates across the docs toolchain.

## Requirements

1. **Authoritative models** – All docs artifacts MUST be represented by
   authoritative Python models (`msgspec.Struct` or equivalent) residing under
   `docs/_types/`. Writers may not emit ad-hoc dictionaries; they must call the
   provided conversion helpers.
2. **Schema conformity** – Models MUST serialize to payloads that validate against
   the published JSON Schema 2020-12 documents housed in `schema/docs/`. Any
   schema violation SHALL raise a structured Problem Details error before files
   are written.
3. **Typed loader facade** – Interaction with Griffe and optional Sphinx
   dependencies MUST occur through typed facades that expose only the attributes
   consumed by the docs pipeline. No `Any`-typed access is permitted in public
   modules.
4. **Problem Details** – Validation failures MUST surface RFC 9457 Problem
   Details with extensions describing the artifact, schema version, and JSON
   pointer for the failing field.
5. **Quality gates** – Ruff, Pyrefly, and mypy MUST run cleanly across the docs
   scripts and supporting types. Missing type coverage fails the capability.
6. **Execution tracking** – The associated change record MUST maintain a
   checkbox-tracked task list covering model scaffolding, facade wiring,
   validation CLI updates, tests, and documentation so reviewers can verify every
   requirement landed prior to archive.

## Acceptance Scenarios

### Scenario 1 – Symbol index round-trip

*Given* a valid `symbols.json` payload

*When* it is loaded via `symbol_index_from_json` and re-serialized via
`symbol_index_to_payload`

*Then* the resulting payload MUST byte-for-byte match the input and MUST pass the
`symbol-index.schema.json` validator.

### Scenario 2 – Schema violation is blocked

*Given* a payload missing the required `path` field

*When* `validate_symbol_index` processes the payload

*Then* it MUST raise `ArtifactValidationError` containing a Problem Details body
with `status = 422`, `type` referencing the docs artifact validation namespace,
and an extension indicating the missing JSON pointer.

### Scenario 3 – Docs build integration

*Given* the docs build pipeline runs `make artifacts`

*When* `build_symbol_index.py` and `symbol_delta.py` complete

*Then* each artifact writer MUST log `status=validated`, emit payloads generated
via the typed models, and the subsequent validation step MUST succeed without
raising exceptions.

### Scenario 4 – Optional dependency fallback

*Given* `docs/conf.py` runs in an environment without AutoAPI

*When* the configuration attempts to load the optional dependency through the
typed facade

*Then* it MUST raise a descriptive `ImportError` (or logged warning) explaining
the missing dependency while keeping mypy type coverage intact.

