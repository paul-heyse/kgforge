## Context
- Logging contexts (`LogContextExtra`), docstring-builder error payloads (`ErrorReport`),
  and CLI results (`CliResult`) rely on loose TypedDict definitions that mark many fields
  optional but are accessed as required. Static analyzers flag these access patterns, and
  runtime code risks `KeyError` if fields are missing.
- Observability guidelines in `AGENTS.md` require structured logs with correlation IDs and
  Problem Details payloads for failure paths. Current implementations inconsistently
  capture these fields.
- Mutation of logging contexts is common (e.g., adding `operation`, `status`) but lacks a
  frozen data model, leading to shared-state bugs and unclear ownership.

## Goals / Non-Goals
- **Goals**
  1. Replace fragile TypedDicts with frozen dataclasses or TypedDicts using PEP 655
     `Required`/`NotRequired` fields to formalize structure.
  2. Provide safe accessor/update helpers that return new instances while preserving
     immutability expectations.
  3. Ensure logging APIs emit RFC 9457 Problem Details with correlation IDs, operation
     names, and status fields, backed by example JSON.
  4. Expand test coverage (pytest + doctest) to verify required/optional partitions,
     helper behavior, and Problem Details emission.
- **Non-Goals**
  - Replacing the logging framework or metrics exporters.
  - Introducing new plugin stages or CLI commands unrelated to context hygiene.
  - Rewriting schema alignment or artifact model code.

## Decisions
- Use frozen dataclasses (`@dataclass(frozen=True)`) for contexts that benefit from rich
  behavior (e.g., `LogContextExtra`) and PEP 655 TypedDicts for purely structural payloads
  (`ErrorReport`, `CliResult`), keeping immutability guarantees.
- Define helper methods (e.g., `with_status`, `with_correlation_id`) that return new
  dataclass instances; for TypedDicts, provide wrapper functions returning new dicts.
- Centralize Problem Details emission in logging helpers, referencing the canonical JSON
  example and ensuring correlation IDs appear in both logs and Problem Details payloads.
- Update tests to use `.get` or helper accessors, preventing direct indexing of optional
  keys; add table-driven tests covering presence/absence scenarios.

## Alternatives Considered
- **Leave TypedDicts as-is with suppressions** — rejected; conflicts with zero-suppression
  mandate and leaves runtime risk.
- **Convert everything to full dataclasses** — considered but TypedDicts remain useful for
  interop; hybrid approach chosen.
- **Introduce runtime schema validation** — out of scope; focus is on typing hygiene and
  observability alignment.

## Risks & Mitigations
- **Mutation expectations** — some callers may rely on mutating contexts. Mitigated by
  providing helper methods that return updated copies and documenting the change.
- **Problem Details drift** — new examples might diverge from actual output. Mitigated by
  writing tests comparing emitted payloads to the stored JSON example and validating via
  `make artifacts`.
- **Test churn** — numerous tests access optional fields directly. Mitigated by creating
  helper functions (e.g., `assert_log_context`) to reduce boilerplate.

## Migration Plan
1. Inventory definitions of `LogContextExtra`, `ErrorReport`, and `CliResult` plus the
   modules/tests that touch them.
2. Introduce frozen dataclasses / TypedDict partitions with explicit required and optional
   fields and migration helpers.
3. Update logging helpers and CLI/docstring-builder code to instantiate and manipulate
   these new structures via safe accessors.
4. Adjust tests to use `.get` or helper methods; add regressions ensuring Problem Details
   with correlation IDs are emitted.
5. Add/refresh Problem Details JSON examples and documentation.
6. Run full quality gates and regenerate docs/artifacts.

