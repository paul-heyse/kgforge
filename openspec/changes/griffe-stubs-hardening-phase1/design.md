## Context
- Vendored Griffe stubs were originally introduced as thin placeholders and still rely
  on `*args: Any`, `**kwargs: Any`, and loosely typed return values. These leaks trigger
  Ruff (`ANN401`), Pyright, and Pyrefly issues and undermine the strict typing posture
  adopted in `AGENTS.md`.
- Docs scripts depend on Griffe’s loader APIs and plugin hooks, but optional dependencies
  (AutoAPI, Sphinx) can be missing at runtime. Without typed facades, we cannot express
  the conditional imports cleanly or provide fallback error paths.
- We intend to contribute improved stubs upstream once they stabilize, so our design must
  balance strict typing with maintainability across Griffe releases.

## Goals / Non-Goals
- **Goals**
  - Eliminate `Any` from the Griffe stubs by providing precise overloads and Protocols
    that mirror runtime behavior.
  - Expose typed facades in `docs/_types/griffe.py` that centralize optional dependency
    handling and keep docs scripts type-clean.
  - Add regression tests verifying stub/runtime parity and type checker compliance.
  - Document the maintenance and upstream contribution process for the stubs.
- **Non-Goals**
  - Replacing Griffe or redesigning the docs build pipeline.
  - Introducing new plugin systems beyond typing the existing hooks.
  - Expanding schema validation (covered by artifact-models hardening).

## Decisions
- Define `Protocol` interfaces for loader callbacks and plugin hooks so overloads can
  reference extensible call signatures without resorting to `Any`.
- Use PEP 695 type parameters (where practical) to express generic loader results,
  ensuring stubs align with the runtime use of `Module`/`Object` types.
- Centralize optional imports behind `typing.TYPE_CHECKING` guards; at runtime, lazily
  import modules and raise descriptive `ImportError` if dependencies are missing.
- Implement parity tests that introspect `griffe` at runtime (when available) and compare
  exported symbol names and overload arity against the stubs. Provide skip markers when
  Griffe is absent.
- Record the compatible Griffe version in the design notes and tasks, and plan to submit
  the typings upstream after verifying stability.

## Alternatives
- **Maintain broad `Any` annotations** – Rejected because it violates Ruff and type
  checker gates, masking real defects.
- **Generate stubs dynamically** – Rejected for now; deterministic, reviewed stub files
  are easier to diff and align with upstream contributions.
- **Wrap Griffe with our own API** – Deferred; typed facades plus precise stubs provide
  sufficient hygiene without creating a large wrapper layer.

## Risks / Trade-offs
- Precise overloads may lag behind future Griffe releases. Mitigated by parity tests and
  upstream collaboration.
- Optional dependency guards could complicate runtime error handling. Mitigated by
  providing dedicated exceptions (e.g., `ArtifactDependencyError`) and thorough tests.
- Additional Protocol definitions may seem verbose. Mitigated by documenting their usage
  and keeping APIs aligned with runtime semantics.

## Migration
1. Audit current Griffe runtime (version recorded in `uv.lock`) and inventory loader
   signatures, plugin APIs, and exceptions used by the docs pipeline.
2. Design Protocols and overloads that cover these APIs, drafting updated stubs under
   `stubs/griffe/`.
3. Refactor `docs/_types/griffe.py` to import the typed APIs, provide fallbacks, and
   expose helper functions consumed by docs scripts.
4. Add regression tests comparing stub signatures against runtime implementations and
   verifying clean Pyright/Pyrefly runs with the new facades.
5. Update documentation with maintenance workflow and upstream contribution guidance;
   prepare a patch for the Griffe project after local validation.

