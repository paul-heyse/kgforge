## Why
The documentation toolchain still implements symbol index builds, symbol delta emission, and artifact validation as loosely structured scripts. Each module (`docs/toolchain/build_symbol_index.py`, `symbol_delta.py`, `validate_artifacts.py`) manages logging, error handling, and metrics differently, often relying on ad-hoc dictionaries and `print` statements. This divergence complicates maintenance, reduces observability consistency, and prevents reuse of lifecycle helpers similar to our CLI tooling abstractions.

## What Changes
- [ ] **ADDED**: capability spec defining context objects and lifecycle helpers for doc generation tasks, including standardized logging and Problem Details emission.
- [ ] **MODIFIED**: `docs/toolchain/build_symbol_index.py`, `docs/toolchain/symbol_delta.py`, and `docs/toolchain/validate_artifacts.py` to consume the shared lifecycle helpers and context objects.
- [ ] **MODIFIED**: shared tooling module (e.g., `docs/_scripts/shared` or new `docs/toolchain/_lifecycle.py`) to expose typed contexts, Problem Details builders, and metrics hooks.

## Impact
- **Capability surface**: new spec for doc toolchain lifecycle context & observability.
- **Code scope**: refactor three doc toolchain entrypoints plus add shared lifecycle helpers; ensure structured logging, metrics, and Problem Details align across tasks.
- **Out of scope**: unit tests and documentation updates (handled later).

- [ ] Ruff / Pyright / Pyrefly clean across refactored modules.
- [ ] Doc toolchain commands emit structured logs, metrics, and Problem Details using shared helpers.
- [ ] Lifecycle context objects allow consistent configuration and dependency injection.

## Out of Scope
- Extending test coverage or regenerating documentation artifacts (planned separately).
- Refactoring unrelated tooling beyond the three doc toolchain scripts.

## Risks / Mitigations
- **Risk:** Refactor might disrupt existing doc build workflows.  
  **Mitigation:** maintain command-line parity, ensure lifecycle helpers preserve behaviour, and stage changes carefully.
- **Risk:** Introducing shared helpers could create circular dependencies.  
  **Mitigation:** place lifecycle module in a neutral location (`docs/toolchain/_shared` or similar) with no reverse imports.

## Alternatives Considered
- Keep scripts independent and patch inconsistencies locally — rejected to avoid ongoing divergence and maintenance cost.
- Rewrite tooling entirely around a new framework — out of scope for this incremental alignment.
