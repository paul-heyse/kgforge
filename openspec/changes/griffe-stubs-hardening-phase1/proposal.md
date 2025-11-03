## Why
Our locally vendored Griffe stubs still expose `*args: Any`, `**kwargs: Any`, and
loosely typed return values. These leaks undermine strict static analysis, force ad-hoc
casts, and prevent typed wrappers from accurately reflecting Griffe’s plugin APIs.
Ruff (`ANN401`), Pyright, and Pyrefly continue to flag the stubs, blocking the quality
gates mandated in `AGENTS.md`. We need a focused hardening effort that aligns the stubs
with current Griffe releases, documents optional plugin dependencies, and guarantees
type-checker parity across our docs toolchain.

## What Changes
- [ ] **MODIFIED**: Rewrite `stubs/griffe/__init__.pyi` with concrete overloads for
  `load`, `load_module`, and plugin registration helpers, removing blanket `Any` usage
  while mirroring Griffe’s runtime signatures.
- [ ] **MODIFIED**: Update `stubs/griffe/loader/__init__.pyi` to annotate loader
  factories, plugin hooks, and optional parameters explicitly, including module-level
  `Protocol` definitions where runtime functions accept callables.
- [ ] **ADDED**: Introduce TYPE_CHECKING-friendly facades in `docs/_types/griffe.py`
  that import heavy dependencies lazily and surface typed entry points consumed by docs
  scripts.
- [ ] **ADDED**: Regression tests validating stub/runtime parity (symbol export list,
  signature comparison) and smoke tests exercising the typed facades under Pyright and
  Pyrefly.
- [ ] **ADDED**: Documentation outlining stub maintenance workflow, upstream
  contribution plan, and dependency markers for optional plugins.

## Impact
- **Affected specs:** `docs-toolchain`
- **Affected code:** `stubs/griffe/**/*.pyi`, `docs/_types/griffe.py`,
  `docs/_scripts/shared.py`, `tests/docs/test_griffe_facade.py`, tooling docs.
- **Data contracts:** None (this change targets typing contracts only), but typed
  facades must remain compatible with existing schema workflows.
- **Rollout:** Develop on branch `openspec/griffe-stubs-hardening-phase1`, run full
  lint/type/test gates, and coordinate with docs maintainers before merging.

## Acceptance
- [ ] Ruff, Pyright, Pyrefly, and MyPy report zero errors related to Griffe stubs or
  the typed facades.
- [ ] Stub overloads match the runtime signatures for `load`, `load_module`, plugin
  registries, and error classes; parity tests confirm exported symbol sets.
- [ ] Optional plugin parameters (e.g., AutoAPI, Sphinx) are annotated via typed
  Protocols, and the facades degrade gracefully when dependencies are absent.
- [ ] Documentation captures maintenance guidance and upstream contribution steps.

## Out of Scope
- Migrating away from Griffe-based loaders (covered by other roadmap items).
- Rewriting docs scripts beyond the adjustments needed to consume the typed facades.
- Introducing new plugin systems; focus is on typing existing entry points.

## Risks / Mitigations
- **Risk:** Griffe upstream changes could invalidate the stubs.
  - **Mitigation:** Pin the compatible Griffe version, add parity tests, and plan an
    upstream PR so future releases incorporate the typings.
- **Risk:** Optional plugin imports may still trigger runtime ImportError.
  - **Mitigation:** Gate heavy imports behind `typing.TYPE_CHECKING` and provide
    runtime guards that raise descriptive `ImportError` with remediation hints.
- **Risk:** Overly strict overloads could reject legitimate plugin usage.
  - **Mitigation:** Collaborate with docs maintainers, add fixtures representing real
    plugin hooks, and allow extensibility via Protocols rather than narrow concrete
    types.

