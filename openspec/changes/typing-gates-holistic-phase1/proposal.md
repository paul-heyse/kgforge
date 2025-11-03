## Why
Ruff currently reports hundreds of `TC00x` violations across `docs/`, `tools/`, and runtime packages because type-only imports execute at runtime. These imports drag heavy dependencies (FAISS, FastAPI, numpy) into tooling contexts where they are unavailable, creating flaky CLI tooling and violating our layering rules. The absence of a sanctioned typing façade also forces modules to reach into private packages (`_types`, `_cache`), while postponed annotations are inconsistently applied, preventing circular-import relief and blocking `pyright` strict mode from stabilising. Without a structural plan we cannot guarantee deterministic imports, nor can we enforce the "type-only" contract across new modules.

## What Changes
- [ ] **ADDED**: `code-quality/typing-gates` capability spec defining postponed-annotation adoption, sanctioned typing façades, and lint/type-check gates for TYPE_CHECKING blocks.
- [ ] **ADDED**: Shared typing infrastructure (`kgfoundry_common.typing`, tooling mirrors) that centralises TYPE_CHECKING exports, safe re-export helpers, and utilities for deferred imports.
- [ ] **MODIFIED**: Ruff configuration, pyproject extras, and project scaffolding to require `from __future__ import annotations` (or Python 3.13 native postponed annotations) and to lint TYPE_CHECKING boundaries via custom Ruff rules.
- [ ] **MODIFIED**: Docs/tooling script packages to move all third-party and heavy imports behind typing gates and to consume the new façade instead of private modules.
- [ ] **ADDED**: Regression tests and CI hooks (`python -m tools.lint.check_typing_gates`) that assert zero runtime-only imports sneak past TYPE_CHECKING blocks and that generated artifacts remain import-clean.
- [ ] **REMOVED**: Ad-hoc private-module imports (`docs._types`, `_cache`, `_ParameterKind`) in favour of public façades.

## Impact
- **Packages affected**: `kgfoundry_common`, `tools`, `docs/_scripts`, `docs/types`, `src/kgfoundry/**`, plus associated stub packages.
- **Tooling**: new helper CLI to verify gating, Ruff config updates, CI gate additions, and migration of existing scripts to module packages with `__init__.py`.
- **Docs & Artifacts**: regeneration required after moving imports and adding explicit package markers; ensure `make artifacts` runs remain deterministic.
- **Testing**: new regression suite ensuring runtime imports succeed without optional deps installed, plus doctest/xdoctest updates to reflect postponed annotations usage.

- [ ] Ruff (`uv run ruff format && uv run ruff check --fix`) reports zero `TC00x`, `INP001`, `EXE00x`, and `PLC2701` violations in the targeted packages.
- [ ] Pyright, Pyrefly, and MyPy complete without new suppressions, confirming postponed annotations and typing façades do not introduce `Any` fallbacks.
- [ ] `python -m tools.lint.check_typing_gates` and accompanying pytest suites enforce TYPE_CHECKING guards and ensure runtime imports remain side-effect free.
- [ ] Docs/tooling CLIs execute without optional dependencies (FAISS, FastAPI) installed, proving deferred imports succeed.

## Out of Scope
- Replacing FastAPI, FAISS, or other third-party libraries.
- Adjusting business-domain logic beyond import/type hygiene.
- Revisiting existing Problem Details payloads (only touched when import changes demand it).

## Risks / Mitigations
- **Risk:** Deferred imports might hide missing runtime dependencies until later execution.  
  **Mitigation:** add smoke tests that execute CLI entry points after TYPE_CHECKING refactors and document required runtime deps in module docstrings.
- **Risk:** Widespread `__future__` additions could conflict with existing module headers.  
  **Mitigation:** provide automated fixer and format gate ensuring the directive precedes other statements; review via targeted Ruff rule.
- **Risk:** Tooling scripts may rely on private modules removed during refactor.  
  **Mitigation:** introduce public façade equivalents with compatibility shims and deprecation warnings before removing private paths.

## Alternatives Considered
- **Ad-hoc suppression of Ruff TC00x rules:** rejected; hides structural problems and blocks future strict typing phases.
- **Selective postponed annotations (only runtime packages):** rejected; tooling would remain fragile and inconsistent, and cross-package contracts would diverge.
- **Runtime dependency vendoring for docs/scripts:** rejected; increases maintenance burden and still fails for optional GPU/HTTP stacks.


