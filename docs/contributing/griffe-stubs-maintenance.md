# Griffe Stubs Maintenance and Upstream Contribution

## Overview

This document describes the maintenance workflow for the vendored Griffe type stubs under `stubs/griffe/` and the typed facades in `docs/_types/griffe.py`. These stubs are aligned with **Griffe 1.14.0** and are designed to be contributed upstream to the Griffe project.

## Stub Files

### Location and Structure

```
stubs/griffe/
├── __init__.pyi          # Core types (Object, Module, Class, etc.) and loader
├── exceptions/__init__.pyi   # Exception hierarchy
└── loader/__init__.pyi    # Loader factory helpers
```

### Key Exports

The `stubs/griffe/__init__.pyi` exports:

- **Object Classes**: `Object`, `Module`, `Package`, `Class`, `Function`, `Attribute`, `TypeAlias`, `Alias`, `Docstring`
- **Loader**: `GriffeLoader` (main loader class) and `load()` (module-level function)
- **Exceptions**: `GriffeError` (base), `LoadingError`, `NameResolutionError`, `AliasResolutionError`, `CyclicAliasError`, `UnimportableModuleError`, `BuiltinModuleError`, `ExtensionError`, `ExtensionNotLoadedError`

### Typing Principles

1. **No `Any` Types**: All parameters and return types are explicitly typed using Protocols and concrete types.
2. **Runtime Parity**: Stubs match Griffe 1.14.0 class and method signatures exactly.
3. **Minimal Docstrings**: Per PYI conventions, stub files contain no docstrings.
4. **Exception Hierarchy**: Simplified to match runtime (flat hierarchy under `GriffeError`).

## Typed Facades

### `docs/_types/griffe.py`

This module provides high-level, type-safe facades for docs pipeline integration:

- **Protocols**: `GriffeNode`, `LoaderFacade`, `MemberIterator`, `GriffeFacade` (all `@runtime_checkable`)
- **Factory**: `build_facade(env)` returns a typed facade from a `BuildEnvironment`
- **Optional Dependencies**: `get_autoapi_loader()`, `get_sphinx_loader()` with graceful degradation (raise `ArtifactDependencyError`)

## Maintenance Workflow

### 1. Version Pinning

When upgrading Griffe, update:

1. `uv.lock` (via `uv sync` or direct dependency update)
2. Record the new version in this document
3. Update stub exports/signatures if API changed

**Current Griffe Version**: 1.14.0 (pinned in `uv.lock`)

### 2. Detecting API Changes

Run the regression tests to detect runtime/stub mismatches:

```bash
uv run pytest -xvs tests/docs/test_griffe_facade.py::TestGriffeStubExports
```

Tests verify:
- Symbol exports match runtime
- Method signatures are present
- Class hierarchies align
- Exception hierarchies are correct

### 3. Updating Stubs

If tests fail due to API changes:

1. **Inspect new exports**:
   ```python
   import griffe, inspect
   for name in dir(griffe):
       if not name.startswith('_'):
           print(f"{name}: {inspect.signature(getattr(griffe, name))}")
   ```

2. **Update `stubs/griffe/__init__.pyi`** with:
   - New classes/functions
   - Changed signatures
   - Updated exception hierarchy (if needed)

3. **Run full test suite** to verify parity:
   ```bash
   uv run pytest tests/docs/test_griffe_facade.py -q
   uv run pyright --pythonversion=3.13 stubs/griffe/
   ```

4. **Format and lint**:
   ```bash
   uv run ruff format && uv run ruff check --fix stubs/griffe/
   ```

## Upstream Contribution (Griffe Project)

### Patch Requirements

When submitting to the Griffe project, include:

1. **Stub Files**: Complete `griffe/**/*.pyi` modules
2. **Coverage**: All public APIs from the runtime
3. **Testing**: Regression test demonstrating parity
4. **Changelog**: Entry describing the stubs addition
5. **License**: Confirm stubs are under Griffe's license (typically ISSL or similar)

### Submission Checklist

- [ ] Stubs validated against current Griffe version
- [ ] All regression tests pass
- [ ] Ruff/Pyright/Pyrefly/MyPy clean
- [ ] Docstrings removed (PYI conventions)
- [ ] Exception hierarchies match runtime exactly
- [ ] PR description includes use case (documentation toolchain)
- [ ] Author/maintainer contact information included

### Expected Upstream Changes

Once accepted upstream, kgfoundry can:

1. Remove vendored stubs from `stubs/griffe/`
2. Use Griffe's official stubs from the package (if `py.typed` included)
3. Reduce maintenance burden

### Communication Template

```markdown
# Contribution: Type Stubs for Griffe

## Motivation
The Griffe package is widely used for documentation generation in type-aware tools.
This PR adds comprehensive PYI stub files that enable IDE autocomplete and static
type checking for Griffe's public APIs.

## What's Included
- `griffe/**/*.pyi` modules covering all public exports
- Overloads for `load()` and `GriffeLoader.load()` with precise parameter types
- Exception class hierarchy matching runtime
- Regression tests verifying stub/runtime parity

## Testing
```bash
# Type checking
pyright stubs/
pyright stubs/
# Runtime verification
pytest tests/griffe_stubs_test.py
```

## Notes
- Stubs tested with Griffe 1.14.0
- Compatible with Pyright (strict), Pyrefly, and MyPy (strict baseline)
- Maintainers can test against new releases; we provide regression test suite
```

## Maintenance Cadence

### Quarterly Review

1. Check for new Griffe releases
2. Run regression tests against new version
3. Update stubs if API changed
4. Update this document with version and any notes

### Breaking Change Handling

If Griffe releases a breaking change:

1. Create a feature branch: `griffe-stubs-v<new-version>`
2. Update stubs and regression tests
3. Update kgfoundry's Griffe version constraint
4. Commit with message: `chore: update Griffe stubs for <new-version>`
5. Tag the commit with version metadata

## Related Documentation

- [AGENTS.md](../../AGENTS.md) — Type checking standards and quality gates
- [typing.md](./typing.md) — Project-wide typing conventions
- `openspec/changes/griffe-stubs-hardening-phase1/` — Original specification

## FAQ

### Q: Why not use Griffe's own type hints?

**A**: Griffe uses internal `_internal` modules and dynamic runtime construction. The runtime module doesn't export type information. Stubs provide a clean, stable interface for type checkers.

### Q: Should I update stubs when upgrading Griffe?

**A**: Yes. Run regression tests first to detect changes. Most minor versions are compatible; major versions usually require stub updates.

### Q: Can I use these stubs in other projects?

**A**: Yes! The stubs are vendored but can be distributed. Once contributed upstream, any Python project can use Griffe's official stubs.

### Q: What happens if I find a stub bug?

**A**: Open an issue in kgfoundry referencing the test that fails, and the exact Griffe version. Include the error from Pyright/Pyrefly/MyPy.
