## Context
- `_namespace_proxy.py` uses `Any` sentinels and dynamic attribute lookups, leading to unchecked exports and Pyrefly/Mypy warnings.
- Stub packages (e.g., `stubs/kgfoundry/agent_catalog/search.pyi`) contain outdated symbol lists, `Any` annotations, and `# type: ignore` directives to silence missing exports.
- Search modules rely on redundant `cast(...)` and lingering `# type: ignore` comments, masking underlying typing issues.

## Goals / Non-Goals
- **Goals**
  - Replace `Any` usage in namespace proxy with typed registries or lazy loading constructs.
  - Align stub files with runtime exports using `type[...]`, `Protocol`, or precise generics, removing redundant ignores.
  - Clean up search modules by removing unnecessary casts/ignores via precise typing.
- **Non-Goals**
  - Introducing new runtime exports (only align existing ones).
  - Overhauling search algorithms or data models.
  - Automating stub generation.

## Decisions
- Implement a `NamespaceRegistry` dataclass encapsulating symbol metadata (name, loader callable) and expose typed methods `register` / `resolve`.
- Use `functools.lru_cache` or explicit dictionaries keyed by module path to implement lazy loading without `Any`.
- Mirror runtime exports in stubs: define `type SearchOptions = kgfoundry.agent_catalog.search.SearchOptions` or typed Protocols when classes expose runtime methods.
- Remove `# type: ignore` entries by tightening function signatures (e.g., annotate helper return types) and verifying type checkers pass.
- Update tests to ensure `from kgfoundry import *` still behaves correctly with typed namespace proxy.

## Alternatives
- Keep dynamic proxy but add targeted type ignores — rejected, fails to meet strict typing goals.
- Replace stubs with pure runtime imports — rejected because stubs provide faster type checking for tooling.
- Use import hooks — rejected as unnecessary complexity for this scope.

## Risks / Trade-offs
- Typed registry might introduce slight overhead on first resolution.
  - Mitigation: Cache resolved modules and retain lazy loading semantics.
- Removing `Any` may require broader typing fixes.
  - Mitigation: Update call sites with precise types, add minimal helper types as needed.
- Stub alignment must remain in sync with future runtime changes.
  - Mitigation: Document workflow in contributing guide and add tests verifying stub/runtime parity.

## Migration
- Refactor namespace proxy first, ensuring baseline tests pass.
- Update stubs to match runtime exports; run Pyrefly/Mypy to confirm alignment.
- Remove redundant casts/ignores and add regression tests verifying helper outputs remain typed.
- Document the workflow for future contributors in `docs/contributing/typing.md`.

