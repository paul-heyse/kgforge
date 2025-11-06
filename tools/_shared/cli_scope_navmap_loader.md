# Migration Scope: `kgfoundry_common.navmap_loader`

`kgfoundry_common.navmap_loader.load_nav_metadata` is widely used by Typer modules (e.g.
`src/orchestration/cli.py`, `src/download/cli.py`) to hydrate `__navmap__` metadata for downstream
documentation and navigation tooling. Right now the loader reads bespoke `_nav.json` sidecars and
falls back to runtime `__navmap__` dictionaries. To keep navigation metadata in lock-step with the
new CLI tooling contracts (augment + registry + Pydantic metadata), we need to refactor
`navmap_loader` so it consumes the same shared sources and exposes a single canonical interface.

## Current State

- Sidecar format is ad-hoc; schema is loosely enforced and diverges from `_augment_cli.yaml` /
  `api_registry.yaml`.
- Runtime modules (including CLIs) often define both `__navmap__` and Typer metadata separately,
  leading to drift.
- No validation is performed; missing keys silently fall back to generated placeholders.
- Downstream tooling (navmap builders, MkDocs generators) depends on the loader’s output, so any
  discrepancies propagate into docs.

## Migration Goals

1. **Derive nav metadata from the shared CLI tooling metadata (augment/registry) whenever
   available**, removing the need for bespoke `_nav.json` content for CLI packages.
2. **Retain backward compatibility for non-CLI packages** that still use sidecars, but validate them
   rigorously.
3. **Expose a richer, typed object** that downstream tooling can consume without re-parsing raw
   dictionaries (align with Pydantic models used elsewhere).
4. **Eliminate duplicate metadata definitions** so CLI packages rely exclusively on `_augment_cli.yaml`
   and `api_registry.yaml` for navigation, tags, summaries, and Problem Details references.

## Detailed Implementation Plan

### 1. Augment/Registry Integration

- Add optional dependency on the shared CLI metadata loader:
  - Introduce a helper (e.g., `_load_cli_metadata(package: str) -> ToolingMetadataModel | None`) that
    resolves a package’s interface ID via `api_registry.yaml` and, when present, returns the relevant
    `ToolingMetadataModel` instance.
  - Extend `load_nav_metadata` so when CLI metadata exists for the package, navmap sections and
    symbol metadata are *derived* from `ToolingMetadataModel` (`augment.tag_groups`,
    `augment.operation_override`, `registry.interface(...).extras`).
- Map augment data to navmap fields:
  - `sections`: create entries based on augment tag groups (`name`, `description`, `tags`).
  - `symbols`: for each exported operation/command, populate `summary`, `handler`, and Problem
    Details links from augment.
  - `module_meta`: use registry interface metadata (owner, stability, spec) and include CLI-specific
    extras (binary name, environment variables).
- Provide warnings (via shared logger) when augment/registry entries are missing; fail fast in CI
  using tests to ensure metadata completeness.

### 2. Sidecar Validation & Compatibility

- Introduce a Pydantic model (e.g., `NavMetadataModel`) representing the navmap structure. Use it to
  validate both derived metadata and existing sidecars.
- If a package lacks CLI metadata *and* a sidecar, continue generating minimal exports-based
  fallback but emit a structured warning.
- When a sidecar is present but conflicts with CLI metadata for the same package, prefer the shared
  metadata and log a deprecation warning; add a configuration option to disable legacy sidecars
  entirely once the migration is complete.
- Add unit tests covering:
  - CLI package with augment/registry entries (no sidecar) → navmap derived from CLI metadata.
  - Package with sidecar only → validated via Pydantic model; runtime `__navmap__` fallback.
  - Error cases (invalid sidecar fields) → raise `NavMetadataError` with Problem Details envelope.

### 3. Typed Return Object

- Refactor `load_nav_metadata` to return an instance of `NavMetadataModel` (which implements
  `.model_dump()` to preserve the current dict interface). Update callers gradually to use typed
  attributes.
- Provide conversion helpers for existing code that expects a dict (`return nav_metadata.model_dump()`
  or implement `__getitem__/__iter__` on the model for seamless drop-in usage).
- Update documentation and `__all__` exports in CLI packages to instantiate `__navmap__` from the
  typed model rather than copying dictionaries.

### 4. Downstream Tooling Adjustments

- **Navmap build scripts** (`tools/navmap/build_navmap.py`, `repair_navmaps.py`, etc.) should be
  updated to expect `NavMetadataModel` and operate inside the new doc toolchain lifecycle helpers
  (`docs/toolchain/_shared/lifecycle.py` once merged). Use the model’s methods to access
  sections/symbols instead of direct dict indexing, and emit logs/metrics via `DocLifecycle` per the
  `docs-toolchain-lifecycle` spec.
- **MkDocs generators** (`gen_module_pages.py`, `gen_interface_pages.py`) can use the typed metadata
  to simplify logic—e.g., retrieving tag groups or operation summaries without manual key checks—and
  when they are migrated onto the doc lifecycle, reuse the same context/logging to stay consistent
  with CLI documentation tooling.
- **OpenAPI/CLI generators** already rely on augment/registry; ensure they read nav metadata through
  the same facade to avoid duplication.
- Document the new contract in developer docs (AGENTS + Navmap section) so teams know sidecar files
  are deprecated for CLI metadata; future nav updates should go through augment/registry.

### 5. Cleanup & Enforcement

- After migrating CLI packages, remove redundant `_nav.json` files if they only contained CLI data.
- Add a lint rule (or pytests) to flag direct sidecar loads outside `kgfoundry_common.navmap_loader`.
- Provide a codemod or helper to convert existing sidecars into augment/registry entries where
  applicable.

## Acceptance Criteria

- CLI packages (`src/orchestration`, `src/download`, future ones) obtain nav metadata solely from
  `ToolingMetadataModel`; no sidecar files required.
- `load_nav_metadata` returns a validated model; downstream tooling updated accordingly.
- Legacy sidecar support is preserved for non-CLI packages but validated; clear warnings when used.
- Documentation updated to describe the new metadata flow; tests cover both CLI-derived and
  sidecar-derived nav metadata.
- Legacy code paths (direct JSON loads, manual `__navmap__` dict assembly) removed.

Once these steps are implemented, `kgfoundry_common.navmap_loader` becomes the bridge between the new
CLI metadata contracts and existing documentation tooling, eliminating duplication and reducing the
risk of drift across the ecosystem.


