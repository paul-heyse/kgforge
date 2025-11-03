# Public API Hardening Phase 1 — Inventory

## Summary

This inventory documents all FBT (boolean-typed positional arguments) and SLF (private member access) violations found in the codebase, organized by subsystem and priority for refactoring.

**Total violations found:** ~50 FBT violations, ~15 SLF violations across docs, tools, and src

## Subsystem: Docstring Builder

### FBT Violations

| File | Line | Type | Current Signature | Proposed Config |
|------|------|------|-------------------|-----------------|
| `tools/docstring_builder/config.py` | 116 | FBT001/002 | `def update_config(..., emit_diff: bool = False, ...)` | `DocstringBuildConfig.emit_diff: bool` |
| `tools/docstring_builder/apply.py` | 141 | FBT001/002 | `def apply(..., diff_required: bool = False)` | `DocstringApplyConfig` |
| `tools/docstring_builder/file_processor.py` | 261 | FBT001 | `def process_file(..., skip_existing: bool)` | `FileProcessConfig` |
| `tools/docstring_builder/file_processor.py` | 211-232 | FBT003 | Calls with positional bool | Move to config object |

### SLF Violations

| File | Line | Member | Proposed Interface |
|------|------|--------|-------------------|
| `tools/docstring_builder/harvest.py` | 103-107 | `_ParameterKind` | `ParameterKindInterface` Protocol |

### Proposed Changes

**Config Models:**
- `tools/docstring_builder/config_models.py`:
  - `DocstringBuildConfig(frozen=True, slots=True)` with fields: `emit_diff`, `enable_plugins`, `timeout_seconds`, etc.
  - `FileProcessConfig` for file processing options
  - `DocstringApplyConfig` for apply operations

**Cache Interface:**
- `tools/docstring_builder/cache/interfaces.py`:
  - `DocstringBuilderCache` Protocol with `get()`, `put()`, `invalidate()` methods
  - Helper: `get_docstring_cache()` returning the interface

**API Changes:**
- Refactor `orchestrator.run()` to `run_build(*, config: DocstringBuildConfig, cache: DocstringBuilderCache)`
- Add deprecation wrapper for old signature

---

## Subsystem: Navmap Toolkit

### FBT Violations

**Note:** Navmap currently has no FBT violations (no boolean positional args in public APIs)

### SLF Violations

| File | Line | Member | Proposed Interface |
|------|------|--------|-------------------|
| `tools/navmap/repair_navmaps.py` | 52 | `_collect_module` | `NavmapCollector` Protocol |

### Proposed Changes

**Config Models:**
- `tools/navmap/config.py`:
  - `NavmapRepairOptions(frozen=True)` with fields: `force`, `dry_run`, `timeout`
  - `NavmapStripOptions(frozen=True)` for strip operations

**Cache/Collector Interface:**
- `tools/navmap/cache/interfaces.py`:
  - `NavmapCollector` Protocol with `collect_module()`, `get_stats()` methods
  - Helper: `get_navmap_collector()` returning the interface

**API Changes:**
- Update `repair_navmaps()` signature to accept `*, options: NavmapRepairOptions`
- Update `strip_navmap_sections()` similarly
- Add deprecation wrappers

---

## Subsystem: Docs Toolchain

### FBT Violations

| File | Line | Type | Current Signature | Proposed Config |
|------|------|------|-------------------|-----------------|
| `docs/_types/artifacts.py` | 262-263 | FBT003 | Function calls with positional bool | Use config object |

### SLF Violations

| File | Line | Member | Proposed Interface |
|------|------|--------|-------------------|
| `docs/conf.py` | 491 | `_parse_file` | Public `parse_file()` helper in `docs.types` |

### Proposed Changes

**Config Models:**
- `docs/toolchain/config.py`:
  - `DocsSymbolIndexConfig(frozen=True)` with fields: `packages`, `output_format`, etc.
  - `DocsDeltaConfig(frozen=True)` for delta generation

**API Changes:**
- Update `build_symbol_index()` to accept `*, config: DocsSymbolIndexConfig`
- Update `symbol_delta()` similarly
- Add deprecation wrappers
- Create helper: `parse_parameter_kind()` returning safe value

---

## Subsystem: Registry & Search

### FBT Violations

| File | Line | Type | Signature | Priority |
|------|------|------|-----------|----------|
| `src/registry/api.py` | 153 | FBT001 | `def create(..., cache_enabled: bool)` | Phase 4 (Low) |
| `src/registry/duckdb_registry.py` | 187 | FBT001 | Similar | Phase 4 (Low) |
| `src/registry/helper.py` | 153 | FBT001 | Similar | Phase 4 (Low) |
| `src/search_client/client.py` | 345 | FBT001/002 | `def __init__(..., ssl_verify: bool = True)` | Phase 4 (Low) |
| `src/kgfoundry_common/config.py` | 315 | FBT001/002 | `def set_runtime(..., enable_gpu: bool = False)` | Phase 4 (Low) |

### SLF Violations

| File | Line | Member | Priority |
|------|------|--------|----------|
| `src/search_api/vectorstore_factory.py` | 248 | `_cpu_matrix` | Phase 4 (Low) |
| `src/kgfoundry/agent_catalog/search.py` | 942, 999 | `_vectors` | Phase 4 (Low) |

### Proposed Changes

**Config Models:**
- `src/orchestration/config.py`:
  - `IndexCliConfig` for CLI index operations
  - `ArtifactValidationConfig` for validation

**Priority:** Phase 4 (lower priority, fewer violations than docstring_builder)

---

## Subsystem: Tools/Codemods

### FBT Violations

| File | Line | Type | Signature | Note |
|------|------|------|-----------|------|
| `tools/codemods/blind_except_fix.py` | 131 | FBT001/002 | Generic codemod visitor | Can be skipped (internal tooling) |
| `tools/codemods/pathlib_fix.py` | 359 | FBT001/002 | Generic codemod visitor | Can be skipped (internal tooling) |

**Note:** These are internal codemod tools, not part of public API. Can defer or skip.

---

## Subsystem: Docs Build

### FBT Violations

| File | Line | Type | Signature | Priority |
|------|------|------|-----------|----------|
| `tools/docs/build_graphs.py` | Multiple | FBT001/003 | Helper functions for graph building | Phase 6 (Low) |
| `tools/docs/scan_observability.py` | 308 | FBT001 | Observability scanner | Phase 6 (Low) |

**Priority:** Phase 6 (lowest, supporting tools)

---

## Subsystem: Auto Docstrings

### SLF Violations

| File | Line | Members | Note |
|------|------|---------|------|
| `tools/auto_docstrings.py` | 30-74 | Multiple `_` prefixed members from third-party | External dependency (docformatter), cannot be changed |

**Status:** SKIP (third-party library internals)

---

## Implementation Priority

1. **Phase 1 (HIGH):** Docstring Builder (~20 violations)
2. **Phase 2 (HIGH):** Navmap Toolkit (~1 violation + configs)
3. **Phase 3 (MEDIUM):** Docs Toolchain (~2 violations)
4. **Phase 4 (LOW):** Registry, Search, Orchestration (~8 violations)
5. **Phase 5 (HIGH):** ConfigurationError & Problem Details infrastructure
6. **Phase 6 (MEDIUM):** Enforcement & cleanup
7. **SKIP:** Codemods, auto_docstrings (internal/external)

---

## Configuration Error & Problem Details

All refactored modules will raise `ConfigurationError` (to be added to `src/kgfoundry_common/errors/exceptions.py`) with structured Problem Details as per RFC 9457.

Schema example: `schema/examples/problem_details/public-api-invalid-config.json`

---

## Status

- [x] Phase 0.1: Inventory complete
- [ ] Phase 0.2: Stakeholder sign-off (pending review)
- [ ] Phase 1: Docstring Builder implementation
- [ ] Phase 2: Navmap implementation
- [ ] Phase 3: Docs Toolchain implementation
- [ ] Phase 4: Registry/Search implementation
- [ ] Phase 5: ConfigurationError infrastructure
- [ ] Phase 6: Enforcement & docs
