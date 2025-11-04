# Public API Hardening Phase 1 — Implementation Strategy

## Overview

This document outlines the strategic approach for implementing the public-api-hardening-phase1 specification. Given the scope of ~65 violations across 6 phases, implementation is prioritized by impact and dependencies.

## Architectural Prerequisites (Already Complete)

✅ **ConfigurationError** exists in `src/kgfoundry_common/errors/exceptions.py`
✅ **ProblemDetails** infrastructure in `src/kgfoundry_common/errors/problem_details.py`
✅ **Packaging** from docs-tools-packaging-phase1 provides stable module structure
✅ **Typing gates** infrastructure in place for import safety

## Implementation Approach

### Strategy: Phased Rollout with Backward Compatibility

Each phase follows this pattern:
1. **Create config models** with validation in `*_config.py` or `config_models.py`
2. **Create interface Protocols** for caches/collectors in `*/cache/interfaces.py`
3. **Update public APIs** to keyword-only with typed config
4. **Add deprecation wrappers** for old signatures (log warnings, delegate to new API)
5. **Comprehensive tests** for both old and new paths
6. **Update docstrings** with examples and Problem Details references

### Key Principles

- **No breaking changes initially:** Deprecation wrappers allow existing code to continue working while warning consumers
- **Type safety first:** All configs are frozen dataclasses with field validators
- **Observable migration:** Deprecation warnings logged with telemetry counters for tracking cleanup progress
- **Schema-driven:** Problem Details examples live in `schema/examples/problem_details/` as single source of truth

## Phase Priority & Effort Estimate

| Phase | Subsystem | Effort | Violations | Status |
|-------|-----------|--------|-----------|--------|
| 0 | Discovery | ✅ DONE | N/A | Complete |
| 5 | Error Infrastructure | 0.5d | N/A | Core dependency |
| 1 | Docstring Builder | 2-3d | ~20 | Highest priority |
| 2 | Navmap | 1-2d | ~1-2 | High priority |
| 3 | Docs Toolchain | 1d | ~2 | Medium priority |
| 4 | Registry/Search | 1-1.5d | ~8 | Lower priority |
| 6 | Enforcement | 0.5d | N/A | Final cleanup |

**Total Estimate:** ~6-8 days of focused work

## Phase 1 Detailed Plan (Docstring Builder)

### 1.1: Create Config Models (`tools/docstring_builder/config_models.py`)

```python
from dataclasses import dataclass
from enum import Enum

class CachePolicy(Enum):
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"

@dataclass(frozen=True, slots=True)
class DocstringBuildConfig:
    """Configuration for docstring builder operations."""
    cache_policy: CachePolicy = CachePolicy.READ_WRITE
    enable_plugins: bool = True
    emit_diff: bool = False
    timeout_seconds: int = 600
    
    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                "timeout_seconds must be positive",
                context={"timeout_seconds": self.timeout_seconds}
            )
```

### 1.2: Create Cache Interface (`tools/docstring_builder/cache/interfaces.py`)

```python
from typing import Protocol

class DocstringBuilderCache(Protocol):
    """Public cache interface for docstring builder."""
    
    def get(self, key: str) -> CachedDocstring | None: ...
    def put(self, key: str, doc: CachedDocstring) -> None: ...
    def invalidate(self, key: str) -> None: ...
    def stats(self) -> CacheStats: ...
```

Add helper in `cache.py`:
```python
def get_docstring_cache() -> DocstringBuilderCache:
    """Accessor for docstring cache interface."""
    return _GLOBAL_CACHE  # or dependency-injected instance
```

### 1.3: Update Orchestrator API

- Old: `orchestrator.run(file_path, emit_diff=True, skip_cache=False, ...)`
- New: `orchestrator.run_build(*, config: DocstringBuildConfig, cache: DocstringBuilderCache)`
- Deprecation wrapper: `run_legacy(*args, **kwargs)` with warning

### 1.4: Update Tests

- Unit tests for config validation
- Integration tests for both old and new APIs
- Verify deprecation warnings emitted
- Verify cache access goes through interface

## Testing Strategy

Each phase includes:
1. **Config validation tests** (`test_config_models.py`)
2. **API compatibility tests** (`test_public_api_compatibility.py`)
3. **Cache interface contract tests** (`test_cache_interfaces.py`)
4. **Problem Details schema validation** (`test_configuration_problem_details.py`)
5. **Deprecation warning tests** (verify warnings emitted exactly once)

## Enforcement Gates (Phase 6)

Final verification runs:
```bash
uv run ruff format && uv run ruff check --fix  # Zero FBT/SLF in target modules
uv run pyrefly check      # Type safety verified
uv run pyright --warnings --pythonversion=3.13
uv run pyright --warnings --pythonversion=3.13
uv run pytest -q          # All tests passing
python -m tools.check_imports  # Architectural boundaries enforced
make artifacts            # Regenerated docs
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Downstream automation breaks | Deprecation warnings + migration guides in AGENTS.md and CHANGELOG |
| Config validation too strict | Add lenient mode with warnings for backward compat during Phase 2 |
| Cache interface incomplete | Thorough inventory of all access patterns before refactoring |
| Problem Details duplication | Centralize in `kgfoundry_common.errors` with shared helpers |

## Success Criteria

1. ✅ Phase 0: Inventory complete
2. ⏳ Phase 1-4: All FBT/SLF violations addressed with new APIs
3. ⏳ Phase 5: ConfigurationError + Problem Details infrastructure live
4. ⏳ Phase 6: All quality gates green, zero violations, docs updated
5. ⏳ Follow-up: Telemetry dashboard tracks zero legacy usage for two weeks before Phase 2 removal

## Current Status

- **Phase 0:** ✅ COMPLETE
- **Phase 1-6:** Starting implementation
- **Target completion:** Within 8 days
- **Blocker:** None identified

## Next Steps

1. Implement Phase 5 (error infrastructure) if not already complete
2. Begin Phase 1 (Docstring Builder) config models
3. Create cache interface and helper
4. Update orchestrator API with deprecation wrapper
5. Add comprehensive test suite
6. Iterate through Phases 2-4
7. Final enforcement in Phase 6
