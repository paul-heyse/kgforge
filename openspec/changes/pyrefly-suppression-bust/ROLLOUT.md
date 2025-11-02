# Pyrefly-Suppression-Bust Rollout Plan

**Date:** November 2, 2025  
**Status:** Ready for merge  
**Change ID:** pyrefly-suppression-bust  
**Validation:** ✅ OpenSpec strict validation passed

---

## Summary

This change hardens type safety across kgfoundry by eliminating unmanaged type suppression pragmas (`# type: ignore` and `# pyrefly: ignore`) and replacing them with typed facades, protocol-based adapters, and structured error handling.

**Impact Scope:**
- ✅ **Agent Catalog & Search**: Refactored to typed APIs with FAISS/NumPy helpers
- ✅ **Tooling & Docs Scripts**: Introduced typed JSON facades and CLI entry points
- ✅ **Governance**: New suppression check script enforces ticket references

**Breaking Changes:** None (all changes are internal; public APIs remain stable).

---

## Quality Assurance

All quality gates passing:

```bash
✓ ruff format && ruff check --fix     (20+ files, 0 issues)
✓ pyrefly check                       (0 errors across modified modules)
✓ mypy --config-file mypy.ini         (0 errors across modified modules)
✓ pytest -q                           (48 passed in 0.36s)
✓ openspec validate --strict          (valid)
```

---

## Affected Modules

### High-Impact Changes
1. **src/kgfoundry/agent_catalog/search.py**
   - Refactored vector scoring to consume typed NumPy helpers
   - Eliminated `type: ignore` pragmas via proper protocol definitions
   - Added `__all__` for explicit public API

2. **docs/_scripts/shared.py** (+141 lines)
   - New: `safe_json_serialize()` with atomic writes
   - New: `safe_json_deserialize()` with type validation
   - Both include comprehensive docstrings and error handling

3. **scripts/check_pyrefly_suppressions.py** (NEW)
   - Governance script that scans for unmanaged suppressions
   - Enforces ticket references (e.g., `# type: ignore[misc] - ticket #456`)
   - Designed for pre-commit and CI gates

### Medium-Impact Changes
- `docs/_scripts/build_symbol_index.py`: Typed `main()` signature
- `docs/_scripts/symbol_delta.py`: Refactored JSON handling
- `docs/_scripts/validation.py`: Schema loading with structured logging
- `src/kgfoundry_common/prometheus.py`: Fixed histogram builder None-handling
- `openspec/AGENTS.md`: New "Safe Tooling Practices" section with examples

### Low-Impact Changes
- Various import/comment cleanup (removed `noqa: PLC2701` where appropriate)
- Type annotations added to CLI entry points

---

## Migration Guide

### For Developers

1. **When adding new suppressions**, include a ticket reference:
   ```python
   # ✅ Good: includes ticket reference
   result = foo()  # type: ignore[misc] - ticket #789
   
   # ❌ Bad: no rationale
   result = foo()  # type: ignore
   ```

2. **Use typed JSON utilities** in docs/tooling scripts:
   ```python
   from docs._scripts import shared
   
   # Safe deserialization with error handling
   data = shared.safe_json_deserialize(path, logger=logger)
   if data is None:
       raise ValueError(f"Failed to load {path}")
   ```

3. **Implement CLI entry points** with proper signatures:
   ```python
   def main(argv: Sequence[str] | None = None) -> int:
       """Document your tool's purpose and outputs."""
       # Implementation here
       return 0
   ```

### For CI/CD Teams

1. **Pre-commit hook integration:**
   ```bash
   # Run before merge
   python scripts/check_pyrefly_suppressions.py src/ docs/_scripts/
   ```

2. **CI job integration** (add to lint stage):
   ```bash
   uv run ruff format && uv run ruff check --fix
   uv run pyrefly check
   uv run mypy --config-file mypy.ini
   python scripts/check_pyrefly_suppressions.py src/
   ```

3. **Suppression escalation path:**
   - If a suppression is truly unavoidable, open an issue and reference it
   - Discuss with the Python steering committee before merging
   - Document the root cause in the ticket

---

## Rollout Checklist

- [x] **Phase 1 (1.1–1.6):** Implementation complete
- [x] **Phase 2 (2.1–2.4):** All quality gates passing
- [x] **Phase 3 (3.1–3.3):** Documentation updated and validated
- [ ] **Phase 4.1:** Communicate change to affected teams (search, tooling, docs)
- [ ] **Phase 4.2:** Monitor CI for regressions post-merge

---

## Communication Checklist

**To:** Search, Tooling, Documentation teams  
**From:** Python & Quality Engineering  
**Subject:** Pyrefly-Suppression-Bust Rollout — Type Safety Hardening

**Message Template:**

> The `pyrefly-suppression-bust` change has been approved and is ready for merge.
> 
> **What's Changing:**
> - Type checking is now stricter; unmanaged suppressions are caught at merge time
> - New governance: all `# type: ignore` pragmas must include a ticket reference
> - New tools: `scripts/check_pyrefly_suppressions.py` enforces this in CI
> 
> **Action Required:**
> - Review the "Safe Tooling Practices" section in `openspec/AGENTS.md`
> - Update any scripts to use typed JSON facades (see `docs/_scripts/shared.py`)
> - Ensure your CLI tools export a typed `main(argv)` entry point
> 
> **Zero Breaking Changes:** All public APIs remain stable; this only affects internal tooling and type safety.
> 
> **Questions?** Link to `openspec/changes/pyrefly-suppression-bust/design.md`

---

## Regression Monitoring

**Post-Merge Checks (24–48 hours):**
- [ ] Monitor CI pyrefly/mypy pass rates
- [ ] Check for any flaky suppression check failures
- [ ] Review any new issues opened by downstream tooling
- [ ] Validate that optional dependency fallbacks (GPU, FAISS) still work

**Metrics to Track:**
- PyRefly error count (target: 0)
- MyPy error count (target: 0)
- Unmanaged suppression count (target: 0)
- CI pipeline success rate (target: >99%)

---

## Rollback Plan

If critical issues arise post-merge:
1. Revert the commit
2. File a follow-up issue with the failing test case
3. Post a brief message to #eng-python explaining the issue
4. Plan a targeted fix in a new OpenSpec change

---

## References

- **Design Document:** `openspec/changes/pyrefly-suppression-bust/design.md`
- **Implementation Tasks:** `openspec/changes/pyrefly-suppression-bust/tasks.md`
- **Governance:** `openspec/AGENTS.md` → "Safe Tooling Practices"
- **Suppression Checker:** `scripts/check_pyrefly_suppressions.py`
- **Tooling Facades:** `docs/_scripts/shared.py`
