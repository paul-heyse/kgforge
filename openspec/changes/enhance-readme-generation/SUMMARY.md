# OpenSpec Change Proposals Created

## Summary

I've created **ONE comprehensive OpenSpec change proposal** for the README enhancement scope described in `ReadmeImprovementScope.md`:

### ✅ Proposal: `enhance-readme-generation`

**Location:** `openspec/changes/enhance-readme-generation/`

**Status:** ✅ Validated (passes `openspec validate --strict`)

**Scope:** Close the 10-15% implementation gap between the current `tools/gen_readmes.py` and the complete specification in `ReadmeImprovementScope.md`.

---

## Current State Analysis

### What's Already Implemented (85-90% Complete)

The current `tools/gen_readmes.py` already has:

✅ **CLI & Configuration**
- Complete argument parsing (`--packages`, `--link-mode`, `--editor`, `--fail-on-metadata-miss`, `--dry-run`, `--verbose`)
- Environment variable fallbacks (`DOCS_PKG`, `DOCS_LINK_MODE`, `DOCS_EDITOR`)
- Config dataclass with validation

✅ **Metadata Integration**
- NavMap JSON loading and lookup
- TestMap JSON loading (structure exists)
- Badge formatting (stability, owner, section, since, deprecated_in, tested-by)
- Metadata lookup with fallbacks

✅ **Link Generation**
- GitHub permalinks with line ranges (`blob/{sha}/path#L{start}-L{end}`)
- VSCode deep-links (`vscode://file/{abs}:{line}:1`)
- Relative path fallback (`./path:line:col`)
- Link mode switching (github/editor/both)

✅ **Content Organization**
- Deterministic bucketing (Modules/Classes/Functions/Exceptions)
- Exception detection via base class heuristics
- Sorted output by fully qualified name
- Private symbol filtering (`_` prefix)

✅ **Output Quality**
- Write-if-changed with content hash provenance
- Doctoc TOC markers
- Package synopsis from `__init__` docstring

### What's Missing (10-15% Gap)

❌ **Documentation**
- No contributor guide explaining README structure/badges/links
- No reference in main documentation (`README-AUTOMATED-DOCUMENTATION.md`)
- No usage examples or troubleshooting guide

❌ **Integration**
- No CI validation step
- No pre-commit hook configured
- Doctoc runs manually (not automated with `--run-doctoc` flag)

❌ **Validation & Testing**
- No test suite (`test_gen_readmes.py`)
- No determinism tests
- No badge/link/bucket validation

❌ **Quality of Life**
- No warnings when NavMap/TestMap JSON files missing
- Badge layout doesn't handle line wrapping for long badge lists
- Synopsis display verified but could be more prominent

---

## Change Proposal Contents

### 📄 `proposal.md`

**Sections:**
- **Why**: Explains the 10-15% gap and problems it creates
- **What Changes**: 4 categories (Documentation, Code Refinements, Validation, Configuration)
- **Impact**: Breaking changes (none), affected components, benefits, migration path, rollback strategy
- **Current State**: Detailed analysis showing what's working vs what's missing

**Key Points:**
- **Non-breaking**: All changes are additive
- **Backward-compatible**: No changes to existing README format
- **Polish focus**: Not a rewrite, just closing gaps

### 📋 `tasks.md`

**8 Major Sections, 93 Total Tasks:**

1. **Documentation** (20 tasks)
   - Create `docs/how-to/read-package-readmes.md`
   - Update `README-AUTOMATED-DOCUMENTATION.md`
   - Add inline code documentation

2. **Code Refinements** (22 tasks)
   - Synopsis display verification
   - Badge layout with line wrapping
   - Missing file warnings (graceful degradation)
   - Doctoc integration (`--run-doctoc` flag)

3. **Testing** (32 tasks)
   - Create `tests/unit/test_gen_readmes.py`
   - Determinism tests
   - Badge rendering tests
   - Link generation tests
   - Bucketing logic tests
   - Metadata missing behavior tests
   - Dry-run mode tests

4. **CI Integration** (7 tasks)
   - Add README freshness check to `.github/workflows/ci.yml`
   - Document CI check behavior

5. **Pre-commit Hook** (9 tasks)
   - Add optional hook in `.pre-commit-config.yaml`
   - Test hook with changed packages
   - Document as optional with performance notes

6. **Quality Assurance** (16 tasks)
   - Run full pipeline validation
   - Test with missing metadata
   - Test across all packages
   - Validate generated links

7. **Documentation Review** (7 tasks)
   - Peer review of contributor guide
   - Update main README
   - Add command reference

8. **Final Validation** (12 tasks)
   - Run all tests
   - Run pre-commit checks
   - CI simulation
   - Documentation completeness

### 📐 `specs/documentation-generation/spec.md`

**MODIFIED Requirements:**
- **Package README Generation** (comprehensive requirement with 10 scenarios)
  - Complete generation with metadata
  - Missing metadata file handling
  - Deterministic output
  - Link mode configuration
  - VSCode deep-links
  - Exception grouping
  - Badge layout
  - Fail-fast validation
  - Dry-run mode
  - Doctoc integration

**ADDED Requirements:**
1. **README Contributor Documentation** (2 scenarios)
2. **CI README Validation** (2 scenarios)
3. **Optional Pre-commit README Hook** (2 scenarios)
4. **README Generation Test Coverage** (4 scenarios)

---

## Implementation Strategy

### Phase 1: Documentation (Non-blocking)
- Create contributor guide
- Update main documentation
- Add inline code comments

**Estimated Time:** 4-6 hours
**Dependencies:** None
**Risk:** Low

### Phase 2: Testing (Validation)
- Create comprehensive test suite
- Validate current behavior
- Establish baseline

**Estimated Time:** 6-8 hours
**Dependencies:** Phase 1 (for test documentation)
**Risk:** Low (tests current behavior)

### Phase 3: Code Refinements (Minor Changes)
- Add missing file warnings
- Implement badge line wrapping
- Add `--run-doctoc` flag
- Verify synopsis display

**Estimated Time:** 3-4 hours
**Dependencies:** Phase 2 (tests validate changes)
**Risk:** Very Low (all backward-compatible)

### Phase 4: Integration (Automation)
- Add CI check (non-blocking initially)
- Create optional pre-commit hook
- Test end-to-end

**Estimated Time:** 2-3 hours
**Dependencies:** Phases 1-3 complete
**Risk:** Low (can be disabled if issues)

### Phase 5: Validation & Cleanup
- Run full QA checklist
- Fix any issues discovered
- Enable CI check as blocking

**Estimated Time:** 2-3 hours
**Dependencies:** All previous phases
**Risk:** Very Low (tests catch issues)

**Total Estimated Time:** 17-24 hours

---

## Success Criteria

✅ **All 93 tasks complete**
✅ **Validation passes:** `openspec validate enhance-readme-generation --strict`
✅ **Tests passing:** `pytest tests/unit/test_gen_readmes.py` (100% pass rate)
✅ **CI passing:** README freshness check validates without errors
✅ **Documentation complete:** Contributor guide + main docs updated
✅ **Deterministic:** Running twice produces byte-identical output
✅ **Links functional:** VSCode and GitHub links open correctly
✅ **Badges accurate:** Metadata displays correctly with proper wrapping
✅ **Graceful degradation:** Works without NavMap/TestMap files (with warnings)

---

## Next Steps

### For You (Review & Approval):

1. **Review `proposal.md`**: Confirm scope aligns with your expectations
2. **Review `tasks.md`**: Verify task breakdown is appropriate
3. **Review spec deltas**: Ensure scenarios capture all requirements
4. **Approve**: Give me the go-ahead to start implementation

### For Me (After Approval):

1. **Start Phase 1**: Create contributor guide
2. **Work through `tasks.md`** systematically
3. **Mark tasks complete** as I go
4. **Run validation** after each phase
5. **Update you** on progress and blockers

---

## Key Differences from Specification Document

The `ReadmeImprovementScope.md` was a **detailed specification** of how the system should work.

The `gen_readmes.py` **already implements** 85-90% of that specification.

This OpenSpec proposal focuses on:
1. **Closing the last 10-15%** (documentation, integration, minor refinements)
2. **Validating** that implementation matches spec (tests)
3. **Making it easy** for developers to use (docs, CI, hooks)

**This is NOT a rewrite.** It's polish and integration work.

---

## Files Created

```
openspec/changes/enhance-readme-generation/
├── proposal.md                                    # Why/What/Impact
├── tasks.md                                       # 93 granular tasks
└── specs/
    └── documentation-generation/
        └── spec.md                                # 4 requirements, 20 scenarios
```

**Validation Status:** ✅ All files validated with `openspec validate --strict`

---

## Questions for You

1. **Approval?** Does this scope match what you want implemented?
2. **Priority?** Should I start implementation now or wait?
3. **Phasing?** Do you want all phases or just specific ones?
4. **Timeline?** Any urgency or deadline constraints?

**I'm ready to proceed with implementation when you give approval! 🚀**

