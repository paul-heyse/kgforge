# Proposal: Enhance README Generation with Complete Metadata Integration

## Why

The current README generation system (`tools/gen_readmes.py`) has been partially updated with metadata badges and improved link generation, but there remain gaps between the current implementation and the comprehensive specification in `ReadmeImprovementScope.md`. These gaps create inconsistencies and limit the utility of generated READMEs for both humans and AI agents.

### Current State Analysis

**What's Already Implemented:**
- ✅ CLI argument parsing with `--packages`, `--link-mode`, `--editor`, `--fail-on-metadata-miss`, `--dry-run`, `--verbose`
- ✅ Environment variable fallbacks (`DOCS_PKG`, `DOCS_LINK_MODE`, `DOCS_EDITOR`)
- ✅ NavMap JSON loading and metadata extraction
- ✅ Test map JSON loading (structure exists)
- ✅ Badge formatting for stability, owner, section, since, deprecated_in, tested-by
- ✅ VSCode deep-link generation (`vscode://file/...`)
- ✅ Deterministic grouping (Modules/Classes/Functions/Exceptions)
- ✅ Exception detection via base class heuristics
- ✅ Write-if-changed with content hash provenance footer
- ✅ Doctoc markers for TOC generation
- ✅ GitHub permalink generation with line ranges

**What's Missing or Needs Refinement:**
- ❌ Package synopsis extraction (first line of `__init__` docstring) not displayed prominently
- ❌ Badge formatting doesn't handle newlines properly (all badges on one line vs spec shows multi-line)
- ❌ No explicit validation that test_map.json exists before referencing it
- ❌ No sub-package recursion explicitly documented in behavior
- ❌ CI integration examples not documented in repo
- ❌ Pre-commit hook not configured
- ❌ No "How to read package READMEs" contributor documentation
- ❌ Doctoc integration not automated (manual step)

### Problems This Solves

1. **Incomplete Implementation**: Current code has most features but lacks documentation, integration, and validation of assumptions.

2. **CI/CD Gap**: No automated check that READMEs stay in sync with code changes.

3. **Documentation Debt**: No guide for contributors explaining README structure, badges, or how to use the links.

4. **Inconsistent Updates**: Developers don't know when to regenerate READMEs or how to validate them.

5. **Missing Guardrails**: Test map JSON might not exist but code assumes it might; no graceful degradation messaging.

## What Changes

### 1. Documentation & Integration

- **Add contributor guide**: `docs/how-to/read-package-readmes.md` explaining badge meanings, link types, update workflow
- **Add CI check**: GitHub Actions workflow step that validates README freshness
- **Add pre-commit hook**: Optional hook for changed packages only (fast)
- **Update main documentation**: Reference README system in `README-AUTOMATED-DOCUMENTATION.md`

### 2. Code Refinements

- **Synopsis Display**: Ensure package synopsis (from `__init__` docstring) displays prominently below H1
- **Badge Layout**: Add newline handling when badges are numerous (keep readable, don't exceed 120 chars)
- **Missing File Warnings**: Log warnings (not errors) when `test_map.json` doesn't exist
- **Doctoc Integration**: Add optional `--run-doctoc` flag to automatically run doctoc after generation

### 3. Validation & Quality

- **Add determinism test**: Test that running twice produces identical output
- **Add badge test**: Verify badges appear when metadata exists, absent when missing
- **Add link format test**: Validate VSCode URLs and GitHub permalinks have correct format
- **Add bucket test**: Verify exceptions are separated from classes

### 4. Configuration Defaults

- **Link mode**: Confirm `both` is most useful default (show both editor + GitHub)
- **Editor**: Confirm `vscode` default with fallback messaging for other editors
- **Fail on missing metadata**: Keep `false` by default (non-blocking, warnings only)

## Impact

### Breaking Changes

**None.** All changes are additive or refinement of existing behavior. The generated README format remains backward-compatible (only adds features, doesn't remove or change existing output structure).

### Affected Components

- **`tools/gen_readmes.py`** - Minor refinements (synopsis display, warnings, doctoc integration)
- **`.github/workflows/ci.yml`** - Add README validation step
- **`.pre-commit-config.yaml`** - Add optional README generation hook
- **`docs/how-to/`** - New contributor guide file
- **`README-AUTOMATED-DOCUMENTATION.md`** - Add section on README generation
- **`tests/unit/`** - Add `test_gen_readmes.py` with determinism/badge/link/bucket tests

### Benefits

- **Complete Implementation**: Close gaps between specification and reality
- **Automated Validation**: CI prevents stale READMEs from being committed
- **Better DX**: Contributors understand README structure and update workflow
- **Increased Confidence**: Tests verify behavior matches specification
- **Machine-Readable**: AI agents benefit from consistent, documented README structure

### Migration Path

1. Add documentation files (non-blocking)
2. Add tests for current behavior (validation)
3. Make code refinements (minor changes)
4. Add CI check (initially non-blocking warning)
5. Add pre-commit hook (opt-in)
6. Enable CI check as blocking after baseline is clean

### Rollback Strategy

All changes are additive:
- Documentation can be removed without affecting functionality
- CI check can be disabled in workflow file
- Pre-commit hook is opt-in (skip installation)
- Code refinements are backward-compatible

If needed, revert specific commits without touching the core generation logic.

## What's Already Working

The current implementation is **85-90% complete** relative to the specification. The core logic—metadata lookup, badge formatting, link generation, bucketing—is fully functional. This proposal focuses on:

1. **Closing the last 10-15%**: Documentation, integration, minor refinements
2. **Validation**: Ensuring behavior matches spec with tests
3. **Developer Experience**: Making the system easy to use and understand

This is a **polish and integration** change, not a rewrite.

