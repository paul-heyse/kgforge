# OpenSpec Change Proposal: CodeIntel Configuration Lifecycle Management

## Quick Links

- **Proposal**: [proposal.md](./proposal.md) - Why this change and what it does
- **Design**: [design.md](./design.md) - Detailed technical design and architecture
- **Tasks**: [tasks.md](./tasks.md) - Step-by-step implementation checklist  
- **Spec**: [specs/codeintel-configuration/spec.md](./specs/codeintel-configuration/spec.md) - Requirements and acceptance scenarios
- **Implementation**: [implementation/](./implementation/) - All code and implementation guide

## For First-Time Readers

Start here if you're new to this change:

1. **Read [implementation/README.md](./implementation/README.md)** - Junior developer friendly explanation with diagrams and examples
2. **Review [proposal.md](./proposal.md)** - High-level summary of the problem and solution
3. **Check [design.md](./design.md)** - Deep dive into architecture decisions and trade-offs

## For Implementers

Follow this sequence:

1. **Setup**: Run `scripts/bootstrap.sh` to prepare environment
2. **Plan**: Read [tasks.md](./tasks.md) - 7 phases with 80+ granular steps
3. **Code**: Use [implementation/config_context.py](./implementation/config_context.py) as reference
4. **Test**: Each phase has unit tests - run them before moving to next phase
5. **Validate**: Run `openspec validate codeintel-config-lifecycle-management --strict`

## Key Changes Summary

### What's Being Added

- **ApplicationContext** - Single source of configuration truth
- **ResolvedPaths** - Canonicalized filesystem paths (no more path resolution duplication)
- **ReadinessProbe** - Comprehensive health checks for all resources
- **Configuration lifecycle** - Load once at startup, fail-fast on invalid config
- **FAISS pre-loading** - Optional eager loading via `FAISS_PRELOAD=1` environment variable

### What's Being Changed

- **All 4 adapters** - Now accept `ApplicationContext` parameter (explicit dependency injection)
- **FastAPI lifespan** - Configuration initialization moved to startup
- **MCP server** - Tool wrappers extract context from FastAPI state
- **Settings** - Added `faiss_preload` field

### What's Being Removed

- **ServiceContext singleton** - Replaced by `ApplicationContext` in FastAPI state
- **`load_settings()` calls in adapters** - Called once at startup instead
- **`@lru_cache` pattern** - Replaced by FastAPI application state management

## Design Principles Enforced

✅ **Zero Redundant Configuration Loading** - Settings parsed once at startup  
✅ **Explicit Dependency Injection** - Context passed as parameter everywhere  
✅ **Fail-Fast Behavior** - Invalid config prevents app from starting  
✅ **No Global State** - No singletons, no module-level caches  
✅ **RFC 9457 Errors** - All errors use Problem Details format  
✅ **100% AGENTS.MD Compliance** - Zero suppressions, zero shortcuts  

## Success Criteria

This change is complete when:

- ✅ Zero `load_settings()` calls in adapter functions
- ✅ All adapters accept `ApplicationContext` as first parameter
- ✅ FastAPI lifespan loads configuration once at startup
- ✅ Missing FAISS index (when pre-loading enabled) prevents startup
- ✅ All tests pass (unit + integration)
- ✅ Zero Ruff/pyright/pyrefly errors
- ✅ 95%+ test coverage on new code
- ✅ Documentation complete and junior-developer friendly
- ✅ `openspec validate codeintel-config-lifecycle-management --strict` passes

## Implementation Timeline

| Phase | Duration | Risk | Can Rollback? |
|-------|----------|------|---------------|
| Phase 1: Foundation | 2 days | Low | Yes (no existing code touched) |
| Phase 2: FastAPI Integration | 2 days | Low | Yes (context not used yet) |
| Phase 3: Adapter Refactoring | 3 days | Medium | Yes (per-adapter rollback) |
| Phase 4: MCP Server Integration | 2 days | Medium | Yes (final cutover) |
| Phase 5: Cleanup | 1 day | Low | N/A |
| Phase 6: OpenSpec Validation | 1 day | Low | N/A |
| **Total** | **11 days** | **Low-Medium** | **Yes (incremental)** |

## Getting Help

### For Concept Questions

- Read [implementation/README.md](./implementation/README.md) - Has junior developer explanations
- Read [design.md](./design.md) - Section "Decisions" explains why each choice was made
- Check "Common Questions" section in implementation README

### For Implementation Questions

- Check [tasks.md](./tasks.md) - Step-by-step instructions for each task
- Read code comments in [implementation/config_context.py](./implementation/config_context.py)
- Run tests to verify understanding: `pytest tests/codeintel_rev/test_config_context.py -v`

### For Testing Questions

- Each phase in [tasks.md](./tasks.md) has verification steps
- See "Testing Your Changes" section in implementation README
- Use `pytest --cov` to check coverage

## Related Documents

- **AGENTS.MD** - Overall design principles and quality standards
- **openspec/AGENTS.md** - How to create and validate openspec proposals
- **codeintel_rev/CodeIntel MCP: Design Review and Improve.md** - Original design review that identified these issues

## Validation Commands

```bash
# Validate this proposal
openspec validate codeintel-config-lifecycle-management --strict

# Run all quality gates (when implementing)
uv run ruff format && uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
uv run pytest tests/codeintel_rev/ -v --cov=codeintel_rev
make artifacts && git diff --exit-code

# Verify no new suppressions
python tools/check_new_suppressions.py codeintel_rev/
```

## Status

**Stage**: Proposed (awaiting review and approval)  
**Created**: 2025-11-08  
**Change ID**: `codeintel-config-lifecycle-management`  
**Branch**: `openspec/codeintel-config-lifecycle-management` (when implementing)  

## Next Steps

1. **Review**: Reviewers assess proposal, design, and tasks
2. **Approve**: Once approved, implementer begins Phase 1
3. **Implement**: Follow tasks.md sequentially, checking off items
4. **Validate**: Run all quality gates at end of each phase
5. **Archive**: Move to `openspec/changes/archive/` when complete

---

**Remember**: This is a structural refactoring with zero external API changes. Clients won't notice any difference - all changes are internal improvements to code quality, testability, and maintainability.

