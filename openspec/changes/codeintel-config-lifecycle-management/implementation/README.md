# CodeIntel Configuration Lifecycle Management - Implementation Guide

This directory contains all implementation code for the configuration lifecycle refactoring described in the parent change proposal.

## For Junior Developers: Understanding the Change

### What Problem Are We Solving?

Imagine you have a house with 4 rooms. In each room, there's a person who needs to know what time it is. Currently, each person walks outside to check the clock tower **every single time** they need the time. This is:
1. **Inefficient** - Why walk outside 4+ times when you could check once?
2. **Inconsistent** - What if the clock tower changes while people are checking?
3. **Slow** - Every walk to the clock tower takes time

**Our solution**: Put one clock in the house's entrance hall. Check the clock tower **once** when you wake up, set the house clock, and everyone uses that clock all day.

In code terms:
- **Clock tower** = Environment variables (slow to read)
- **House clock** = ApplicationContext (fast to access)
- **People in rooms** = Adapter functions (files, search, history, semantic)
- **Checking time** = Reading configuration

### Key Concepts Explained

#### 1. ApplicationContext (The House Clock)

```python
@dataclass(slots=True)
class ApplicationContext:
    """ONE place holding ALL configuration for the entire application."""
    settings: Settings              # ← Configuration (from environment)
    paths: ResolvedPaths           # ← File paths (already resolved)
    vllm_client: VLLMClient        # ← Long-lived HTTP client
    faiss_manager: FAISSManager    # ← Index manager
```

**Why this matters**: Instead of parsing 20+ environment variables on every request, we parse them **once** at startup and store the result here.

#### 2. Dependency Injection (Passing the Clock)

**Before (BAD)**:
```python
def list_paths(...) -> dict:
    settings = load_settings()  # ❌ Walks to clock tower
    repo_root = Path(settings.paths.repo_root).resolve()
    # ...
```

**After (GOOD)**:
```python
def list_paths(context: ApplicationContext, ...) -> dict:
    repo_root = context.paths.repo_root  # ✅ Looks at house clock
    # ...
```

**Why this matters**: The clock (context) is **passed in** as a parameter, so you can see exactly what the function needs. This makes testing easy (just create a fake clock) and makes the code honest about its dependencies.

#### 3. FastAPI Lifespan (Setting Up the House)

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs when app starts (sets up house clock)."""
    # 1. Check clock tower (read environment variables)
    context = ApplicationContext.create()
    
    # 2. Put clock in entrance hall (store in app.state)
    app.state.context = context
    
    # 3. Open the house for business
    yield
    
    # 4. Close the house (cleanup)
    # ... cleanup code ...
```

**Why this matters**: This runs **once** when the application starts, not on every request. If setup fails (e.g., clock tower is broken), the application refuses to start - this is called "fail-fast."

#### 4. Request Handler (People Getting the Time)

```python
@mcp.tool()
def list_paths(request: Request, path: str | None = None, ...) -> dict:
    # Step 1: Get the house clock
    context = _get_context(request)
    
    # Step 2: Pass clock to the person doing work
    return files_adapter.list_paths(context, path, ...)
```

**Why this matters**: The request handler is like a receptionist. It gets the house clock and passes it to whoever needs it. The adapter (person doing work) doesn't need to know where the clock came from - it just uses it.

## Implementation Files

### Core Infrastructure

1. **`config_context.py`** - The house clock and clock setup logic
   - `ResolvedPaths` - All file paths, already converted to absolute paths
   - `ApplicationContext` - The main clock holding all configuration
   - `resolve_application_paths()` - Validates paths at startup
   - `ApplicationContext.create()` - Sets up the house clock

2. **`readiness.py`** - Health check system
   - `CheckResult` - Result of checking one thing (healthy/unhealthy)
   - `ReadinessProbe` - Checks all critical resources (files, services)
   - Integrates with Kubernetes `/readyz` endpoint

### Modified Files

3. **`app/main.py`** - Application startup
   - `lifespan()` - Sets up house clock, runs health checks
   - `_preload_faiss_index()` - Optional: load AI index at startup
   - Updated `/readyz` endpoint to use ReadinessProbe

4. **Adapter files** (files.py, text_search.py, history.py, semantic.py)
   - All functions now accept `context: ApplicationContext` parameter
   - No more `load_settings()` calls
   - Use `context.paths.*` for file paths
   - Use `context.settings.*` for configuration

5. **`mcp_server/server.py`** - MCP tool wrappers
   - New `_get_context(request)` helper - gets house clock
   - All `@mcp.tool()` functions extract context and pass to adapters
   - External API unchanged (clients don't see internal changes)

### Test Files

6. **Test files** - Verify everything works
   - `test_config_context.py` - Tests for ApplicationContext
   - `test_readiness.py` - Tests for health checks
   - `test_app_lifespan.py` - Tests for application startup
   - `test_*_adapter.py` - Tests for each adapter with mocked context

## Step-by-Step Implementation Guide

### Phase 1: Build the Foundation (No Impact on Existing Code)

**Goal**: Create new files without touching anything that exists.

**Tasks**:
1. Create `config_context.py` with all classes and functions
2. Create `readiness.py` with health check system
3. Write unit tests for both files
4. Run: `pytest tests/codeintel_rev/test_config_context.py -v`

**How to verify**: Tests pass, zero errors. Existing code still works because we haven't changed it yet.

### Phase 2: Integrate with FastAPI (Minimal Changes)

**Goal**: Wire up the house clock in application startup.

**Tasks**:
1. Modify `app/main.py` to create `ApplicationContext` in `lifespan()`
2. Store context in `app.state.context`
3. Update `/readyz` endpoint
4. Write integration tests

**How to verify**: Application starts successfully, `/readyz` works, context is available in `app.state`.

### Phase 3: Refactor Adapters One at a Time (Incremental)

**Goal**: Update each adapter to use the house clock.

**Order** (easiest to hardest):
1. `files.py` - Simplest, just needs repo_root
2. `history.py` - Only uses repo_root and path utils
3. `text_search.py` - Uses repo_root and settings
4. `semantic.py` - Most complex, uses vLLM and FAISS

**For each adapter**:
1. Add `context: ApplicationContext` parameter (first position)
2. Replace `load_settings()` with `context.settings`
3. Replace path resolution with `context.paths.*`
4. Write unit tests with mocked context
5. Run tests, verify zero errors

**How to verify**: Each adapter's tests pass independently. Can rollback one adapter without affecting others.

### Phase 4: Connect MCP Server (Final Integration)

**Goal**: Wire up request handlers to extract and pass context.

**Tasks**:
1. Add `_get_context(request)` helper to `server.py`
2. Update all `@mcp.tool()` functions to extract context
3. Pass context to adapter functions
4. Write integration tests

**How to verify**: Full request flow works end-to-end. Can make actual MCP tool calls.

### Phase 5: Cleanup and Documentation

**Goal**: Remove old code, polish everything.

**Tasks**:
1. Delete `service_context.py` (no longer needed)
2. Remove unused imports
3. Write `CONFIGURATION.md` documentation
4. Run all quality gates (Ruff, pyright, pyrefly)

**How to verify**: Zero errors, zero warnings, all tests pass, documentation is clear.

## Common Questions

### Q: Why can't we just keep using global singletons?

**A**: Global singletons have three problems:
1. **Testing is hard** - You can't mock them easily, can't run tests in parallel
2. **Hidden dependencies** - Functions that use globals don't declare what they need
3. **Initialization timing** - You can't control when they're initialized or validate config at startup

### Q: Won't passing context everywhere make code verbose?

**A**: It adds one parameter per function, but:
1. **Explicit is better than implicit** - You can see what each function needs
2. **Testing becomes trivial** - Just create a mock context object
3. **Debugging is easier** - You can trace where configuration comes from
4. **No magic** - Junior developers can understand the flow

### Q: What if I forget to pass context?

**A**: Python's type checker (pyright) will catch it immediately:
```python
# Forgot to pass context
result = list_paths(path="src")  # ❌ ERROR: Missing required parameter 'context'

# Correct
result = list_paths(context, path="src")  # ✅ OK
```

### Q: How do I test an adapter function?

**A**: Create a mock context with test values:
```python
def test_list_paths():
    # Create a mock context with test paths
    mock_context = Mock(spec=ApplicationContext)
    mock_context.paths = ResolvedPaths(
        repo_root=Path("/test/repo"),
        # ... other paths
    )
    
    # Call the adapter with mock context
    result = list_paths(mock_context, path="src")
    
    # Verify behavior
    assert "items" in result
```

No environment variables needed, no global state to worry about!

### Q: What happens if configuration is invalid?

**A**: The application **refuses to start**. This is called "fail-fast" and it's a good thing:
```python
# Invalid repo root
REPO_ROOT=/nonexistent/path uvicorn app.main:app

# Application immediately fails with clear error:
# ConfigurationError: Repository root does not exist: /nonexistent/path
# Context: {"repo_root": "/nonexistent/path", "source": "REPO_ROOT env var"}
```

Better to fail immediately than to start serving requests and fail randomly later!

### Q: How do I know what environment variables are needed?

**A**: Check `CONFIGURATION.md` (created in Phase 5). Key variables:
- `REPO_ROOT` - Required, must exist
- `FAISS_INDEX` - Required (if pre-loading), must exist
- `FAISS_PRELOAD` - Optional, `0` (lazy) or `1` (eager)
- `VLLM_URL` - Optional, defaults to `http://127.0.0.1:8001/v1`

## Testing Your Changes

### Unit Tests (Fast, Isolated)

```bash
# Test configuration context
uv run pytest tests/codeintel_rev/test_config_context.py -v

# Test readiness probe
uv run pytest tests/codeintel_rev/test_readiness.py -v

# Test an adapter
uv run pytest tests/codeintel_rev/adapters/test_files_adapter.py -v
```

### Integration Tests (Full Flow)

```bash
# Test application startup
uv run pytest tests/codeintel_rev/test_app_lifespan.py -v

# Test MCP server integration
uv run pytest tests/codeintel_rev/test_mcp_server.py -v
```

### Manual Testing (Real Application)

```bash
# Start the application
uvicorn codeintel_rev.app.main:app --port 8000

# In another terminal:
# 1. Check health
curl http://localhost:8000/healthz

# 2. Check readiness (should show all resources healthy)
curl http://localhost:8000/readyz | jq

# 3. Test a tool (if you have an MCP client)
# ... MCP tool call ...
```

## Troubleshooting

### "ApplicationContext not initialized in app state"

**Cause**: Application startup failed but didn't fully crash.

**Fix**: Check logs for ConfigurationError. Verify all environment variables are set correctly.

### "FAISS index not found"

**Cause**: `FAISS_INDEX` points to non-existent file.

**Fix**: Run indexing pipeline first, or disable pre-loading with `FAISS_PRELOAD=0`.

### "Repository root does not exist"

**Cause**: `REPO_ROOT` environment variable is wrong.

**Fix**: Set to actual repository path: `export REPO_ROOT=/path/to/kgfoundry`.

### Tests fail with "No such file or directory"

**Cause**: Test fixtures not creating required directory structure.

**Fix**: Check `test_repo` fixture creates all directories (`data/`, `data/vectors/`, `data/faiss/`).

## Next Steps After Implementation

1. **Performance Verification**:
   - Measure startup time with/without FAISS pre-loading
   - Verify request latency hasn't regressed
   - Document timing numbers

2. **Documentation Review**:
   - Have a junior developer read `CONFIGURATION.md`
   - Ask: "Can you understand how to configure this?"
   - Update based on feedback

3. **Team Training**:
   - Present the new architecture to team
   - Explain dependency injection pattern
   - Show how to test with mocked context

4. **Production Deployment**:
   - Start with `FAISS_PRELOAD=0` (safe default)
   - Monitor startup time and first-request latency
   - If latency is issue, enable `FAISS_PRELOAD=1`
   - Update Kubernetes readiness probe `initialDelaySeconds` accordingly

## Architecture Diagrams

See `design.md` for detailed architecture diagrams showing:
- Configuration loading sequence
- Request handling flow
- Component relationships

## Further Reading

- **AGENTS.MD** - Design principles (dependency injection, fail-fast, structured errors)
- **RFC 9457** - Problem Details standard for error handling
- **12-Factor App** - Configuration best practices
- **FastAPI Lifespan** - Application startup/shutdown management

---

**Remember**: The goal is not just to make the code work, but to make it **understandable**. If you're confused about any part of this, that's valuable feedback - please ask questions so we can improve the documentation!

