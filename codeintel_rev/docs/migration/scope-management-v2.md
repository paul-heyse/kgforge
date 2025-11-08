# Migration Guide: Scope Management v2

This guide helps clients upgrade to the scope-aware version of CodeIntel MCP server, which introduces session-scoped query constraints that persist across multiple requests.

## Overview

**Version**: v2.0.0  
**Release Date**: 2025-11-08  
**Migration Complexity**: Low (additive changes only, backward compatible)

The scope management feature allows clients to set query constraints (path patterns, languages, repositories) once per session, and subsequent queries automatically apply those constraints. This eliminates the need to pass scope parameters with every query.

## What Changed

### New Features

1. **Session-Scoped Scope Storage**: `set_scope` now stores scope in a registry keyed by session ID
2. **Automatic Scope Application**: All search/file operations (`semantic_search`, `search_text`, `list_paths`) automatically apply session scope
3. **Session ID Management**: Support for client-provided or auto-generated session IDs via `X-Session-ID` header
4. **Scope Precedence**: Explicit parameters override session scope (allows one-off queries without clearing scope)

### API Changes (Additive Only)

#### `set_scope` Response

**Before**:
```json
{
  "effective_scope": {
    "languages": ["python"]
  },
  "status": "ok"
}
```

**After**:
```json
{
  "effective_scope": {
    "languages": ["python"]
  },
  "session_id": "a3f8b2c1-4d5e-6789-abcd-ef0123456789",
  "status": "ok"
}
```

**Impact**: Additive change. Existing clients can ignore the new `session_id` field.

#### `AnswerEnvelope` (semantic_search response)

**Before**:
```json
{
  "findings": [...],
  "method": {...}
}
```

**After**:
```json
{
  "findings": [...],
  "scope": {
    "languages": ["python"]
  },
  "method": {...}
}
```

**Impact**: Additive change. The `scope` field shows which scope was applied to the query. Existing clients can ignore it.

#### New Header: `X-Session-ID`

**New Optional Header**: Clients can send `X-Session-ID` header to maintain session across requests.

**Impact**: Optional. If not provided, server auto-generates session ID.

## Breaking Changes

**None**. All changes are additive and backward compatible.

- Existing clients continue to work without modification
- Queries without `set_scope` behave identically (no filters applied)
- Explicit parameters continue to work as before

## Upgrade Steps

### Step 1: Update Client to Send Session ID (Optional but Recommended)

**Why**: Sending `X-Session-ID` header ensures session persistence across client restarts and allows explicit session management.

**Action**: Add `X-Session-ID` header to all requests:

```python
import httpx
import uuid

# Generate session ID once per client session
session_id = str(uuid.uuid4())

# Include in all requests
headers = {"X-Session-ID": session_id}

# First request: set scope
response = httpx.post(
    "http://localhost:8000/mcp/tools/set_scope",
    headers=headers,
    json={"languages": ["python"]}
)

# Subsequent requests: use same session ID
response = httpx.post(
    "http://localhost:8000/mcp/tools/semantic_search",
    headers=headers,
    json={"query": "data processing"}
)
```

**Alternative**: If you don't send the header, the server auto-generates a session ID and returns it in the `set_scope` response. Use that session ID for subsequent requests:

```python
# No header: server generates session ID
response = httpx.post(
    "http://localhost:8000/mcp/tools/set_scope",
    json={"languages": ["python"]}
)
session_id = response.json()["session_id"]

# Use returned session ID in subsequent requests
headers = {"X-Session-ID": session_id}
response = httpx.post(
    "http://localhost:8000/mcp/tools/semantic_search",
    headers=headers,
    json={"query": "data processing"}
)
```

### Step 2: Call `set_scope` to Establish Query Constraints

**Before**: Pass scope parameters with every query:

```python
# Old way: pass filters with every query
results = semantic_search(query="data", languages=["python"])
results = search_text(query="def main", paths=["src/"])
results = list_paths(include_globs=["**/*.py"])
```

**After**: Set scope once, then queries automatically apply it:

```python
# New way: set scope once
set_scope({
    "languages": ["python"],
    "include_globs": ["src/**"]
})

# Subsequent queries automatically apply scope
results = semantic_search(query="data")  # Only Python files in src/
results = search_text(query="def main")  # Only searches src/
results = list_paths()  # Only lists Python files in src/
```

**Example Migration**:

```python
# Before: Repetitive scope parameters
def search_python_code(query: str):
    return semantic_search(
        query=query,
        languages=["python"],
        include_globs=["src/**"]
    )

# After: Set scope once, then search
def setup_python_scope(session_id: str):
    set_scope({
        "languages": ["python"],
        "include_globs": ["src/**"]
    }, session_id=session_id)

def search_python_code(query: str):
    return semantic_search(query=query)  # Scope applied automatically
```

### Step 3: Verify Subsequent Queries Respect Scope

**Testing Checklist**:

1. **Set scope**:
   ```python
   set_scope({"languages": ["python"]})
   ```

2. **Verify semantic_search applies scope**:
   ```python
   results = semantic_search(query="function")
   # Check: all results have .py extension
   assert all(r["path"].endswith(".py") for r in results["findings"])
   ```

3. **Verify search_text applies scope**:
   ```python
   results = search_text(query="def main")
   # Check: only Python files searched
   ```

4. **Verify list_paths applies scope**:
   ```python
   results = list_paths()
   # Check: only Python files listed
   assert all(item["path"].endswith(".py") for item in results["items"])
   ```

5. **Verify explicit parameters override scope**:
   ```python
   # Scope: Python files
   set_scope({"languages": ["python"]})
   
   # Override: search TypeScript files
   results = list_paths(include_globs=["**/*.ts"])
   # Check: only TypeScript files returned (scope ignored)
   ```

## Backward Compatibility

### Existing Clients Continue to Work

**No Migration Required**: Clients that don't use `set_scope` continue to work exactly as before:

```python
# This still works (no scope, searches everything)
results = semantic_search(query="data")
results = search_text(query="def main")
results = list_paths()
```

**Behavior**: Without `set_scope`, queries behave identically to pre-v2 behavior (no filters applied).

### Auto-Generated Session IDs

**Transparent**: If clients don't send `X-Session-ID` header:

- Server auto-generates UUID v4 for each request
- Session ID returned in `set_scope` response
- Clients can use returned session ID for subsequent requests
- No breaking changes for existing clients

### Explicit Parameters Still Work

**Unchanged**: Explicit parameters continue to work as before:

```python
# Explicit parameters override scope (same as before)
results = list_paths(include_globs=["**/*.ts"])  # TypeScript files only
results = search_text(query="def", paths=["src/"])  # Only src/ directory
```

**Precedence Rule**: Explicit parameters always override session scope (unchanged behavior).

## Rollback Plan

If issues arise after upgrading, clients can rollback without server changes:

### Option 1: Ignore `set_scope` (No Server Changes)

**Action**: Simply don't call `set_scope`. All queries work as before (no scope applied).

```python
# Don't call set_scope
# All queries work identically to pre-v2
results = semantic_search(query="data")
results = search_text(query="def main")
```

**Impact**: Zero changes needed. Behavior identical to pre-v2.

### Option 2: Use Explicit Parameters Only

**Action**: Continue using explicit parameters instead of session scope:

```python
# Instead of set_scope + queries
# Use explicit parameters with every query
results = semantic_search(
    query="data",
    languages=["python"],
    include_globs=["src/**"]
)
```

**Impact**: Slightly more verbose, but functionally equivalent.

### Option 3: Clear Scope Before Each Query

**Action**: Call `set_scope` with empty scope before queries:

```python
# Clear scope (no filters)
set_scope({})

# Then query (no scope applied)
results = semantic_search(query="data")
```

**Impact**: Minimal changes. One extra call per query session.

## Common Migration Scenarios

### Scenario 1: Simple Client (No Scope Needed)

**Current Code**:
```python
results = semantic_search(query="data")
```

**Migration**: None required. Code continues to work.

**Optional Enhancement**: Add session ID header for future scope support:
```python
headers = {"X-Session-ID": str(uuid.uuid4())}
results = semantic_search(query="data", headers=headers)
```

### Scenario 2: Client Using Explicit Parameters

**Current Code**:
```python
results = semantic_search(
    query="data",
    languages=["python"],
    include_globs=["src/**"]
)
```

**Migration Option A**: Keep using explicit parameters (no changes):
```python
# Same code, still works
results = semantic_search(
    query="data",
    languages=["python"],
    include_globs=["src/**"]
)
```

**Migration Option B**: Use session scope (reduces repetition):
```python
# Set scope once
set_scope({
    "languages": ["python"],
    "include_globs": ["src/**"]
})

# Then simplify queries
results = semantic_search(query="data")  # Scope applied automatically
```

### Scenario 3: Multi-Query Client (Most Benefit)

**Current Code**:
```python
# Repetitive scope parameters
results1 = semantic_search(query="data", languages=["python"])
results2 = search_text(query="def main", paths=["src/"])
results3 = list_paths(include_globs=["**/*.py"])
```

**Migration**: Set scope once, then simplify queries:
```python
# Set scope once per session
session_id = str(uuid.uuid4())
set_scope({
    "languages": ["python"],
    "include_globs": ["src/**"]
}, session_id=session_id)

# Simplified queries (scope applied automatically)
results1 = semantic_search(query="data")
results2 = search_text(query="def main")
results3 = list_paths()
```

**Benefit**: Less repetitive code, easier to maintain.

## FAQ

### How do I reset scope?

**Answer**: Call `set_scope` with empty fields:

```python
# Reset to no filters
set_scope({})
```

**Alternative**: Clear scope explicitly (if `clear_scope` endpoint exists):
```python
clear_scope(session_id)
```

### How do I disable scope for one query?

**Answer**: Pass explicit parameters that override scope:

```python
# Session scope: Python files only
set_scope({"languages": ["python"]})

# Override: search TypeScript files (scope ignored)
results = list_paths(include_globs=["**/*.ts"])
```

**Precedence Rule**: Explicit parameters always override session scope.

### What happens if my session expires?

**Answer**: Sessions expire after 1 hour of inactivity. When expired:

1. `get_scope(session_id)` returns `None`
2. Queries fall back to no filters (search everything)
3. Re-set scope to restore filters:

```python
# Session expired, re-set scope
set_scope({"languages": ["python"]})
```

**Prevention**: Send requests regularly (within 1 hour) or re-set scope periodically.

### Can I use multiple scopes in parallel?

**Answer**: Yes, use different session IDs:

```python
# Session A: Python files
session_a = str(uuid.uuid4())
set_scope({"languages": ["python"]}, session_id=session_a)

# Session B: TypeScript files
session_b = str(uuid.uuid4())
set_scope({"languages": ["typescript"]}, session_id=session_b)

# Use appropriate session ID for each query
results_a = semantic_search(query="data", headers={"X-Session-ID": session_a})
results_b = semantic_search(query="data", headers={"X-Session-ID": session_b})
```

### How do I check which scope is currently active?

**Answer**: Check the `scope` field in `semantic_search` response:

```python
results = semantic_search(query="data")
current_scope = results.get("scope", {})
print(f"Active scope: {current_scope}")
```

**Note**: `scope` field is empty `{}` if no scope was applied.

### What if I don't send `X-Session-ID` header?

**Answer**: Server auto-generates a session ID:

1. Server generates UUID v4
2. Returns it in `set_scope` response
3. Use returned session ID for subsequent requests:

```python
# No header: server generates session ID
response = set_scope({"languages": ["python"]})
session_id = response["session_id"]

# Use returned session ID
headers = {"X-Session-ID": session_id}
results = semantic_search(query="data", headers=headers)
```

**Recommendation**: Send `X-Session-ID` header for better session management.

### Can I use scope with existing explicit parameters?

**Answer**: Yes, they merge with explicit parameters taking precedence:

```python
# Session scope: Python files in src/
set_scope({
    "languages": ["python"],
    "include_globs": ["src/**"]
})

# Override include_globs, keep languages
results = list_paths(include_globs=["tests/**"])
# Result: Python files in tests/ (languages from scope, paths from explicit)
```

**Merge Rules**: Explicit parameters override corresponding scope fields; unspecified fields use scope defaults.

### What glob patterns are supported?

**Answer**: Unix shell-style globs via `fnmatch`:

- `*` matches any characters (including `/`)
- `?` matches any single character
- `[seq]` matches any character in seq
- `[!seq]` matches any character not in seq
- `**` matches zero or more directories

**Examples**:
- `["**/*.py"]` - All Python files
- `["src/**"]` - All files in src/ directory
- `["src/**/*.py"]` - Python files in src/ directory
- `["**/test_*.py"]` - Test files matching pattern

### What languages are supported?

**Answer**: See `LANGUAGE_EXTENSIONS` mapping. Common languages:

- `python`, `typescript`, `javascript`, `rust`, `go`, `java`
- `kotlin`, `scala`, `cpp`, `c`, `csharp`, `ruby`, `php`
- `swift`, `objectivec`, `bash`, `powershell`
- `yaml`, `json`, `toml`, `xml`, `html`, `css`, `markdown`
- `sql`, `r`, `perl`, `lua`, `haskell`, `elixir`, `erlang`
- `clojure`, `dart`, `vue`, `svelte`

**Language names are case-insensitive**: `"Python"`, `"python"`, `"PYTHON"` all work.

### How do I migrate from global scope to session scope?

**Answer**: If you were using a global scope pattern (storing scope in client state):

**Before** (global scope in client):
```python
# Client stores scope globally
global_scope = {"languages": ["python"]}

# Pass scope with every query
results = semantic_search(query="data", languages=global_scope["languages"])
```

**After** (session scope on server):
```python
# Set scope on server once
set_scope({"languages": ["python"]})

# Queries automatically apply scope
results = semantic_search(query="data")
```

**Benefit**: Scope persists across client restarts (if session ID persists).

## Troubleshooting

### Issue: Scope not being applied

**Symptoms**: Queries return results outside the set scope.

**Causes**:
1. **Session ID mismatch**: Using different session IDs across requests
   - **Fix**: Ensure `X-Session-ID` header is consistent across requests
2. **Session expired**: Session expired (1 hour inactivity)
   - **Fix**: Re-set scope or send requests more frequently
3. **Explicit parameters override**: Explicit params override scope
   - **Fix**: Remove explicit parameters to use scope, or verify override is intentional

**Debugging**:
```python
# Check active scope in response
results = semantic_search(query="data")
print(f"Applied scope: {results.get('scope', {})}")

# Verify session ID consistency
print(f"Session ID: {session_id}")
```

### Issue: Session ID not persisting

**Symptoms**: Each request gets a new session ID, scope doesn't persist.

**Causes**:
1. **Not sending header**: Client not sending `X-Session-ID` header
   - **Fix**: Include `X-Session-ID` header in all requests
2. **Header value changes**: Header value changes between requests
   - **Fix**: Store session ID in client state, reuse across requests

**Debugging**:
```python
# Verify header is sent
headers = {"X-Session-ID": session_id}
response = httpx.post(url, headers=headers)
print(f"Response session_id: {response.json().get('session_id')}")
```

### Issue: Explicit parameters not overriding scope

**Symptoms**: Explicit parameters are ignored, scope always applied.

**Causes**: This should not happen—explicit parameters always override scope.

**Debugging**: Verify parameter names match API:
- `list_paths`: `include_globs`, `exclude_globs` (not `paths`)
- `search_text`: `paths` parameter (not `include_globs`)

## Support

For questions or issues during migration:

1. **Check FAQ**: Common questions answered above
2. **Review Architecture**: See `docs/architecture/scope-management.md`
3. **Check OpenAPI Spec**: See `openapi/codeintel-scope.yaml`
4. **File Issue**: Report bugs or request clarifications

## Summary

- **Breaking Changes**: None (additive only)
- **Migration Complexity**: Low (optional enhancement)
- **Backward Compatibility**: 100% (existing clients work without changes)
- **Recommended**: Adopt session scope for multi-query clients to reduce repetition
- **Rollback**: Simple (ignore `set_scope`, use explicit parameters)

The scope management feature is designed to be **opt-in** and **non-breaking**. Existing clients continue to work without modification, while new clients can adopt session scope for improved usability.

