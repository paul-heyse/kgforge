# Test Migration Guide

**Updating Tests to Use New CodeIntel Patterns**

This guide shows how to update existing tests to use the new exception types, dependency injection, and observability features.

---

## ✅ **Migration Checklist**

- [ ] Replace generic exceptions with specific CodeIntel exceptions
- [ ] Use `ServerContext` for configuration injection
- [ ] Add observability assertions for metrics
- [ ] Use Pydantic validators for input validation tests
- [ ] Update error message assertions to match new exceptions

---

## 🔄 **Pattern Migrations**

### 1. Exception Handling

#### Before (Generic Exceptions)
```python
def test_unsupported_language():
    langs = load_langs()
    
    with pytest.raises(ValueError) as exc_info:
        get_language(langs, "nonexistent")
    
    assert "unsupported" in str(exc_info.value).lower()
```

#### After (Specific Exceptions)
```python
from codeintel.errors import LanguageNotSupportedError

def test_unsupported_language():
    langs = load_langs()
    
    with pytest.raises(LanguageNotSupportedError) as exc_info:
        get_language(langs, "nonexistent")
    
    # Verify structured context
    assert exc_info.value.extensions["requested"] == "nonexistent"
    assert "python" in exc_info.value.extensions["available"]
    
    # Verify Problem Details
    problem = exc_info.value.to_problem_details()
    assert problem["status"] == 400
    assert problem["type"] == "urn:kgf:problem:codeintel:language-not-supported"
```

---

### 2. Configuration & Dependency Injection

#### Before (Global State)
```python
from codeintel.config import LIMITS

def test_file_size_limits():
    # Tests depend on global LIMITS
    assert LIMITS.max_ast_bytes > 0
```

#### After (Dependency Injection)
```python
from codeintel.config import ServerContext, ServerLimits

@pytest.fixture
def test_context(tmp_path):
    """Provide isolated test configuration."""
    limits = ServerLimits.permissive()
    return ServerContext(limits=limits, repo_root=tmp_path)

def test_file_size_limits(test_context):
    # Use injected configuration
    assert test_context.limits.max_ast_bytes > 0
    assert test_context.repo_root.exists()
```

---

### 3. Observability Assertions

#### Before (No Metrics)
```python
def test_parse_operation():
    lang = get_language(langs, "python")
    tree = parse_bytes(lang, b"def hello(): pass")
    
    assert tree is not None
```

#### After (With Metrics)
```python
from codeintel.observability import PARSE_DURATION_SECONDS

def test_parse_operation():
    # Get baseline metric value
    before = PARSE_DURATION_SECONDS.labels(lang="python")._value._sum
    
    lang = get_language(langs, "python")
    tree = parse_bytes(lang, b"def hello(): pass")
    
    assert tree is not None
    
    # Verify metrics were recorded
    after = PARSE_DURATION_SECONDS.labels(lang="python")._value._sum
    assert after > before
```

---

### 4. Input Validation Tests

#### Before (Manual Validation)
```python
def test_empty_path_rejected():
    with pytest.raises(ValueError):
        get_file("")
```

#### After (Pydantic Validators)
```python
from pydantic import ValidationError
from codeintel.mcp_server.server import GetFileRequest

def test_empty_path_rejected():
    with pytest.raises(ValidationError) as exc_info:
        GetFileRequest(path="")
    
    # Verify specific validation error
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("path",) for e in errors)
    assert any("non-empty" in e["msg"] for e in errors)

def test_null_bytes_rejected():
    with pytest.raises(ValidationError) as exc_info:
        GetFileRequest(path="test\x00.py")
    
    errors = exc_info.value.errors()
    assert any("null bytes" in e["msg"] for e in errors)
```

---

### 5. Test Fixtures

#### Before (Ad-hoc Setup)
```python
def test_something(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # ... manual setup
```

#### After (Reusable Fixtures)
```python
@pytest.fixture
def codeintel_repo(tmp_path):
    """Standard CodeIntel test repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    
    # Create sample files
    (repo / "test.py").write_text("def hello(): pass")
    (repo / "test.json").write_text('{"key": "value"}')
    
    return repo

@pytest.fixture
def codeintel_context(codeintel_repo):
    """Pre-configured ServerContext for tests."""
    return ServerContext.for_testing(repo_root=codeintel_repo)

def test_something(codeintel_context):
    # Use pre-configured context
    assert codeintel_context.repo_root.exists()
```

---

## 📝 **Example: Complete Migration**

### Original Test

```python
def test_index_and_search(tmp_path):
    from codeintel.index.store import IndexStore, ensure_schema, index_incremental, search_symbols
    
    # Setup
    (tmp_path / "test.py").write_text("def calculator(): pass")
    db = tmp_path / "index.db"
    
    # Index
    with IndexStore(db) as store:
        ensure_schema(store)
        count = index_incremental(store, tmp_path, changed_only=False)
    
    # Search
    with IndexStore(db) as store:
        results = search_symbols(store, query="calculator")
        assert len(results) > 0
```

### Migrated Test

```python
from codeintel.config import ServerContext
from codeintel.index.store import IndexStore, ensure_schema, index_incremental, search_symbols
from codeintel.observability import INDEX_SIZE_SYMBOLS, log_operation
from codeintel.errors import IndexNotFoundError

@pytest.fixture
def indexed_repo(tmp_path):
    """Repository with pre-built index."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test.py").write_text("def calculator(): pass")
    
    db = repo / ".kgf" / "index.db"
    db.parent.mkdir(parents=True)
    
    with IndexStore(db) as store:
        ensure_schema(store)
        with log_operation("test_index_build", repo=str(repo)):
            index_incremental(store, repo, changed_only=False)
    
    return repo, db

def test_index_and_search(indexed_repo):
    """Test indexing and search with new patterns."""
    repo, db = indexed_repo
    
    # Verify index exists
    assert db.exists()
    
    # Search with context
    context = ServerContext.for_testing(repo_root=repo)
    
    with IndexStore(db) as store:
        results = search_symbols(store, query="calculator", lang="python")
        
        # Structured assertions
        assert len(results) > 0
        assert results[0]["name"] == "calculator"
        assert results[0]["lang"] == "python"
        assert "test.py" in results[0]["path"]

def test_search_missing_index(tmp_path):
    """Test that missing index raises appropriate error."""
    db = tmp_path / "nonexistent.db"
    
    # New exception type
    with pytest.raises(IndexNotFoundError) as exc_info:
        with IndexStore(db) as store:
            search_symbols(store, query="anything")
    
    # Verify Problem Details
    problem = exc_info.value.to_problem_details()
    assert problem["status"] == 404
    assert "index_path" in problem
```

---

## 🎯 **Quick Reference**

| Old Pattern | New Pattern |
|-------------|-------------|
| `ValueError` | `LanguageNotSupportedError`, `QuerySyntaxError` |
| `RuntimeError` | `ManifestError`, `IndexCorruptedError` |
| `FileNotFoundError` | `IndexNotFoundError` |
| Global `LIMITS` | `ServerContext.limits` |
| No metrics | `PARSE_DURATION_SECONDS`, `TOOL_CALLS_TOTAL` |
| Manual validation | Pydantic `field_validator` |
| String error messages | `exception.extensions` dict |

---

## ✨ **Benefits of Migration**

1. **Type Safety**: Catch errors at development time
2. **Testability**: Inject custom configurations easily
3. **Observability**: Verify metrics in tests
4. **Maintainability**: Structured error context
5. **Documentation**: Self-documenting error types

---

**Status**: All patterns documented  
**Last Updated**: November 7, 2024

