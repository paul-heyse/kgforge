# CodeIntel API Reference & Usage Guide

**Production-Ready Code Intelligence for KGFoundry**

CodeIntel provides Tree-sitter powered code analysis, symbol indexing, and MCP (Model Context Protocol) integration for AI-assisted development workflows.

---

## 📚 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Core APIs](#core-apis)
3. [Exception Handling](#exception-handling)
4. [Configuration](#configuration)
5. [Observability](#observability)
6. [Testing](#testing)
7. [MCP Integration](#mcp-integration)

---

## 🚀 **Quick Start**

### Installation

```bash
cd /path/to/kgfoundry
scripts/bootstrap.sh
python -m codeintel.build_languages
```

### Basic Usage

```python
from codeintel.indexer.tscore import load_langs, get_language, parse_bytes

# Load Tree-sitter grammars
langs = load_langs()

# Parse Python code
lang = get_language(langs, "python")
code = b"def hello():\n    return 'world'\n"
tree = parse_bytes(lang, code)

print(tree.root_node.sexp())  # S-expression AST
```

---

## 🔧 **Core APIs**

### Language Loading

```python
from codeintel.indexer.tscore import load_langs, get_language

# Load all languages (cached)
langs = load_langs()

# Get specific language
python_lang = get_language(langs, "python")
json_lang = get_language(langs, "json")

# Available languages
from codeintel.indexer.tscore import LANGUAGE_NAMES
print(LANGUAGE_NAMES)  # {'python', 'json', 'yaml', 'toml', 'markdown'}
```

### Parsing & Queries

```python
from codeintel.indexer.tscore import parse_bytes, run_query

# Parse source code
lang = get_language(langs, "python")
code = b"class Calculator:\n    def add(self, x, y):\n        return x + y\n"
tree = parse_bytes(lang, code)

# Run Tree-sitter query
query = """
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params)
"""

captures = run_query(lang, query, tree, code)
for cap in captures:
    print(f"{cap['capture']}: {cap['text']}")
```

### Query Registry

```python
from codeintel.queries import load_query, validate_query, list_available_queries

# List available queries
languages = list_available_queries()
print(languages)  # ['python', 'json', 'yaml', 'toml', 'markdown']

# Load built-in query
python_query = load_query("python")

# Validate custom query
validate_query("python", "(function_definition) @func")  # Raises QuerySyntaxError if invalid
```

### Symbol Indexing

```python
from pathlib import Path
from codeintel.index.store import IndexStore, ensure_schema, index_incremental

# Create/open index
db_path = Path(".kgf/codeintel.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

with IndexStore(db_path) as store:
    ensure_schema(store)
    
    # Index repository
    repo_root = Path(".")
    file_count = index_incremental(store, repo_root, changed_only=False)
    print(f"Indexed {file_count} files")
```

### Symbol Search

```python
from codeintel.index.store import search_symbols, find_references

with IndexStore(db_path) as store:
    # Search symbols
    results = search_symbols(store, query="Calculator", lang="python", limit=10)
    for sym in results:
        print(f"{sym['name']} at {sym['path']}:{sym['start']}")
    
    # Find references
    refs = find_references(store, qualname="Calculator.add", limit=10)
    for ref in refs:
        print(f"Reference at {ref['path']}:{ref['line']}")
```

---

## ⚠️ **Exception Handling**

CodeIntel uses RFC 9457 Problem Details for structured error reporting.

### Exception Hierarchy

```python
from codeintel.errors import (
    CodeIntelError,              # Base exception
    SandboxError,                # Path outside repository (403)
    LanguageNotSupportedError,   # Unknown language (400)
    QuerySyntaxError,            # Invalid Tree-sitter query (422)
    IndexNotFoundError,          # Missing index database (404)
    FileTooLargeError,           # File exceeds limits (413)
    ManifestError,               # Language manifest error (500)
    OperationTimeoutError,       # Operation exceeded timeout (504)
    RateLimitExceededError,      # Too many requests (429)
    IndexCorruptedError,         # Database corrupted (500)
)
```

### Usage Example

```python
try:
    lang = get_language(langs, "rust")  # Not installed
except LanguageNotSupportedError as e:
    # Structured error with context
    print(e.extensions["requested"])   # "rust"
    print(e.extensions["available"])   # ["python", "json", ...]
    
    # Problem Details for HTTP
    problem = e.to_problem_details()
    print(problem["type"])    # "urn:kgf:problem:codeintel:language-not-supported"
    print(problem["status"])  # 400
```

### Best Practices

```python
from codeintel.errors import QuerySyntaxError, LanguageNotSupportedError

def safe_query(language: str, query: str):
    """Execute query with proper error handling."""
    try:
        langs = load_langs()
        lang = get_language(langs, language)
        # ... parse and query
        
    except LanguageNotSupportedError as e:
        # Log with context
        logger.error("Unsupported language", extra=e.extensions)
        raise  # Re-raise for caller
        
    except QuerySyntaxError as e:
        # Syntax errors are client mistakes
        logger.warning("Invalid query syntax", extra=e.extensions)
        return {"error": str(e)}
```

---

## ⚙️ **Configuration**

### Environment Variables

```bash
# Repository sandbox
export KGF_REPO_ROOT=/workspace/myrepo

# Resource limits
export CODEINTEL_MAX_AST_BYTES=2097152       # 2 MiB
export CODEINTEL_MAX_OUTLINE_ITEMS=5000
export CODEINTEL_TOOL_TIMEOUT_S=30.0

# Rate limiting
export CODEINTEL_RATE_LIMIT_QPS=10.0
export CODEINTEL_RATE_LIMIT_BURST=20

# Features
export CODEINTEL_ENABLE_TS_QUERY=1  # Enable arbitrary queries
```

### Programmatic Configuration

```python
from codeintel.config import ServerLimits, ServerContext
from pathlib import Path

# Production config from environment
context = ServerContext.from_env()

# Custom config for testing
limits = ServerLimits(
    max_ast_bytes=10_485_760,  # 10 MiB
    tool_timeout_s=60.0,
    rate_limit_qps=100.0,
    rate_limit_burst=200,
    enable_ts_query=True,
)
context = ServerContext(limits=limits, repo_root=Path("/workspace"))

# Use in tests
@pytest.fixture
def test_context(tmp_path):
    return ServerContext.for_testing(repo_root=tmp_path)
```

---

## 📊 **Observability**

### Prometheus Metrics

```python
from codeintel.observability import (
    instrument_tool,
    record_parse,
    update_index_metrics,
    log_operation,
)

# Instrument async functions
@instrument_tool("code.getOutline")
async def get_outline(path: str):
    # Metrics recorded automatically: duration, status, errors
    pass

# Record parse operations
duration = time.monotonic() - start
record_parse(language="python", size_bytes=len(data), duration_s=duration)

# Update index metrics
symbol_counts = {"python": 1234, "json": 56}
update_index_metrics(symbol_counts, ref_count=500, file_count=100)

# Structured logging with context
with log_operation("index_build", lang="python", files=100):
    # Operation logged with duration
    index_incremental(store, repo_root)
```

### Available Metrics

- `codeintel_tool_calls_total` - Counter by tool and status
- `codeintel_tool_duration_seconds` - Histogram by tool
- `codeintel_tool_errors_total` - Counter by tool and error type
- `codeintel_index_symbols_total` - Gauge by language
- `codeintel_index_refs_total` - Gauge
- `codeintel_index_files_total` - Gauge
- `codeintel_index_build_duration_seconds` - Histogram
- `codeintel_parse_duration_seconds` - Histogram by language

---

## 🧪 **Testing**

### Unit Tests

```python
from codeintel.config import ServerContext
from codeintel.errors import LanguageNotSupportedError

def test_language_validation():
    """Test that unsupported languages raise appropriate errors."""
    langs = load_langs()
    
    with pytest.raises(LanguageNotSupportedError) as exc_info:
        get_language(langs, "nonexistent")
    
    # Verify error context
    assert "nonexistent" in exc_info.value.extensions["requested"]
    assert "python" in exc_info.value.extensions["available"]
```

### Integration Tests

```python
def test_end_to_end_workflow(tmp_path):
    """Test complete indexing and search workflow."""
    # Setup
    (tmp_path / "test.py").write_text("def hello(): pass")
    db = tmp_path / "index.db"
    
    # Index
    with IndexStore(db) as store:
        ensure_schema(store)
        index_incremental(store, tmp_path)
    
    # Search
    with IndexStore(db) as store:
        results = search_symbols(store, query="hello", lang="python")
        assert len(results) == 1
        assert results[0]["name"] == "hello"
```

### Benchmark Tests

```bash
# Run with pytest-benchmark
pytest tests/codeintel/test_benchmarks.py --benchmark-only

# Save baseline
pytest tests/codeintel/test_benchmarks.py --benchmark-save=baseline

# Compare to baseline
pytest tests/codeintel/test_benchmarks.py --benchmark-compare=baseline
```

---

## 🤖 **MCP Integration**

### Starting the Server

```bash
# Direct
python -m codeintel.mcp_server.server

# Via CLI
python -m codeintel.cli mcp serve --repo .
```

### Available Tools

| Tool | Description |
|------|-------------|
| `code.health` | Comprehensive health diagnostics |
| `code.listFiles` | List repository files with filters |
| `code.getFile` | Read file contents (with chunking) |
| `code.getOutline` | Extract code outline (functions, classes) |
| `code.getAST` | Get syntax tree snapshot |
| `code.searchSymbols` | Search indexed symbols |
| `code.findReferences` | Find symbol references |
| `ts.query` | Execute arbitrary Tree-sitter query (if enabled) |

### Example: Health Check

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "code.health",
    "arguments": {}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "status": "healthy",
    "timestamp": "2024-11-07T12:34:56Z",
    "components": {
      "manifest": {"status": "healthy", "languages": ["python", "json"]},
      "grammars": {"status": "healthy"},
      "queries": {"status": "healthy"},
      "index": {"status": "healthy", "symbols": 1234},
      "sandbox": {"status": "healthy", "writable": true}
    }
  }
}
```

---

## 📖 **Further Reading**

- **ADRs**: See `codeintel/docs/adrs/` for architecture decisions
- **AGENTS.md**: Repository-wide development standards
- **SUMMARY.md**: Implementation overview
- **FINAL_STATUS.md**: Complete feature status

---

## 🎯 **Best Practices**

### 1. Always Handle Exceptions

```python
from codeintel.errors import CodeIntelError

try:
    result = some_codeintel_operation()
except CodeIntelError as e:
    # All CodeIntel errors have structured context
    logger.error(f"Operation failed: {e}", extra=e.extensions)
```

### 2. Use Dependency Injection for Testing

```python
def my_function(context: ServerContext):
    # Don't use global LIMITS
    if len(data) > context.limits.max_ast_bytes:
        raise FileTooLargeError(...)
```

### 3. Validate Input Early

```python
from pydantic import BaseModel, field_validator

class MyRequest(BaseModel):
    path: str
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("path must not contain null bytes")
        return v.strip()
```

### 4. Record Metrics for All Operations

```python
from codeintel.observability import log_operation

with log_operation("complex_analysis", language="python", files=100):
    # Automatically logged with duration
    result = analyze(files)
```

---

**Version**: 1.0.0  
**Last Updated**: November 7, 2024  
**Status**: Production Ready ✅

