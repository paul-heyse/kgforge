# Type Safety & Namespace Alignment Guide

This guide describes best practices for maintaining type safety across the kgfoundry codebase, with emphasis on namespace proxy patterns and stub file alignment.

## Overview

The kgfoundry codebase uses two complementary mechanisms to ensure type safety:

1. **NamespaceRegistry** (`src/kgfoundry/_namespace_proxy.py`): A typed registry for lazy-loading module symbols without relying on `Any` types.
2. **Stub Files** (`stubs/kgfoundry/**/*.pyi`): Type stub files that mirror runtime module exports, enabling fast type checking and IDE support.

Both mechanisms must stay synchronized to prevent type-checking errors and broken documentation tooling.

## NamespaceRegistry Usage

### What It Does

The `NamespaceRegistry` class provides:

- **Typed Symbol Registration**: Register symbols with loaders that return specific types
- **Lazy Loading**: Symbols are loaded only when accessed, reducing import overhead
- **Caching**: Resolved symbols are cached to avoid repeated loader invocations
- **Error Handling**: Clear error messages listing available symbols when resolution fails

### Basic Pattern

```python
from kgfoundry._namespace_proxy import NamespaceRegistry

# Create a registry
registry = NamespaceRegistry()

# Register symbols with lazy loaders
registry.register("module_a", lambda: importlib.import_module("kgfoundry.module_a"))
registry.register("module_b", lambda: importlib.import_module("kgfoundry.module_b"))

# Resolve symbols
module_a = registry.resolve("module_a")
module_b = registry.resolve("module_b")

# List available symbols
available = registry.list_symbols()  # Returns: ["module_a", "module_b"]
```

### Benefits Over Dynamic `Any` Types

```python
# ❌ BEFORE: Dynamic with Any
def __getattr__(name: str) -> Any:
    """Type-unsafe dynamic attribute resolution."""
    return _load(name)

# ✅ AFTER: Using NamespaceRegistry
registry = NamespaceRegistry()
# ... register symbols with loaders ...

def __getattr__(name: str) -> object:
    """Type-safe lazy loading with explicit registration."""
    return registry.resolve(name)  # Raises KeyError with helpful message
```

## Stub File Alignment Workflow

### When to Update Stubs

Update stub files whenever you:

1. Add new public exports to a module
2. Change function signatures or return types
3. Modify dataclass fields or add new dataclasses
4. Change type aliases or Protocol definitions

### Step-by-Step: Adding a New Export

#### 1. Implement the Runtime Feature

```python
# src/kgfoundry/agent_catalog/search.py

class NewSearchHelper:
    """A new public helper class."""
    
    def __init__(self, config: str) -> None:
        self.config = config

# Update __all__
__all__ = [
    # ... existing exports ...
    "NewSearchHelper",
]
```

#### 2. Update the Stub File

```python
# stubs/kgfoundry/agent_catalog/search.pyi

@dataclass(slots=True)
class NewSearchHelper:
    """A new public helper class."""
    
    config: str
    
    def __init__(self, config: str) -> None: ...
```

#### 3. Verify Parity

Run the parity check script to ensure alignment:

```bash
python tools/check_stub_parity.py
```

Expected output:
```
Checking: kgfoundry.agent_catalog.search
======================================================================
  ✓ All runtime exports present in stub
  ✓ No problematic Any types found
```

### Best Practices for Stubs

#### ✅ DO

- **Use precise types**: Replace `Any` with concrete types or Protocol definitions
- **Include docstrings**: Copy docstrings from runtime code for IDE tooltips
- **Document limitations**: Add comments explaining why casts or ignores are needed
- **Re-export explicitly**: Use `as` syntax for clarity: `from module import Name as Name`
- **Test parity regularly**: Run `python tools/check_stub_parity.py` before committing

#### ❌ DON'T

- **Don't use `Any` in stubs**: This defeats the purpose of type-safe documentation
- **Don't leave stale symbols**: Remove symbols that no longer exist in runtime
- **Don't assume concordance**: Always verify stubs match runtime via the parity script
- **Don't hide complexity**: Use Protocol definitions to clarify expected interfaces

### Handling Complex Types

#### Type Aliases for Clarity

```python
# stubs/kgfoundry/agent_catalog/search.pyi

# Define type aliases for JSON-compatible values
JsonValue = str | int | float | bool | None

class SearchContext:
    """Context for search operations."""
    
    metadata: Mapping[str, JsonValue]  # Clear, reusable
```

#### Structural Typing with Protocols

```python
from typing import Protocol

class EmbeddingModelProtocol(Protocol):
    """Protocol describing embedding model interface."""
    
    def encode(self, sentences: Sequence[str], **kwargs: object) -> Sequence[Sequence[float]]: ...
```

## Type Checking Standards

### Ruff + Mypy + Pyrefly Gates

All changes must pass these quality gates with **zero suppressions** in the changed files:

```bash
# Format & lint
uv run ruff format && uv run ruff check --fix

# Type checks
uv run pyrefly check
uv run mypy --config-file mypy.ini

# Tests
uv run pytest -q
```

### Handling Legitimate Type Errors

When type-checker limitations require suppressions:

1. **Document the reason**: Add a comment explaining why the suppression is needed
2. **Use specific error codes**: Target the exact error, not broad ignores
3. **Reference constraints**: Link to type stub limitations, framework issues, etc.

**Example:**

```python
# numpy array indexing returns Any due to stub limitations (not a code quality issue)
query_row: np.ndarray = cast(np.ndarray, indices[0, :])  # type: ignore[type-arg,misc]
```

## Common Patterns & Anti-Patterns

### ✅ Proper Type Narrowing

```python
# GOOD: Type-narrow with isinstance before casting
docfacts_payload = symbol.get("docfacts")
docfacts_input: Mapping[str, JsonLike] | None = (
    cast(Mapping[str, JsonLike], docfacts_payload)
    if isinstance(docfacts_payload, Mapping)
    else None
)
```

### ❌ Problematic Patterns

```python
# BAD: Cast without validation
payload: dict[str, Any] = cast(dict[str, Any], unknown_data)

# BAD: Suppress without explanation
result = some_function()  # type: ignore

# BAD: Stubs with Any
def process_data(data: Any) -> Any: ...  # Defeats type safety goal
```

## Verification Commands

### Quick Checks

```bash
# Check namespace exports
python tools/check_stub_parity.py

# Check specific module
uv run mypy --config-file mypy.ini src/kgfoundry/agent_catalog/search.py
uv run pyrefly check src/kgfoundry/agent_catalog/search.py

# Run namespace tests
uv run pytest tests/test_namespace_proxy.py -v
```

### Full Quality Gate

```bash
# Complete pre-commit checks
uv run ruff format && uv run ruff check --fix
uv run pyrefly check
uv run mypy --config-file mypy.ini
uv run pytest -q
```

## Troubleshooting

### Stub Parity Check Fails

**Issue**: `Missing in stub: ['NewSymbol']`

**Solution**:
1. Add `NewSymbol` to the stub file
2. Use matching type signature
3. Rerun: `python tools/check_stub_parity.py`

### Type Checker Reports Unused Ignore

**Issue**: Ruff reports `RUF100: Unused noqa directive`

**Solution**:
1. Remove the `# noqa` comment if truly unnecessary
2. If suppression is still needed, ensure it targets an actual error
3. Run full type checks to confirm necessity

### Circular Import in Stubs

**Issue**: Stub imports cause circular dependency

**Solution**:
```python
# Use TYPE_CHECKING to avoid runtime circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kgfoundry.other_module import SomeType

def function() -> SomeType | None: ...  # OK in TYPE_CHECKING context
```

## Further Reading

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 561 - Distributing Type Information](https://peps.python.org/pep-0561/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pyrefly Documentation](https://github.com/pyrefly/pyrefly)
