# ADR 003: Dependency Injection for Configuration

**Status**: Accepted  
**Date**: 2024-11-07  
**Context**: CodeIntel Enhancement Implementation

## Context

The original CodeIntel implementation used global state (`LIMITS` constant loaded at module import time), making it:
1. Hard to test with different configurations
2. Impossible to run multiple server instances with different limits
3. Difficult to swap configurations dynamically
4. Violates AGENTS.md Principle 7 (no global state)

## Decision

We refactored `codeintel/config.py` to support **dependency injection** with immutable dataclasses and factory methods.

### Core Design

```python
@dataclass(frozen=True)
class ServerLimits:
    """Immutable configuration for resource limits."""
    max_ast_bytes: int
    max_outline_items: int
    tool_timeout_s: float
    # ... more fields

    @classmethod
    def from_env(cls) -> ServerLimits:
        """Load from environment variables."""
        pass

    @classmethod
    def defaults(cls) -> ServerLimits:
        """Standard defaults for production."""
        pass

    @classmethod
    def permissive(cls) -> ServerLimits:
        """Relaxed limits for testing."""
        pass
```

### Dependency Injection Pattern

```python
@dataclass(frozen=True)
class ServerContext:
    """Bundles configuration + dependencies for services."""
    limits: ServerLimits
    repo_root: Path

    @classmethod
    def from_env(cls) -> ServerContext:
        """Production configuration from environment."""
        return cls(
            limits=ServerLimits.from_env(),
            repo_root=Path(os.environ.get("KGF_REPO_ROOT", Path.cwd()))
        )

    @classmethod
    def for_testing(cls, repo_root: Path | None = None) -> ServerContext:
        """Test configuration with permissive limits."""
        return cls(
            limits=ServerLimits.permissive(),
            repo_root=repo_root or Path.cwd()
        )
```

### Backward Compatibility

```python
# Module-level singleton for existing code
def get_limits() -> ServerLimits:
    """Get lazily-loaded singleton (backward compat)."""
    global _LIMITS
    if _LIMITS is None:
        _LIMITS = ServerLimits.from_env()
    return _LIMITS

# Exposed at module level
LIMITS = get_limits()
```

## Consequences

### Positive

- ✅ **Testable**: Tests can inject custom configurations
- ✅ **Multi-instance**: Different servers can have different limits
- ✅ **Type-safe**: Pydantic/dataclass validation
- ✅ **Explicit dependencies**: No hidden globals
- ✅ **12-factor compliant**: Configuration via environment

### Negative

- ⚠️ **More verbose**: Must pass `context` to functions
- ⚠️ **Migration needed**: Old code using `LIMITS` needs updating

### Neutral

- 📊 **Backward compatible**: `LIMITS` still works for gradual migration
- 📊 **Documented**: Comprehensive docstrings with examples

## Usage Examples

### Production (Environment Variables)

```python
# Server startup
context = ServerContext.from_env()
server = MCPServer(context)
```

### Testing (Explicit Configuration)

```python
# Test fixture
@pytest.fixture
def test_context(tmp_path):
    return ServerContext.for_testing(repo_root=tmp_path)

def test_tool(test_context):
    result = run_tool(test_context, ...)
```

### Custom Configuration

```python
limits = ServerLimits(
    max_ast_bytes=2_097_152,  # 2 MiB
    tool_timeout_s=30.0,
    # ... other fields with defaults
)
context = ServerContext(limits=limits, repo_root=Path("/workspace"))
```

## Implementation Notes

### Immutability

All configuration classes are `frozen=True` dataclasses, preventing accidental mutation:

```python
limits = ServerLimits.defaults()
limits.max_ast_bytes = 9999  # ❌ FrozenInstanceError
```

### Environment Variables

```bash
CODEINTEL_MAX_AST_BYTES=2097152
CODEINTEL_TOOL_TIMEOUT_S=30.0
CODEINTEL_RATE_LIMIT_QPS=10.0
KGF_REPO_ROOT=/workspace/myrepo
```

## Alternatives Considered

### 1. Keep Global State
**Rejected**: Violates AGENTS.md, makes testing difficult

### 2. Pydantic Settings
**Considered**: Could use `pydantic_settings.BaseSettings`, but dataclasses are simpler and sufficient

### 3. Dependency Injection Framework
**Rejected**: Overkill for current needs; explicit passing is clear

## References

- 12-Factor App: https://12factor.net/config
- AGENTS.md Principle 6: Configuration via environment variables
- AGENTS.md Principle 7: No global state, explicit DI

