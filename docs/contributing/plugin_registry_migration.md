# Plugin Registry Migration Guide

## Overview

The plugin registry has been hardened to use **factory-based registration** instead of direct Protocol instantiation. This guide explains the changes and how to migrate custom plugins.

## What Changed

### Before (Old Pattern - Deprecated)

```python
from tools.docstring_builder.plugins.base import TransformerPlugin

class MyPlugin(TransformerPlugin):
    name = "my-plugin"
    stage = "transformer"
    
    def apply(self, context, payload):
        return payload
```

### After (New Pattern - Required)

The plugin itself doesn't change! Instead of registering the class directly, it's now used as a factory:

```python
from tools.docstring_builder.plugins import load_plugins

# MyPlugin class is the same as before
# But now load_plugins() wraps it in a factory automatically

manager = load_plugins(
    config,
    repo_root,
    builtin=[MyPlugin],  # Pass the class, not an instance!
)
```

## Key Requirements for Plugins

1. **No-argument initialization**: Plugin classes must have `__init__(self)` with no required parameters:
   ```python
   ✅ GOOD
   def __init__(self) -> None:
       self._cache = {}
   
   ❌ BAD
   def __init__(self, config: Config) -> None:  # Required parameter!
       self.config = config
   ```

2. **Concrete classes only**: Plugins must be concrete implementations, not Protocols or abstract classes:
   ```python
   ✅ GOOD
   class MyPlugin(TransformerPlugin):
       def apply(self, context, payload):
           return payload
   
   ❌ BAD
   from typing import Protocol
   class MyPlugin(Protocol):
       ...  # Protocol classes are rejected!
   ```

3. **Standard plugin attributes**: All plugins must define:
   ```python
   class MyPlugin(TransformerPlugin):
       name: str = "my-unique-name"     # For identification
       stage: PluginStage = "transformer"  # For routing
       
       def on_start(self, context: PluginContext) -> None:
           """Setup before processing."""
           ...
       
       def on_finish(self, context: PluginContext) -> None:
           """Cleanup after processing."""
           ...
       
       def apply(self, context: PluginContext, payload: SemanticResult) -> SemanticResult:
           """Main plugin logic."""
           return payload
   ```

## Error Handling

If your plugin violates these requirements, you'll get a detailed `PluginRegistryError` with:

```json
{
  "type": "https://kgfoundry.dev/problems/configuration-error",
  "title": "PluginRegistryError",
  "status": 500,
  "detail": "Cannot register Protocol class ...",
  "instance": "urn:tool:plugin-registry:my-plugin",
  "code": "configuration-error",
  "reason": "is-protocol",
  "plugin_name": "my-plugin",
  "stage": "transformer"
}
```

## Migration Checklist

For existing custom plugins:

- [ ] Ensure plugin class has `__init__(self)` with no required parameters
- [ ] Ensure plugin is a concrete class (not Protocol, not abstract)
- [ ] Ensure `name` and `stage` attributes are defined
- [ ] Implement all required methods: `on_start`, `on_finish`, `apply`
- [ ] Test by passing the class to `load_plugins(builtin=[MyPlugin])`

## Built-in Plugins

All built-in plugins have been validated and work correctly:

- `DataclassFieldDocPlugin` - Populates dataclass parameter docs
- `LLMSummaryRewritePlugin` - Rewrites summaries with LLM
- `NormalizeNumpyParamsPlugin` - Normalizes parameter descriptions

## Legacy Support

Legacy plugins using the old `run()` API are automatically wrapped in a compatibility adapter and emit a `DeprecationWarning`. Migration is recommended but not required.

## Example: Custom Plugin

```python
from tools.docstring_builder.plugins.base import (
    TransformerPlugin,
    PluginContext,
)
from tools.docstring_builder.semantics import SemanticResult

class MyCustomPlugin(TransformerPlugin):
    """My custom transformation plugin."""
    
    name: str = "my-custom-plugin"
    stage: str = "transformer"
    
    def __init__(self) -> None:
        """Initialize with no required parameters."""
        self._metadata = {}
    
    def on_start(self, context: PluginContext) -> None:
        """Setup: called once before processing begins."""
        self._metadata = {"files_processed": 0}
    
    def on_finish(self, context: PluginContext) -> None:
        """Teardown: called once after processing completes."""
        print(f"Processed {self._metadata['files_processed']} files")
    
    def apply(
        self,
        context: PluginContext,
        payload: SemanticResult,
    ) -> SemanticResult:
        """Apply transformation to semantic analysis result."""
        self._metadata["files_processed"] += 1
        # Your transformation logic here
        return payload


# Usage
if __name__ == "__main__":
    from tools.docstring_builder.plugins import load_plugins
    from tools.docstring_builder.config import BuilderConfig
    from pathlib import Path
    
    config = BuilderConfig()
    manager = load_plugins(
        config,
        Path.cwd(),
        builtin=[MyCustomPlugin],
    )
    print(f"Loaded plugins: {manager.enabled_plugins()}")
```

## FAQ

**Q: Do I need to change my plugin class?**
A: No! The plugin class code stays the same. Just ensure it meets the requirements above.

**Q: What about entry points?**
A: Entry point plugins are discovered and validated the same way. The factory pattern is transparent.

**Q: Can I have initialization parameters?**
A: Not in the `__init__` method. Use `on_start()` to access the `PluginContext` and initialize from there.

**Q: What if my plugin needs configuration?**
A: Access `context.config` in the `apply()` or `on_start()` methods.

## Support

For issues or questions about plugin migration:
1. Check this guide's examples
2. Review the spec at `openspec/changes/docbuilder-plugin-registry-hardening-phase1/`
3. Look at built-in plugins for reference implementations
