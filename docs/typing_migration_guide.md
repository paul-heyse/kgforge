# Typing Gates Migration Guide

## Purpose

This guide helps developers migrate the codebase to use the new typing gates infrastructure, ensuring type-only imports are properly guarded and postponed annotations are applied consistently.

## Quick Start

### For Existing Modules

If you're working on a module that needs updating:

1. **Add postponed annotations** (if not already present):
   ```python
   from __future__ import annotations
   ```
   This must be the first import (after shebang/encoding).

2. **Move type-only imports behind TYPE_CHECKING**:
   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       import numpy as np
       from fastapi import FastAPI
   ```

3. **Use the typing façade for runtime access**:
   ```python
   from kgfoundry_common.typing import gate_import
   
   # Only when type is genuinely needed at runtime:
   np = gate_import("numpy", "array reshaping")
   ```

4. **Run quality checks**:
   ```bash
   uv run ruff format && uv run ruff check --fix
   uv run pyright --warnings --pythonversion=3.13
   uv run mypy --config-file mypy.ini
   python -m tools.lint.check_typing_gates .
   ```

### For New Modules

Always start with:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only imports here
    pass
```

## Common Patterns

### Pattern 1: Type Alias Usage
```python
# Before
from docs.types.artifacts import NavMap, ProblemDetails

# After
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docs.types.artifacts import NavMap, ProblemDetails
```

### Pattern 2: Runtime Dependency (Rare)
```python
# Before: Eager import breaks when dependency is missing
import numpy as np

def process(arr: np.ndarray) -> None:
    result = np.sum(arr)

# After: Lazy import with clear intent
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from kgfoundry_common.typing import gate_import

def process(arr: np.ndarray) -> None:
    np = gate_import("numpy", "sum operation in process()")
    result = np.sum(arr)
```

### Pattern 3: Façade Imports (Preferred)
```python
# Before: Direct private module access
from docs._types import SymbolDeltaPayload

# After: Use canonical façade
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kgfoundry_common.typing import SymbolDeltaPayload
```

## Ruff Configuration

The following Ruff rules are now **errors** by default:
- `TC001–TC003`: Type-checking import violations
- `INP001`: Implicit namespace packages (require `__init__.py`)
- `EXE001–002`: Executable shebang rules
- `PLC2701`: Type-only import used at runtime (except in façade modules)

**Per-file ignores** allow controlled re-exports in:
- `src/kgfoundry_common/typing/__init__.py`
- `tools/typing/__init__.py`
- `docs/typing/__init__.py`
- `docs/_scripts/**` (internal tooling)
- `docs/types/` (temporary during migration)

## Tools & Utilities

### Apply Postponed Annotations
```bash
# To entire src/ directory
python -m tools.lint.apply_postponed_annotations src/

# Check without modifying
python -m tools.lint.apply_postponed_annotations --check-only tools/
```

### Check Typing Gates
```bash
# Check for unguarded type-only imports
python -m tools.lint.check_typing_gates src/

# JSON output for CI integration
python -m tools.lint.check_typing_gates --json tools/ docs/
```

## Testing Your Changes

1. **Unit tests** must pass:
   ```bash
   uv run pytest tests/ -q
   ```

2. **Type checking** must pass:
   ```bash
   uv run pyright --warnings --pythonversion=3.13
   uv run mypy --config-file mypy.ini
   uv run pyrefly check
   ```

3. **Linting** must pass:
   ```bash
   uv run ruff format
   uv run ruff check --fix
   ```

4. **Typing gates** must pass:
   ```bash
   python -m tools.lint.check_typing_gates src/ tools/ docs/
   ```

## Migration Timeline

| Phase | Target | Deadline | Action |
|-------|--------|----------|--------|
| 1 | Core infrastructure (✅ complete) | Complete | Façade modules + fixer tools deployed |
| 2 | Runtime modules batch 1 | Next sprint | Refactor src/kgfoundry_common, src/kgfoundry |
| 2 | Runtime modules batch 2 | +1 sprint | Refactor src/search_api, src/orchestration |
| 2 | Tooling/docs | +1 sprint | Refactor tools/, docs/_scripts/ |
| 3 | Remove compat shims | +2 releases | Enforce façade-only imports |

## Troubleshooting

### "Cannot proceed with X: 'numpy' is not installed"
This error means `gate_import()` was called but the module is missing. Either:
1. Install the missing optional dependency: `pip install numpy`
2. Move the import behind a TYPE_CHECKING block
3. Check if the import is truly needed at runtime

### Circular import after adding TYPE_CHECKING
This is expected—annotations don't execute at runtime. Ensure the import is only in the TYPE_CHECKING block:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mymodule import Type  # Safe here, not executed at runtime
```

### "TC003: Move standard library import into TYPE_CHECKING block"
Standard library imports like `pathlib.Path`, `collections.abc.*`, `types.*` used only in annotations should move to TYPE_CHECKING blocks:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

# Runtime code uses string annotations instead
def process(path: Path, cb: Callable) -> None:
    ...  # Annotations aren't evaluated, so Path/Callable are safe
```

## Getting Help

1. Check the [Typing Gates](https://github.com/kgfoundry/kgfoundry/blob/main/AGENTS.md#typing-gates-postponed-annotations--type_checking-hygiene) section in AGENTS.md
2. Review examples in `tests/test_typing_facade.py` and `tests/test_runtime_determinism.py`
3. Ask in #typing-gates or open an issue with the `typing-gates` label

## References

- **PEP 563**: Postponed Evaluation of Annotations
- **RFC 9457**: Problem Details for HTTP APIs
- **Ruff Documentation**: [Type-checking imports](https://docs.astral.sh/ruff/rules/flake8-type-checking/)
- **Typing Best Practices**: [AGENTS.md Typing Gates section](https://github.com/kgfoundry/kgfoundry/blob/main/AGENTS.md#typing-gates-postponed-annotations--type_checking-hygiene)
