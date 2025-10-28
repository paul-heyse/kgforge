## Environment Setup (Preferred Order)

1. **direnv (recommended default)**
   - Install `direnv` for your shell (`sudo apt install direnv` or `brew install direnv`) and hook it via `eval "$(direnv hook bash)"` / `eval "$(direnv hook zsh)"` in your shell rc.
   - Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
   - From the repo root run `direnv allow` once. The committed `.envrc` will:
     - create or reuse `.venv/` via `uv venv`
     - run `uv sync --frozen --extra docs` (no `gpu` extras)
     - activate the environment for that shell and install pre-commit hooks automatically.
   - Future shell entries auto-refresh whenever `pyproject.toml`, `uv.lock`, or `.env` change.

2. **Bootstrap script (when direnv is unavailable)**
   - Execute `bash scripts/bootstrap.sh` from the repository root.
   - The script mirrors the `.envrc` workflow: pins Python if needed, creates `.venv`, runs `uv sync --frozen --extra docs`, activates the environment, and installs pre-commit hooks.
   - Supports overrides such as `OFFLINE=1`, `USE_WHEELHOUSE=1`, or additional extras (GPU installs require `ALLOW_GPU=1`).

3. **Manual uv flow (last resort / CI snippets)**
   - `uv python pin 3.13`
   - `uv venv`
   - `uv sync --frozen --extra docs`
   - `uvx pre-commit install -t pre-commit -t pre-push`
   - Activate via `. .venv/bin/activate` or run tools with `uv run` / `uvx`.

## Code Quality Standards


### Python Version & Environment

- **Python Version**: 3.13 (pinned)
- **Package Manager**: `uv` (fast, deterministic)
- **Environment**: `.venv/` (never use system Python)

### Code Formatting & Style

#### Ruff (Primary Tool)
- **Line length**: 100 characters
- **Quote style**: Double quotes (`"`)
- **Indentation**: 4 spaces (no tabs)
- **Import order**: stdlib → third-party → first-party (auto-sorted)
- **Trailing commas**: Required for multiline structures
- **Blank lines**: 2 before top-level definitions, 1 between methods

**Pre-commit hooks**:
1. `ruff --select I --fix` - Sorts imports
2. `ruff --fix` - Auto-fixes linting issues
3. `ruff format` - Formats code

**Key rules enforced**:
- F (pyflakes) - undefined names, unused imports
- E4/E7/E9 (pycodestyle) - whitespace, indentation
- I (isort) - import sorting
- N (naming) - PEP 8 naming conventions
- UP (pyupgrade) - modern Python syntax
- SIM (simplify) - simplification opportunities
- B (bugbear) - likely bugs
- ANN (annotations) - type annotations required
- D (docstrings) - docstring presence and format

**Complexity limits**:
- Max cyclomatic complexity: 10
- Max branches per function: 12
- Max returns per function: 6

#### Black (Safety Net)
- **Line length**: 100 characters
- **Target**: py312 format (code runs on py313)
- Runs after Ruff to catch any formatting drift

### Type Checking

#### Mypy Configuration
- **Strict mode enabled**
- **Python version**: 3.13
- **Key settings**:
  - `disallow_untyped_defs = true` - All functions must have type hints
  - `no_implicit_optional = true` - Explicit Optional[] required
  - `warn_unused_ignores = true` - No unnecessary # type: ignore
  - `warn_redundant_casts = true` - No unnecessary casts
  - `strict_equality = true` - Strict comparison checking

**Type annotation requirements**:
- All function parameters must be typed
- All return types must be specified
- Use `typing.Any` only when truly unavoidable (and document why)
- Prefer specific types over general ones (e.g., `list[str]` not `list`)

### Docstring Standards

#### NumPy Style (Required)
- **Convention**: NumPy docstring format (enforced by pydocstyle and pydoclint)
- **Coverage requirement**: 90% minimum (enforced by interrogate)
- **Validation**: numpydoc checks GL01, SS01, ES01, RT01, PR01

#### Docstring Structure

**Module docstrings**:
```python
"""Brief module description (one line).

Extended description can span multiple paragraphs. Explain the module's
purpose, main components, and usage patterns.

Examples
--------
>>> import mymodule
>>> mymodule.do_something()
```

**Function/Method docstrings**:
```python
def function_name(param1: str, param2: int = 5) -> bool:
    """Brief description in imperative mood.

    Extended description providing context, usage notes, and important
    details about the function's behavior.

    Parameters
    ----------
    param1 : str
        Description of param1. Note the space after the colon.
    param2 : int, optional
        Description of param2, by default 5

    Returns
    -------
    bool
        Description of what is returned.

    Raises
    ------
    ValueError
        When param1 is empty.
    TypeError
        When param2 is not an integer.

    See Also
    --------
    related_function : Brief description.
    another_function : Brief description.

    Notes
    -----
    Additional notes about implementation, performance characteristics,
    or important caveats.

    Examples
    --------
    >>> function_name("hello", 10)
    True
    >>> function_name("world")
    True
    """
```

**Class docstrings**:
```python
class MyClass:
    """Brief class description.

    Extended description of the class purpose and usage.

    Parameters
    ----------
    arg1 : str
        Description of constructor arg1.
    arg2 : int, optional
        Description of constructor arg2, by default 10

    Attributes
    ----------
    attr1 : str
        Description of attribute.
    attr2 : int
        Description of attribute.

    Methods
    -------
    method_name
        Brief description.

    Examples
    --------
    >>> obj = MyClass("value", 20)
    >>> obj.method_name()
    """
```

#### Docstring Requirements

1. **Summary line**: 
   - Use imperative mood ("Return..." not "Returns...")
   - One line, under 100 characters
   - End with period

2. **Parameters section**:
   - Format: `name : type` or `name : type, optional`
   - Include default values in description: "by default VALUE"
   - One blank line before and after section

3. **Returns section**:
   - Always include for non-None returns
   - Format: `type` on first line, description below
   - For multiple returns, list each separately

4. **Raises section**:
   - Document all exceptions that callers should handle
   - Format: `ExceptionType` followed by description
   - Include conditions that trigger the exception

5. **Examples section**:
   - Always include when possible
   - Use doctest format (`>>>` prompts)
   - Examples must be executable (validated by xdoctest)

#### Automated Docstring Generation

When adding new code without docstrings:
1. Run `make docstrings` - generates NumPy-style skeletons
2. Fill in descriptions manually
3. Run `make docstrings` again - formats and validates
4. Pre-commit hooks enforce coverage and style

### Import Organization

#### Import Order (Enforced by Ruff)
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel

# First-party imports
from kgfoundry_common.config import load_config
from search_api.models import SearchRequest
```

#### Import Conventions (Enforced)
- `numpy` → `np`
- `pandas` → `pd`
- `matplotlib.pyplot` → `plt`
- **No relative imports** (use absolute imports from `src/`)

### Naming Conventions

#### Variables & Functions
- **snake_case**: `my_variable`, `calculate_total()`
- **Private**: prefix with underscore `_internal_helper()`
- **Constants**: `UPPER_SNAKE_CASE` at module level

#### Classes & Exceptions
- **PascalCase**: `MyClass`, `CustomException`
- **Exceptions**: suffix with `Error` or `Exception`

#### Modules & Packages
- **snake_case**: `my_module.py`, `my_package/`
- **Short, descriptive names**: prefer `config.py` over `configuration_manager.py`

### Comments

#### Inline Comments
```python
# Use comments to explain WHY, not WHAT
result = calculate_value(x)  # Avoid: "Calculate value"
result = calculate_value(x)  # Good: "Cached for performance"
```

#### Block Comments
```python
# Use block comments for complex algorithms or non-obvious logic.
# Explain the approach and any important context.
# Keep lines under 100 characters.
```

#### TODO Comments
```python
# TODO(username): Description of what needs to be done
# FIXME(username): Description of what's broken and needs fixing
# NOTE(username): Important context or caveat
```

### Testing Standards

#### Test File Organization
- Test files mirror `src/` structure in `tests/`
- Name pattern: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function_name>_<scenario>()`

#### Docstring Examples (xdoctest)
- All `Examples` sections in docstrings are tested
- Run via `pytest --xdoctest`
- Use `...` for ellipsis matching
- Use `# doctest: +SKIP` to skip problematic examples

#### Test Markers
```python
@pytest.mark.integration  # Hits real network services
@pytest.mark.benchmark    # Performance benchmarks
@pytest.mark.real_vectors # Requires real vector data
@pytest.mark.scale_vectors # Large dataset tests
```

### Pre-Commit Hook Execution Order

When you commit, these hooks run automatically in order:

1. **Ruff (imports)** - Sorts imports
2. **Ruff (lint+fix)** - Auto-fixes linting issues
3. **Ruff (format)** - Formats code
4. **Black** - Additional formatting safety net
5. **Mypy** - Type checking (fails commit if errors)
6. **docformatter** - Formats docstrings to PEP 257
7. **pydoclint** - Validates parameter/return parity
8. **pydocstyle** - Lints docstrings for NumPy convention
9. **interrogate** - Enforces 90% docstring coverage
10. **navmap-build** - Regenerates navigation index
11. **navmap-check** - Validates navigation metadata
12. **readme-generator** (optional) - Updates package READMEs

**All hooks must pass before commit is allowed.**

### Quick Commands

```bash
# Format code
make fmt
# or manually:
uvx ruff check --fix && uvx ruff format && black .

# Type check
uvx mypy --strict src

# Generate/validate docstrings
make docstrings

# Run tests
uv run pytest -q

# Run tests with doctests
uv run pytest -q --xdoctest

# Pre-commit check (before committing)
pre-commit run --all-files

# Update documentation
tools/update_docs.sh
```

### Common Violations & Fixes

| Error | Meaning | Fix |
|-------|---------|-----|
| `D100` | Missing module docstring | Add module-level `"""Description."""` |
| `D103` | Missing function docstring | Add function docstring |
| `ANN001` | Missing type annotation | Add type hint to parameter |
| `ANN201` | Missing return type | Add `-> ReturnType` annotation |
| `F401` | Unused import | Remove import or use `# noqa: F401` if needed for re-export |
| `E501` | Line too long | Break line (Black handles this) |
| `C901` | Too complex | Refactor function (reduce branches) |

### File-Specific Exemptions

- **`tests/**`**: Docstrings optional (D100-D104 ignored)
- **`docs/_build/**`**: Excluded from all checks
- **`site/**`**: Excluded from all checks

### Editor Integration

**VS Code**:
- Ruff extension handles formatting on save
- Mypy extension shows type errors inline
- Python extension uses `.venv/` automatically

**PyCharm**:
- Configure external tool for Ruff
- Enable mypy integration
- Set Python interpreter to `.venv/bin/python`

### CI/CD Integration

All quality checks run in CI:
```bash
# Format check (fail if not formatted)
uvx ruff format --check src tests

# Lint check (fail if violations exist)
uvx ruff check src tests

# Type check (fail if type errors exist)
uvx mypy --strict src

# Run tests
uv run pytest -q

# Documentation check
tools/update_docs.sh
git diff --exit-code docs/ src/  # Fail if docs out of sync
```

### Best Practices Summary

✅ **Do**:
- Run `make fmt` before committing
- Add type hints to all new functions
- Write docstrings for all public APIs
- Keep functions small and focused (complexity < 10)
- Use descriptive variable names
- Add examples to docstrings
- Run `pre-commit run --all-files` before pushing

❌ **Don't**:
- Use `--no-verify` to skip pre-commit hooks
- Leave TODOs without attribution
- Use `# type: ignore` without explanation
- Create functions with >10 complexity
- Omit type hints (mypy strict mode)
- Write docstrings in non-NumPy format
- Commit code with <90% docstring coverage

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Environment Setup (Preferred Order)

1. **direnv (recommended default)**
   - Install `direnv` for your shell (`sudo apt install direnv` or `brew install direnv`) and hook it via `eval "$(direnv hook bash)"` / `eval "$(direnv hook zsh)"` in your shell rc.
   - Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
   - From the repo root run `direnv allow` once. The committed `.envrc` will:
     - create or reuse `.venv/` via `uv venv`
     - run `uv sync --frozen --extra docs` (no `gpu` extras)
     - activate the environment for that shell and install pre-commit hooks automatically.
   - Future shell entries auto-refresh whenever `pyproject.toml`, `uv.lock`, or `.env` change.

2. **Bootstrap script (when direnv is unavailable)**
   - Execute `bash scripts/bootstrap.sh` from the repository root.
   - The script mirrors the `.envrc` workflow: pins Python if needed, creates `.venv`, runs `uv sync --frozen --extra docs`, activates the environment, and installs pre-commit hooks.
   - Supports overrides such as `OFFLINE=1`, `USE_WHEELHOUSE=1`, or additional extras (GPU installs require `ALLOW_GPU=1`).

3. **Manual uv flow (last resort / CI snippets)**
   - `uv python pin 3.13`
   - `uv venv`
   - `uv sync --frozen --extra docs`
   - `uvx pre-commit install -t pre-commit -t pre-push`
   - Activate via `. .venv/bin/activate` or run tools with `uv run` / `uvx`.