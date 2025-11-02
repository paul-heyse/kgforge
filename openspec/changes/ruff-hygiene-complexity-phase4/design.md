## Context
- Ruff reports `C901` and `PLR091x` violations in catalog loading, SQLite symbol attachment, and orchestration index builders, making the code brittle and hard to test.
- Multiple modules still catch bare `Exception`, masking real failures and preventing our exception taxonomy from surfacing.
- Deferred imports proliferate within search/index modules without justification, hurting cold-start predictability and static analysis.
- Pickle usage in embeddings/index builders lacks allow-list checks, raising security risks; file operations rely on `os.path` and string manipulation.
- Deprecated typing aliases (`typing.Dict`, `typing.List`) linger in `kgfoundry/__init__.py` and elsewhere, conflicting with Ruff’s modernization rules.

## Goals / Non-Goals
- **Goals**
  - Reduce complexity of identified hotspots by extracting pure helpers, typed dataclasses, and Protocols.
  - Replace `except Exception` with specific taxonomy exceptions and document each catch block rationale.
  - Normalize import strategy (top-level or justified late imports) and sanitize pickle usage.
  - Standardize on modern typing constructs and `pathlib` across filesystem interactions.
- **Non-Goals**
  - Changing high-level algorithms beyond structural decomposition.
  - Introducing new serialization formats beyond secure wrappers.
  - Refactoring unrelated modules not flagged by Ruff or Pyrefly.

## Decisions
- Create domain-specific exception classes (e.g., `CatalogLoadError`, `SymbolAttachmentError`, `IndexBuildError`) in their respective modules, all deriving from a shared `CatalogRuntimeError` to aid taxonomy reporting.
- Extract pure functions for catalog parsing, symbol attachment, and index building steps; encapsulate shared state in typed dataclasses (e.g., `IndexBuildContext`) to clarify inputs/outputs.
- Use `Protocol` definitions for loader interfaces (e.g., `SymbolLoader`, `IndexWriter`) enabling dependency injection and easier testing.
- Move all imports to module top level. For unavoidable late imports (e.g., optional GPU dependencies), add documented justification and unit tests covering missing dependency behavior.
- Introduce `safe_pickle` utility with allow-list of expected classes and version headers; wrap legacy pickle calls and add tests covering rejection paths.
- Replace `os.path` usage with `pathlib.Path`; ensure new helpers accept `Path` objects and maintain POSIX compatibility.
- Update typing across modules (`list[...]`, `dict[...]`, typed `NamedTuple`/`dataclass`) and update stubs accordingly.

## Alternatives
- Adopt a full dependency injection framework — rejected as overkill for this scope.
- Replace pickle with JSON/MsgPack — deferred to a future security-focused phase; wrappers suffice here.
- Apply Ruff suppressions — rejected to keep lint budgets meaningful.

## Risks / Trade-offs
- Extracting helpers may introduce subtle behavior changes.
  - Mitigation: Write characterization tests before refactoring; keep helpers pure and covered by unit tests.
- Adding new exception classes might complicate downstream handlers.
  - Mitigation: Provide migration notes and ensure new classes subclass existing ones where appropriate.
- `pathlib` migration could break string-based path assumptions.
  - Mitigation: Audit all call sites, adjust tests to use `Path`, and provide `.as_posix()` conversions only at boundaries.

## Migration
- Tackle modules one at a time, starting with catalog models, then SQLite, orchestration CLI, and embeddings/search modules.
- After each module refactor, run `uv run ruff check <module>` and targeted tests to confirm stability.
- Publish updated exception taxonomy and logging guidelines; add docstrings referencing the taxonomy.
- Update release notes to alert downstream consumers about potential exception type adjustments and pickle safeguards.

