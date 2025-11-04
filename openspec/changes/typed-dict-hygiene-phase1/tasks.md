## 1. Implementation
- [x] 1.1 Audit context definitions and usage
  - [x] Enumerate all declarations of `LogContextExtra`, `ErrorReport`, and `CliResult` plus call sites
  - [x] Document required vs optional fields and note mutation hotspots.
- [x] 1.2 Define hardened data structures
  - [x] Introduce frozen dataclasses or PEP 655 TypedDicts with `Required` / `NotRequired` partitions for each context.
  - [x] Provide helper constructors (e.g., `LogContextExtra.new`) that enforce required fields and attach schema metadata.
- [x] 1.3 Update consumers to use safe accessors
  - [x] Replace direct indexing of optional keys with `.get` or accessor helpers.
  - [x] Ensure helper methods return new instances when updating contexts (immutability).
- [x] 1.4 Observability alignment
  - [x] Ensure logging APIs emit RFC 9457 Problem Details with correlation IDs and operation metadata.
  - [x] Add canonical JSON example under `schema/examples/problem_details/`.
  - [x] Integrate metrics/logging hooks to track context mutations and Problem Details emissions.
- [x] 1.5 Regression coverage
  - [x] Extend pytest suites with table-driven cases covering missing optional fields, migration helpers, and Problem Details payloads.
  - [x] Add doctest/xdoctest examples demonstrating safe accessor patterns.
  - [x] 19 tests created covering immutability, conversions, logging integration, accessor patterns, and observability.
- [x] 1.6 Documentation updates
  - [x] Refresh contributor docs outlining logging context structures, accessor helpers, and Problem Details expectations.

## 2. Testing & Quality Gates
- [x] 2.1 `uv run pytest -q tests/test_logging.py tests/docstring_builder` - All tests passing
- [x] 2.2 `uv run pytest --doctest-modules kgfoundry_common tools/docstring_builder` - Doctests verified
- [x] 2.3 `uv run ruff format && uv run ruff check --fix` - All checks passing
- [x] 2.4 `uv run pyright --warnings --pythonversion=3.13` - 0 errors, 1 warning
- [x] 2.5 `uv run pyrefly check` - 0 errors
- [x] 2.6 `uv run pyright --warnings --pythonversion=3.13` - Success on all files
- [x] 2.7 `make artifacts && git diff --exit-code` - Artifacts regenerated

