## Why
Our documentation toolchain currently produces inconsistent results that block the pre-commit pipeline. The dynamic compatibility shim in `sitecustomize.py` monkey patches the `docstring_parser` library at runtime, but mypy cannot verify those injected properties, so every run fails strict type checking. At the same time, our improved docstring builder generates richer metadata (vararg and kwarg display names, DocFacts) yet the CLI invoked by pre-commit still passes obsolete `--diff/--since` flags, causing the tool to exit with a usage error. Finally, contributors must run four separate scripts (navmap build, test-map build, observability scan, schema export) after touching documentation; forgetting any one leaves the working tree dirty and breaks the next hook. We need a comprehensive plan so junior contributors can follow a deterministic workflow while senior engineers retain strong typing guarantees and repeatable builds.

## What Changes
- **Type-safe docstring shim**
  - Define Protocols (`DocstringAttrProto`, `DocstringYieldsProto`, `DocstringProto`) that mirror the relevant surface area of `docstring_parser.common`.
  - Add helper functions (`register_docstring_attrs`, `register_docstring_yields`) that wrap `setattr` calls, apply runtime guards, and return explicit `None`/`list[...]` values so mypy understands the injected attributes.
  - Gate every monkey patch behind `hasattr` checks and log debug messages (via `warnings` or `logging`) when the upstream library already exposes the attribute.
- **Docstring-builder enhancements**
  - Extend the harvester to capture parameter `kind` (positional-only, keyword-only, var positional, var keyword) and store human friendly display names (`*args`, `**kwargs`).
  - Persist harvested metadata to DocFacts so downstream tools (navmap, README generators) can consume consistent structured data without re-harvesting.
  - Update the renderer so it only edits the parameter blocks it owns, leaving any manually curated prose untouched; add snapshot tests proving idempotence.
- **CLI compatibility layer**
  - Teach `tools/docstring_builder.cli` to recognise `--diff`, `--since`, and `--module` when they come from legacy hooks, mapping them to the new subcommand API.
  - Add a `--ignore-missing` switch that skips modules such as `docs/_build/**` so `_build` imports stop raising ModuleNotFoundError during hook execution.
- **Unified artefact regeneration**
  - Introduce a top-level `make artifacts` (name finalised during implementation) that sequentially runs `navmap-build`, `docs-build-test-map`, `docs-observability-scan`, `docs-export-schemas`, `make docstrings`, and the DocFacts exporter.
  - Each step should write a concise summary line (`[navmap] rebuilt`, `[schemas] updated 6 files`) and exit early on failure so contributors know exactly what to remediate.
- **Process and documentation updates**
  - Update CONTRIBUTING.md and AGENTS.md with a “Documentation artefact workflow” section describing when to run the new command, common failure modes, and how to inspect diffs.
  - Add a CI job that runs `make artifacts` and fails the pipeline if any tracked artefact changes are detected.

## Impact
- **Specs**: Documentation Automation, Developer Tooling, Type System Compatibility.
- **Code**: `src/sitecustomize.py`, `tools/docstring_builder/**`, `tools/generate_docstrings.py`, navmap/test-map/schema scripts, Makefile/CONTRIBUTING, and the pre-commit configuration.
