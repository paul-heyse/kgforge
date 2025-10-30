# Docstring Builder Hardening — Contributor Guide

This change introduces typed models, stricter schema validation, plugin protocols, and secure CLI infrastructure for the docstring builder ecosystem. Junior developers should use this guide alongside `proposal.md`, `design.md`, `tasks.md`, and the capability spec under `specs/docstring-tooling/`.

## Quick Start Checklist
1. `scripts/bootstrap.sh` — ensures Python 3.13.9 and `uv` tooling match CI.
2. Read:
   - `proposal.md` (scope, acceptance, risks)
   - `design.md` (architecture, typed models, plugin protocol)
   - `tasks.md` (step-by-step execution plan)
   - `specs/docstring-tooling/spec.md` (requirements + scenarios)
3. Inspect the new canonical definitions:
   - `tools/docstring_builder/models.py` for typed IR, DocFacts, CLI payload, and exception taxonomy.
   - `schema/tools/docstring_builder_cli.json` for the Draft 2020-12 CLI schema.
4. Capture baseline lint/type failures so you can demonstrate improvement:
   ```bash
   uv run ruff check tools/docstring_builder
   uv run mypy tools/docstring_builder
   ```
5. Run the builder CLI once to understand current behavior:
   ```bash
   uv run python -m tools.docstring_builder.cli generate --json --dry-run
   ```

## Key Deliverables
- `tools/docstring_builder/models.py` — authoritative typed definitions for DocFacts entries, Docstring IR, CLI payloads, Problem Details, and the builder’s exception taxonomy.
- `schema/tools/docstring_builder_cli.json` — machine-readable contract for `--json` outputs (Draft 2020-12).
- `tools/_shared/logging.py`, `tools/_shared/proc.py` — shared structured logging and safe subprocess utilities.
- Updated core modules (`normalizer.py`, `policy.py`, `render.py`, `cli.py`) refactored to use the typed models and to emit/validate Problem Details payloads.
- Plugin Protocol in `plugins/base.py` plus updated bundled plugins.
- Tests covering schemas, plugins, CLI flows, and observability (see `design.md` testing matrix).
- Migration notes documenting the feature-flag rollout and compatibility shim retirement.

## Resources & References
- DocFacts schema: `docs/_build/schema_docfacts.json`
- CLI schema: `schema/tools/docstring_builder_cli.json`
- Typed models: `tools/docstring_builder/models.py`
- Docstring tooling backlog: `openspec/changes/docstring-tooling-phase-2/tasks.md`
- Security standards: `docs/contributing/quality.md`
- Structured logging patterns: search for `get_logger` in `src/`

## Support
- Domain owner: Doc Tooling team (#doc-tooling channel)
- Implementation owner: Listed in PR assignees
- For schema questions: reach out to Docs Platform architects

Keep the PR description up to date with:
- 4-item design note
- Command outputs for acceptance gates
- Links to generated artifacts (docs site, Agent Portal, schema validation logs)


