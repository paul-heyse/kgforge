## 1. Metadata Verification & Tests
- [ ] 1.1 Add unit tests guaranteeing `ParameterHarvest` already captured `inspect.Parameter.kind` and `display_name`, including positional-only, keyword-only, varargs, and kwargs cases.
- [ ] 1.2 Extend DocFacts tests to assert `display_name` and `kind` fields propagate correctly and remain idempotent.
- [ ] 1.3 Verify downstream consumers (navmap, README generators) read the enriched metadata; add failing tests if gaps exist.

## 2. Docstring Regeneration
- [ ] 2.1 Run the builder (post-tests) on key modules (`src/docling/canonicalizer.py`, `src/download/harvester.py`, `src/embeddings_sparse/bm25.py`, others as identified).
- [ ] 2.2 Replace placeholder descriptions with accurate text (builder heuristics + domain knowledge).
- [ ] 2.3 Regenerate DocFacts; confirm idempotence (second run yields no diff).
- [ ] 2.4 Commit regenerated docstrings + DocFacts together.

## 3. CLI Restructure & Governance
- [ ] 3.1 Refactor existing command handlers into shared runner functions.
- [ ] 3.2 Implement subcommands (`generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`) with argparse help.
- [ ] 3.3 Define exit code constants and enforce them across subcommands.
- [ ] 3.4 Add `--config` flag with precedence (CLI > env `KGF_DOCSTRINGS_CONFIG` > default config).
- [ ] 3.5 Move stub drift checker behind `docstring-builder doctor --stubs`; enhance output with sorted missing/extra symbol lists.
- [ ] 3.6 Add CI job invoking `docstring-builder doctor --stubs`; fail on drift.
- [ ] 3.7 (Optional) Package stubs as PEP-561 extras; document update workflow.

## 4. Observability & Developer Experience
- [ ] 4.1 Emit `docs/_build/observability_docstrings.json` with counts, timings, cache hits/misses, top errors.
- [ ] 4.2 Generate HTML drift previews under `docs/_build/drift/` for docfacts/docstrings/navmap deltas; link them in PR guidance.
- [ ] 4.3 Provide editor tasks/snippets for `generate`, `lint`, `check`, `doctor`.
- [ ] 4.4 Document observability outputs and drift previews for reviewers.

## 5. Security Hardening
- [ ] 5.1 Normalise/validate input paths; reject traversal/symlink exploits.
- [ ] 5.2 Audit for unsafe evaluation (`eval`, `exec`, dynamic imports); replace or guard them.
- [ ] 5.3 Add regression tests covering malicious path scenarios.

## 6. Sitecustomize Deprecation Path
- [ ] 6.1 Add `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE` (default `1`); emit `DeprecationWarning` when applied.
- [ ] 6.2 Add CI coverage with the flag disabled to ensure pipeline success without patches.
- [ ] 6.3 Document timeline/migration steps for eventual removal.

## 7. Documentation & Workflow
- [ ] 7.1 Update CONTRIBUTING/AGENTS with the docstring regeneration checklist (builder → artifacts → pyrefly → pre-commit).
- [ ] 7.2 Provide examples for common scenarios (e.g., new dataclass field) and emphasise builder usage over manual edits.
- [ ] 7.3 Document plugin authoring, stub maintenance, policy configs, observability outputs, and `doctor` troubleshooting.

## 8. Validation
- [ ] 8.1 Run `uv run mypy` and `uv run ruff check` over updated modules.
- [ ] 8.2 Execute end-to-end CLI tests for subcommands (`generate/lint/fix/diff/check/schema/doctor/measure`).
- [ ] 8.3 Run `uv run pyrefly check`, `uv run pre-commit run --all-files`, and `make artifacts` to validate regenerated docstrings.
- [ ] 8.4 Ensure DocFacts consumers (navmap, README generator, etc.) operate correctly with enriched metadata.
