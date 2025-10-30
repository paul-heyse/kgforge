## 1. DocFacts 2.0 (Schema, Versioning, Richer Metadata)
- [x] 1.1 Define `docs/_build/schema_docfacts.json` (JSON Schema) with `docfactsVersion`.
- [x] 1.2 Enrich DocFacts: `filepath`, `lineno`, `end_lineno`, `decorators`, `is_async`, `is_generator`, `owned`, and include return/raise descriptions.
- [x] 1.3 Add provenance block (builder version, config hash, commit hash, timestamp).
- [x] 1.4 Validate DocFacts against schema in every run; add CI gate failing on invalid/drifted payloads.

## 2. Metadata Verification & Renderer Parity
- [ ] 2.1 Add tests guaranteeing `ParameterHarvest` fidelity for positional-only, keyword-only, varargs, kwargs.
- [ ] 2.2 Assert renderer uses `display_name` and kinds correctly in generated sections.
- [x] 2.3 Extend DocFacts tests to assert propagation and idempotence of enriched fields.
- [ ] 2.4 Verify downstream consumers (navmap/README generators) tolerate enriched DocFacts; add failing tests if gaps exist.

## 3. Policy Engine Upgrades
- [ ] 3.1 Implement rules: missing examples, non‑imperative summaries, dataclass/attrs field parity.
- [ ] 3.2 Add `doctor --policy` summary (active rules, exceptions, expiries).
- [ ] 3.3 Unit/E2E tests for new rules and exceptions.

## 4. Plugins for Higher-Fidelity Content
- [ ] 4.1 Dataclass/attrs plugin: derive parameter descriptions from field metadata/defaults.
- [ ] 4.2 Enhance AST exception inference; add parameter-description seeders (names/defaults aware).
- [ ] 4.3 Optional LLM-assisted summary rewriter behind opt-in flag; dry-run mode and tests.

## 5. Rendering & Formatting Improvements
- [ ] 5.1 Add optional “Signature” line (pos-only/kw-only/*args/**kwargs) gated by config.
- [ ] 5.2 Introduce `fmt` subcommand to normalize spacing/sections without regeneration.
- [ ] 5.3 Improve normalization of multiline defaults and types; add tests.

## 6. CLI Performance & Machine Outputs
- [ ] 6.1 Implement `--jobs N` parallel processing with stable ordering and deterministic outputs.
- [ ] 6.2 Add `--json` output (per-file outcomes, summary) for CI consumption.
- [ ] 6.3 Add `--baseline <rev>` comparison mode; tests for local and CI defaults.

## 7. Observability & Drift UX
- [ ] 7.1 Generate consolidated docstrings drift HTML; link from manifest.
- [ ] 7.2 Add trend metrics (coverage, violations) to observability JSON.
- [ ] 7.3 Provide PR guidance and link previews; add editor tasks/snippets for `generate`, `lint`, `check`, `doctor`, `artifacts`.

## 8. Security Hardening
- [ ] 8.1 Normalize/validate input paths with explicit diagnostics; reject traversal/symlink exploits.
- [ ] 8.2 Audit for unsafe evaluation (`eval`, `exec`, dynamic imports); replace/guard them.
- [ ] 8.3 Add regression tests covering nested symlinks and malicious path scenarios.

## 9. Sitecustomize Deprecation Path
- [ ] 9.1 Add CI matrix leg with `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=0`.
- [ ] 9.2 Document timeline/migration steps for removal; verify pipeline health with shim disabled.

## 10. Documentation & Workflow
- [ ] 10.1 Update CONTRIBUTING/AGENTS with the regeneration checklist (builder → artifacts → pyrefly → pre-commit).
- [ ] 10.2 Provide examples (e.g., new dataclass field) and emphasize builder usage over manual edits.
- [ ] 10.3 Document plugin authoring, stub maintenance, policy configs, observability outputs, and drift previews.

## 11. Docstring Regeneration
- [ ] 11.1 Run the builder (post-tests) on key modules (`src/docling/canonicalizer.py`, `src/download/harvester.py`, `src/embeddings_sparse/bm25.py`, others as identified).
- [ ] 11.2 Replace placeholder descriptions with accurate text (builder heuristics + domain knowledge).
- [x] 11.3 Regenerate DocFacts; confirm idempotence (second run yields no diff).
- [ ] 11.4 Commit regenerated docstrings + DocFacts together.

## 12. Validation
- [ ] 12.1 Run `uv run mypy` and `uv run ruff check` over updated modules.
- [ ] 12.2 Execute end-to-end CLI tests for subcommands and new flags (`generate/lint/fix/diff/check/schema/doctor/measure/fmt` + `--jobs/--json/--baseline`).
- [ ] 12.3 Run `uv run pyrefly check`, `uv run pre-commit run --all-files`, and `make artifacts` to validate regenerated docstrings.
- [ ] 12.4 Ensure DocFacts consumers (navmap, README generator, etc.) operate correctly with enriched metadata.

## Acceptance
- [ ] A versioned DocFacts file validates against its schema; drift shows an HTML preview.
- [ ] Parameter kinds/display names correct across positional-only/kw-only/*args/**kwargs and methods/dataclass constructors.
- [ ] Policy coverage ≥ threshold; violations include missing examples and non‑imperative summaries.
- [ ] `--jobs` yields measurable runtime reduction; `--json` integrates with CI; `--baseline` comparisons work locally and in CI.
