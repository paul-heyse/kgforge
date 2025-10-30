## Why
Phase 1 hardened the docstring ecosystem: typed shims, local stubs with drift checks, a deterministic builder (manifest + policy + plugins), and pre-commit enforcement. We now have subcommands, exit codes, config precedence, DocFacts emission, and observability logs.

Phase 2 focuses on making the pipeline authoritative and first-class for reviewers and CI: formalizing DocFacts with a schema and provenance, enriching metadata, adding high-signal drift previews and trend metrics, tightening policy rules, improving performance and machine-readable outputs, and completing the deprecation path for the runtime shim. We will also refresh key module docstrings to replace placeholders.

## What Changes
- **DocFacts 2.0 (schema, versioning, richer metadata)**
  - Define and publish a JSON Schema for DocFacts with a version tag; validate on every run and in CI.
  - Enrich DocFacts entries with `filepath` (repo-relative), `lineno`/`end_lineno`, `decorators`, `is_async`, `is_generator`, and `owned` markers; include return/raise descriptions.
  - Add a provenance block: builder version, config hash, commit hash, generation timestamp.

- **Metadata verification & renderer parity**
  - Unit tests proving `ParameterHarvest` fidelity for positional-only, keyword-only, varargs, kwargs; renderer must consume `display_name` and preserve kinds.
  - DocFacts tests asserting propagation and idempotence of enriched fields across runs.

- **Regenerate authoritative docstrings**
  - Refresh high-value modules (`src/docling/canonicalizer.py`, `src/download/harvester.py`, `src/embeddings_sparse/bm25.py`, …), replacing placeholders with accurate descriptions.
  - Commit regenerated docstrings and DocFacts together; guarantee idempotence.

- **CLI performance and machine outputs**
  - Add `--jobs N` for parallel file processing with per-file isolation and stable ordering.
  - Add `--json` to emit machine-readable per-file outcomes; add `--baseline <rev>` to compare against a fixed reference revision.

- **Observability and drift UX**
  - Generate a consolidated “docstrings drift” HTML preview alongside DocFacts/NavMap/Schema previews; add trend metrics (coverage/violations over time) to observability JSON.
  - Provide PR guidance and links to drift previews; ship editor tasks/snippets for frequent commands.

- **Policy engine upgrades**
  - New rules: “missing examples”, “non‑imperative summaries”, and “dataclass/attrs field parity”.
  - Expose policy exceptions and expiries via `doctor --policy` summary.

- **Plugins for higher-fidelity content**
  - Dataclass/attrs plugin: derive parameter descriptions from field metadata/defaults.
  - Enhance exception inference (AST) and add parameter-description seeders based on names/defaults.
  - Optional LLM-assisted post-processor behind an opt-in flag to rewrite summaries into imperative mood.

- **Rendering and formatting improvements**
  - Optional “Signature” line rendering (positional-only/keyword-only/varargs/kwargs) gated by config.
  - Add `fmt` subcommand to normalize sections/spacing without regeneration; improve normalization for multiline defaults/types.

- **Security hardening**
  - Strict path normalization and symlink traversal checks with explicit diagnostics; tests for nested symlinks and malicious paths.

- **Sitecustomize deprecation path**
  - Add a CI matrix leg with `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=0`; document and communicate the removal timeline.

- **Documentation refresh**
  - Update CONTRIBUTING/AGENTS with a regeneration checklist (builder → artifacts → pyrefly → pre-commit), plugin/stub guides, policy config, observability/trend outputs, and drift previews.

## Impact
- **Specs affected**: Developer Tooling, Documentation Automation, Type System Compatibility, Observability, CLI & Developer Experience, Security.
- **Code touched**: `tools/docstring_builder/{docfacts.py,cli.py,policy.py,render.py,semantics.py}`, new DocFacts JSON Schema, drift preview generation, parallelism and JSON output, plugins, tests, security/path helpers, `src/sitecustomize.py`, documentation (CONTRIBUTING, AGENTS, README), editor tasks, and CI scripts.
- **Artifacts produced**: Versioned DocFacts with richer metadata + schema, consolidated docstrings drift preview, trend observability JSON, regenerated docstrings, updated policy reports, refreshed contributor docs, and optional PEP‑561 stub packages.
