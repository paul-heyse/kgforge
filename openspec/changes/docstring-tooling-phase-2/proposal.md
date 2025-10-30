## Why
Phase 1 hardened the docstring ecosystem: monkey-patch shims now have typings, local stubs exist with CI drift checks, the docstring builder runs deterministically with manifests/policy gates/plugins, and pre-commit stops regressions early. The next phase needs to make the tooling delightful and authoritative. Today we still see:
- Incomplete metadata in DocFacts (parameter kinds/display names) which causes regenerated docstrings to lose fidelity and pushes contributors toward manual edits.
- High-value modules (`src/docling/canonicalizer.py`, `src/download/harvester.py`, `src/embeddings_sparse/bm25.py`, …) still contain placeholder docstrings or drift from runtime signatures.
- A CLI that exposes legacy flag-driven entry points rather than clear subcommands, making it harder for contributors to adopt the workflow.
- Stub governance scripts that aren’t surfaced in the CLI, limited observability (no metrics/diff previews), and missing documentation describing the end-to-end regeneration process.

Phase 2 combines the tooling upgrades and metadata synchronisation into a single initiative so our docstring pipeline becomes authoritative, observable, secure, and easy for junior contributors to follow.

## What Changes
- **Verify builder metadata & DocFacts**
  - Add unit tests proving `ParameterHarvest` already captures `inspect.Parameter.kind`, formatted `display_name`, and defaults, covering positional-only, keyword-only, varargs, and kwargs.
  - Extend DocFacts tests to assert the enriched metadata persists and remains usable by downstream tooling.
  - Ensure the renderer consumes `display_name` and fails tests if signature fidelity regresses.
- **Regenerate authoritative docstrings**
  - Re-run the builder for docstring-heavy modules (starting with `src/docling/canonicalizer.py`, `src/download/harvester.py`, `src/embeddings_sparse/bm25.py`) and replace placeholder text with accurate descriptions (builder heuristics + domain knowledge).
  - Commit regenerated docstrings and DocFacts together; ensure the pipeline is idempotent.
- **Restructure the CLI**
  - Replace flag-driven commands with explicit subcommands (`generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`).
  - Share runner functions, define exit code constants (0 success, 1 policy failure, 2 config error, 3 internal error), and add `--config` with documented precedence.
- **Surface stub governance**
  - Expose the drift checker via `docstring-builder doctor --stubs`; integrate it into CI with actionable diff output.
  - (Optional) Package stubs as PEP-561 extras and document maintenance steps.
- **Observability & developer experience**
  - Emit `docs/_build/observability_docstrings.json` with counts, timings, cache hits/misses, top errors.
  - Generate HTML drift previews (docfacts/docstrings/navmap) under `docs/_build/drift/` for reviewers; ship editor tasks/snippets for frequent commands.
- **Security hardening**
  - Normalise/validate input paths, reject traversal/symlink exploits, and remove unsafe evaluation paths.
  - Add regression tests covering malicious inputs.
- **Sitecustomize deprecation path**
  - Introduce `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE` flag (default `1`), emit `DeprecationWarning`, and run CI with patches disabled to guarantee the pipeline works without them.
- **Documentation refresh**
  - Update CONTRIBUTING/AGENTS with a “docstring regeneration checklist” (builder → artifacts → pyrefly/pre-commit), examples, and troubleshooting using `doctor`.
  - Cover plugin authoring, stub maintenance, policy configs, observability outputs, and HTML drift previews.

## Impact
- **Specs affected**: Developer Tooling, Documentation Automation, Type System Compatibility, Observability, CLI & Developer Experience, Security.
- **Code touched**: `tools/docstring_builder/harvest.py`, `render.py`, DocFacts generation, `cli.py`, config utilities, stub drift checker, security/path helpers, `src/sitecustomize.py`, documentation (CONTRIBUTING, AGENTS, README), editor task definitions, and CI scripts.
- **Artifacts produced**: Updated DocFacts with parameter kinds/display names, regenerated docstrings for targeted modules, `docs/_build/observability_docstrings.json`, HTML drift previews, refreshed contributor docs, and optional PEP‑561 stub packages.
