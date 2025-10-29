## 1. Planning & Scaffolding
- [ ] 1.1 Create `tools/docstring_builder/` package with `__init__.py`, modules (`harvest.py`, `semantics.py`, `schema.py`, `render.py`, `apply.py`, `cli.py`, `config.py`, `cache.py`, `docfacts.py`).
- [ ] 1.2 Add `docstring_builder.toml` with include/exclude globs, per-package summary verbs, ownership markers, opt-outs, and optional dynamic probe settings.
- [ ] 1.3 Update `README-AUTOMATED-DOCUMENTATION.md` to reflect the Griffe/doq/Jinja pipeline and the optional DocFacts export.

## 2. Harvest Phase (Program model)
- [ ] 2.1 Implement `harvest.py` using Griffe to collect modules/classes/functions/methods, signatures, annotations, docstrings, decorators, and source locations.
- [ ] 2.2 Parse the file with `libcst` only for precise source edits (quote style/indent preservation). Maintain a map from Griffe objects to CST nodes.
- [ ] 2.3 Emit `SymbolHarvest` (qname, parameters with kinds/defaults, return annotation, existing docstring text, ownership status, file/line/col).
- [ ] 2.4 Tests covering positional-only args, `*args/**kwargs`, dataclasses, properties, async generators, and absent docstrings.

## 3. Semantic Analysis & Synthesis
- [ ] 3.1 Resolve annotations via Griffe (fallback to `typing.get_type_hints` if needed); handle postponed evaluation.
- [ ] 3.2 Infer optionality, unions, container types, generator/async semantics, and None returns.
- [ ] 3.3 Scan bodies (AST) for `raise`/`raise from`, collect exceptions; classify as public/internal.
- [ ] 3.4 Synthesize constraints: common shapes/dtypes (numpy.typing), parameter invariants (simple range checks), and complexity hints where easy to infer.
- [ ] 3.5 Optional: minimal dynamic probe adapters (dev-only) to confirm return shapes or exceptions for safe functions; off by default.
- [ ] 3.6 Tests for unions, generators, numpy.typing patterns, and exception detection.

## 4. Docstring Schema & Rendering
- [ ] 4.1 Define schema dataclasses for NumPy sections (Summary, Extended, Parameters, Returns/Yields, Raises, Notes, See Also, Examples).
- [ ] 4.2 Provide Jinja2 NumPy templates (enforce ordering, punctuation, optional markers, default wording). Offer specialty snippets for Protocols, Exceptions, Dataclasses.
- [ ] 4.3 Merge curated prose with generated sections; manage owned blocks only; insert TODOs when info is missing.
- [ ] 4.4 Golden-file renderer tests for representative symbols; verify pydoclint/numpydoc acceptance on samples.

## 5. Mutation & Idempotence
- [ ] 5.1 Implement `apply.py` to emit `libcst` edits preserving indentation/quotes and nearby comments.
- [ ] 5.2 Mark owned blocks (e.g., `<!-- auto:docstring-builder v1 -->`) so re-runs are predictable.
- [ ] 5.3 Keep manual prose intact unless explicitly opted-in.
- [ ] 5.4 Idempotence tests: double-run produces no diff.

## 6. CLI & Integration
- [ ] 6.1 Implement `cli.py` with `update|check|list|clear-cache` and `--module/--since` filters.
- [ ] 6.2 Incremental cache keyed by file mtime + config hash.
- [ ] 6.3 Update `tools/generate_docstrings.py` to call the new CLI (retain logging); keep doq only for one-time skeleton creation.
- [ ] 6.4 Replace `make docstrings` and pre-commit docstring step with builder (`--check` in hooks; changed-files mode).
- [ ] 6.5 Document usage, overrides, and recovery in `README-AUTOMATED-DOCUMENTATION.md`.

## 7. Configuration & Metadata Hooks
- [ ] 7.1 Parse `docstring_builder.toml` (include/exclude, overrides, toggles) and expose to renderer/synthesizer.
- [ ] 7.2 Integrate navmap metadata (owner/stability) and bibliography (`sphinxcontrib-bibtex`) references via See Also/References.
- [ ] 7.3 Escape hatches: per-symbol disable, preserve sections, read-only analysis mode.
- [ ] 7.4 Document precedence/defaults and examples.

## 8. Migration & Cleanup
- [ ] 8.1 One-time skeleton fill: run doq with custom Jinja templates where docstrings are missing.
- [ ] 8.2 Run builder across `src/` and `tools/`; resolve TODOs via config overrides or targeted edits.
- [ ] 8.3 Produce DocFacts JSON; wire into README/navmap checks as a data source.
- [ ] 8.4 Remove obsolete `tools/auto_docstrings.py` and vendor templates when no longer used directly.
- [ ] 8.5 Metrics: pydoclint violations, interrogate coverage, runtime.

## 9. Testing & Enforcement
- [ ] 9.1 Unit tests for harvest/semantics/renderer/apply + CLI smoke tests.
- [ ] 9.2 CI: builder `--check`, pydoclint, numpydoc validation, flake8-rst-docstrings/rstcheck, doctests; separate Sphinx linkcheck job; enable build caching.
- [ ] 9.3 Remove `src/sitecustomize.py` shim once we pin a compatible `docstring_parser` or adjust builder to avoid the import.
- [ ] 9.4 Tighten pydoclint gate post-migration.

## 10. Sign-off
- [ ] 10.1 Demo to maintainers/teams; capture feedback on AI-agent usability (DocFacts utility, examples richness).
- [ ] 10.2 Archive change; file follow-ups for optional probes/LLM-aided summarization if desired.
