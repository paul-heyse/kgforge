## Context
Our documentation and linting pipeline is assembled from several independent tools: `docstring_parser` (third-party), our custom `sitecustomize.py` shim, `tools/docstring_builder`, navmap/test-map generators, schema exporters, and observability scanners. Each grew organically, so they do not agree on typing, command-line interfaces, or regeneration workflows.

### Current pain points (observable today)
1. **Type-checker friction.** `sitecustomize.py` monkey patches `docstring_parser.common.Docstring` by adding properties (`attrs`, `yields`, `many_yields`, `size`). The upstream library ships no typing information for these properties. Under mypy's strict mode we assign a `(self) -> list[object]` property onto a class whose stub does not contain such attributes; this triggers `attr-defined`, `name-defined`, and `valid-type` errors. Contributors must silence mypy manually, which defeats the purpose of static analysis.
2. **Docstring-builder CLI mismatch.** We recently rewrote the docstring-builder CLI to expose `update|check|list` subcommands. However, the pre-commit hook still invokes `python -m tools.docstring_builder --diff --since <rev>`. The CLI rejects these flags, so the hook exits with usage errors even when the generated docstrings are correct.
3. **Artefact regeneration sprawl.** Updating documentation artefacts requires running four scripts (`tools/navmap/build_navmap.py`, `tools/docs/build_test_map.py`, `tools/docs/scan_observability.py`, `tools/docs/export_schemas.py`) plus `make docstrings`. Forgetting any step leaves the tree dirty. Junior engineers struggle to remember the sequence and often spend hours chasing unexpected diffs.
4. **Metadata gaps.** Our enhanced docstring builder now records `display_name` for `*args/**kwargs`, yet DocFacts JSON does not expose this, so downstream tools cannot rely on it. Missing metadata also means docstrings may drift even after successful regeneration.

Without a cohesive plan, every documentation change becomes a manual exercise, blocking pre-commit and increasing merge friction.

## Goals / Non-Goals
### Goals
- **G1.** Make `sitecustomize.py` mypy-clean without `# type: ignore` directives by providing type-safe abstractions for all monkey-patched attributes.
- **G2.** Ensure the docstring-builder CLI remains backwards compatible with legacy hook flags so pre-commit works out-of-the-box.
- **G3.** Capture and persist richer metadata (parameter kinds, display names, docfacts) to keep generated docstrings deterministic across workstations.
- **G4.** Deliver a single “one-stop” command that regenerates every documentation artefact with consistent logging and failure semantics.
- **G5.** Document the workflow and encode it in CI so contributors learn it once and automation enforces it thereafter.

### Non-Goals
- **N1.** We are not changing the navmap/test-map/schema formats themselves—only the way they are regenerated.
- **N2.** No redesign of the Sphinx documentation site structure; we only coordinate the supporting artefacts.
- **N3.** We do not attempt to replace `docstring_parser`; we continue to patch it conservatively.

## Decisions
### D1. Typed shims for docstring-parser
We will introduce Protocols (`DocstringProto`, `DocstringAttrProto`, `DocstringYieldsProto`) that describe exactly the attributes and return types we depend on. Helper functions (`ensure_docstring_attrs`, `ensure_docstring_yields`, `ensure_docstring_size`) will:
- Check whether the attribute already exists (covers newer library versions).
- Wrap the `setattr` in a try/except and return `True/False` so callers can log what happened.
- Cast any runtime values (`meta` entries) to the Protocol types before returning them. This keeps type inference precise while keeping runtime behaviour identical.
*Rationale:* Using Protocols allows mypy to reason structurally about the types without forcing us to maintain external stub packages.

### D2. Metadata-first docstring builder
Enhance `ParameterHarvest` to store both the raw annotation string and the `inspect.Parameter.kind` enum. Add a helper `parameter_display_name(parameter: ParameterHarvest) -> str` that formats `*args`, `**kwargs`, or `param` depending on the kind. Persist these values into DocFacts JSON. During rendering we reuse the stored display name, ensuring that a second run generates byte-identical docstrings. We also add snapshot tests covering representative signatures (pure positional, keyword-only, varargs, kwargs) to prove idempotence.

### D3. CLI compatibility layer
Extend the CLI parser to accept `--diff`, `--since`, `--module`, mapping them internally to the existing subcommands (`update`, `check`). If an unsupported combination is passed (e.g., `--diff` without `check`), the CLI prints a targeted error suggesting the modern invocation. We also add an `--ignore-missing` flag and default configuration that suppresses ModuleNotFoundError for directories such as `docs/_build/**`. Unit tests simulate the exact command line pre-commit uses to prevent regressions.

### D4. Unified artefact regeneration
Create a new make target `artifacts` that runs the following steps in order, failing fast on the first error:
1. `uv run python tools/docstring_builder/cli.py update --all` (docstrings + DocFacts).
2. `uv run python tools/navmap/build_navmap.py --write site/_build/navmap/navmap.json`.
3. `uv run python tools/docs/build_test_map.py --write docs/_build/test_map.json`.
4. `uv run python tools/docs/scan_observability.py --write docs/_build/observability.json`.
5. `uv run python tools/docs/export_schemas.py`.
Each step logs `[artifacts] <step> OK` or `[artifacts] <step> FAILED: <reason>`. The target becomes the canonical regeneration workflow referenced in documentation and CI.

### D5. Documentation and enforcement
Update CONTRIBUTING.md and AGENTS.md with a section titled “Regenerating documentation artefacts.” Include a table of triggers (e.g., “Changed navmap metadata → run `make artifacts`”) and troubleshooting tips (“If `make artifacts` fails with ModuleNotFoundError _build, run `uv run python -m build` first”, etc.). Add a CI job (`docs-artifacts`) that checks out the branch, runs `make artifacts`, and fails if `git diff` finds uncommitted artefact changes.

## Risks / Trade-offs
- **R1. Protocol rot.** If future versions of `docstring_parser` rename fields, our Protocol might diverge. *Mitigation:* concentrate all attribute access in helper functions and add regression tests that import the shim both with and without the third-party dependency present.
- **R2. Longer regeneration time.** `make artifacts` will run several scripts. *Mitigation:* make the target optional for local iterations but mandatory before merging; optimise by enabling caching where possible.
- **R3. Increased CLI surface.** Mapping legacy flags alongside new subcommands adds complexity. *Mitigation:* isolate the translation layer in a dedicated function with unit tests.
- **R4. DocFacts schema drift.** Adding new metadata fields could break downstream consumers. *Mitigation:* version the DocFacts schema or provide compatibility fields, and communicate changes to dependent teams.

## Migration Plan
1. **Week 1:** Implement the typed shim, add unit tests, and land the mypy-clean version of `sitecustomize.py`.
2. **Week 2:** Upgrade docstring-builder harvesting/rendering plus DocFacts export, regenerate fixtures, and add snapshot tests.
3. **Week 3:** Enhance CLI compatibility and write regression tests. Update pre-commit hook to rely on the new behaviour.
4. **Week 4:** Introduce `make artifacts`, document the workflow, update CONTRIBUTING/AGENTS, and wire the CI job.
5. **Ongoing:** Monitor CI runs and adjust caching or logging based on feedback.

## Open Questions
- Do any downstream scripts parse DocFacts directly? If so, coordinate schema changes and provide migration notes.
- Should `_build` directories be completely excluded from harvesting, or do we prefer to make the builder resilient to missing modules? Need confirmation from the documentation team.
- Would it be useful to add optional parallel execution to `make artifacts` to shorten runtime, or is sequential execution safer for now?
