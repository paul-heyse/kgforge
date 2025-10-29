## Why

Docstring quality is currently enforced by pydoclint, numpydoc, and interrogate, but our generation tooling cannot keep pace with the strict rules or the richness we want. `tools/generate_docstrings.py` + `tools/auto_docstrings.py` produce skeletal text that often diverges from signatures, omits Raises, and requires manual curation. We also want documentation that serves two audiences equally well: humans and AI agents. That means high-fidelity structure for linters and Sphinx, plus richer, synthesized context (constraints, shapes/dtypes, behavior summaries, examples) that stays in sync with the code.

We need a robust, largely automated pipeline that:
- Harvests authoritative API metadata with Griffe.
- Uses doq + Jinja2 templates for bulk, consistent skeleton creation where docstrings are missing.
- Synthesizes richer narrative content and structured sections from static analysis, optional dynamic probes, and project metadata.
- Applies edits safely and idempotently, and validates with multiple gates.

## What Changes

- Introduce a new `tools/docstring_builder` package that orchestrates:
  - Griffe for program analysis (signatures, annotations, docstrings, locations, decorators) as the metadata backbone.
  - doq with custom Jinja2 NumPy templates for one-time skeleton creation of missing docstrings (Parameters/Returns/Raises/Examples/Notes/See Also pre-populated).
  - A synthesis layer that enriches content:
    - Static analysis: exceptions raised, generator semantics, common shapes/dtypes (based on numpy.typing), optionality/defaults.
    - Optional dynamic probes (dev-only): tiny harnesses to observe typical return shapes or confirm raised exceptions in safe, hermetic scenarios.
    - Pydantic enrichment via autodoc_pydantic and/or griffe-pydantic for field constraints and validators.
    - Cross-references from navmap metadata (owner/stability) and Sphinx intersphinx targets.
  - A renderer that merges curated prose with generated sections using Jinja2 templates; it preserves human edits, manages only owned blocks, and inserts TODOs for unresolved details.
- Provide a sidecar machine-readable export (DocFacts) as JSON per module/class/function capturing synthesized facts (param defaults, inferred constraints, raises, complexity markers, shapes). This enables AI agents to consume consistent, structured signals without parsing free text.
- Ship a CLI (`python -m tools.docstring_builder update|check|list`) powering `make docstrings` and pre-commit in changed-file mode; supports incremental caches and targeting by module.
- Tighten Sphinx build hygiene: nitpicky mode, robust intersphinx, and sphinxcontrib-bibtex for citations; keep our AutoAPI setup and integrate type rendering as needed.

## Impact

- Touches documentation automation and CI gates across the repo: builder package, make targets, pre-commit, Sphinx config.
- Increases documentation richness while reducing manual edits; missing sections are generated, and existing prose is preserved.
- Adds a sidecar JSON (DocFacts) that AI agents and internal tools can leverage, improving downstream automation beyond human-facing HTML.

## Risks / Mitigations

- **Over-generation noise**: The builder only manages owned blocks; editors can opt-out at symbol or module level. Rendered TODO markers make missing details explicit.
- **Runtime probes brittleness**: Keep probes optional and opt-in, scoped to trivial inputs, and never run in CI. Prefer static analysis first.
- **Tool conflicts**: Clearly separate roles—Griffe for metadata, doq/Jinja for skeletons, our renderer for synchronization—and validate with pydoclint, numpydoc validation, rst checks, and doctests.

## Success Metrics

- New or missing docstrings are auto-created with complete NumPy sections and pass pydoclint/numpydoc checks.
- Synchronizer is idempotent (second run → no diff) and fast on changed files.
- Sidecar DocFacts JSON is produced for all managed symbols and consumed by at least one downstream tool (README generator or navmap checks).
