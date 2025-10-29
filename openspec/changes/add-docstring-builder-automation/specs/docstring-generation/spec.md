## ADDED Requirements

### Requirement: Deterministic, Griffe-Backed Docstring Synchronization
A builder CLI SHALL synchronize NumPy-style docstrings for auto-managed symbols using Griffe metadata and repository templates, producing idempotent results.

#### Scenario: Harvested symbols feed structured schema
- **WHEN** `python -m tools.docstring_builder update` runs against a module
- **THEN** the tool uses Griffe to harvest functions/classes/methods with signatures, annotations, decorators, locations, and existing docstrings; it maps them to CST spans for safe, minimal edits

#### Scenario: Idempotent rewrites
- **WHEN** the builder is executed twice without intervening code changes
- **THEN** the second run exits with zero modifications (no git diff) because owned blocks and formatting are preserved

#### Scenario: Ownership markers respected
- **WHEN** a docstring lacks the auto-generated marker or is explicitly opted out in configuration
- **THEN** the builder leaves the docstring untouched and reports the skip in verbose output

#### Scenario: CLI supports update and check modes
- **WHEN** developers invoke `python -m tools.docstring_builder check`
- **THEN** the tool analyses files, reports discrepancies, and exits non-zero without writing changes so pre-commit/CI can gate merges

### Requirement: Rich Synthesis For Humans And AI Agents
Generated docstrings MUST include canonical NumPy sections and synthesized details (constraints, shapes/dtypes, raises) while remaining valid for pydoclint, numpydoc, and doctests.

#### Scenario: Template-driven skeletons for missing docs
- **WHEN** a callable lacks a docstring
- **THEN** a doq+Jinja2 NumPy template scaffolds Summary, Parameters, Returns/Yields, Raises, Notes, See Also, and Examples with placeholders

#### Scenario: Parameters align with annotations
- **WHEN** parameters are annotated (including Optional/Union/defaults)
- **THEN** the `Parameters` section lists name, resolved type, optional marker, and default text with NumPy conventions (", optional" and "by default …")

#### Scenario: Returns and yields documented
- **WHEN** a function returns a value or yields
- **THEN** include `Returns` or `Yields` with inferred types and concise descriptions; omit only when returning `None`

#### Scenario: Raises derived from code
- **WHEN** code contains `raise`/`raise from`
- **THEN** the `Raises` section lists normalized exception names with brief conditions, unless suppressed by config

#### Scenario: Shapes/dtypes captured
- **WHEN** numpy.typing annotations or simple analysis suggests shapes/dtypes
- **THEN** add shape/dtype notes and examples clarifying expectations

### Requirement: Sidecar Machine-Readable DocFacts
The builder SHALL output a DocFacts JSON describing each managed symbol (parameters with defaults, resolved types, optionality, inferred constraints, raises, yields, simple complexity, and example links).

#### Scenario: DocFacts export produced
- **WHEN** the builder runs in update or check mode
- **THEN** a `docs/_build/docfacts.json` (or per-module JSON) is generated/updated for downstream tools and AI agents

#### Scenario: README/nav integration
- **WHEN** README generation runs
- **THEN** it may incorporate DocFacts metrics or links to examples/notes to keep READMEs synchronized

### Requirement: Workflow Integration & Enforcement
The new builder SHALL replace legacy generators, integrate with pre-commit/CI, and keep drift out of PRs.

#### Scenario: Make target delegates to builder
- **WHEN** contributors run `make docstrings`
- **THEN** the make target executes `python -m tools.docstring_builder update` and streams progress

#### Scenario: Pre-commit/CI gates
- **WHEN** pre-commit/CI runs
- **THEN** the pipeline invokes builder `--check`, pydoclint, numpydoc validation, flake8-rst-docstrings/rstcheck, and doctests; Sphinx linkcheck runs as a separate CI job with caching

#### Scenario: Legacy scripts retired
- **WHEN** migration completes
- **THEN** `tools/auto_docstrings.py` and legacy doq templates are removed or archived; docs point to the new workflow
