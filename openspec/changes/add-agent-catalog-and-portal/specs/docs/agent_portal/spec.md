## ADDED Requirements

### Requirement: Agent Catalog JSON v1.0
The system SHALL generate a single, machine-readable catalog at `docs/_build/agent_catalog.json` that enumerates and links all documentation artifacts for agent consumption.

- The catalog SHALL include top-level fields: `version`, `generated_at` (ISO-8601 UTC), `repo.sha` (short SHA), and `link_policy`.
- The catalog SHALL define `artifacts` pointers to: DocFacts, NavMap, Symbols index, Sphinx `by_file`/`by_module`, Schemas (docstrings + agent_catalog), Graphs, Observability, and Search index.
- The catalog SHALL list `packages[*].modules[*]` with `module`, `paths` (`source`, `page`, `fjson`), `anchors` (open-in-editor, open-in-github), and `docfacts` (symbol qnames present for the module).
- The catalog SHALL use relative paths that are valid within the repo checkout and the built site; absolute URLs MAY be included when GitHub mode is active.
- The generator SHALL emit a JSON Schema at `docs/_build/schema_agent_catalog.json` and validate the catalog before writing.

#### Scenario: Successful generation
- WHEN required inputs exist (DocFacts, Sphinx indices)
- THEN `docs/_build/agent_catalog.json` is written with `version`, `generated_at`, `repo.sha`, `link_policy`, `artifacts`, and at least one `packages[*].modules[*]`
- AND the catalog validates against `schema_agent_catalog.json`

#### Scenario: Missing critical inputs
- WHEN DocFacts or Sphinx indices are missing or unreadable
- THEN generation fails with a clear error message naming the missing artifact
- AND the process exits non-zero without writing a partial catalog

#### Scenario: Link policy editor mode
- GIVEN `DOCS_LINK_MODE=editor`
- WHEN the catalog is generated
- THEN `link_policy.mode` is `editor` and `anchors.open_in_editor` are populated with `vscode://file/{path}:{line}` links

#### Scenario: Link policy GitHub mode
- GIVEN `DOCS_LINK_MODE=github` and `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`
- WHEN the catalog is generated
- THEN `link_policy.mode` is `github` and `anchors.open_in_github` resolve to `https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}`


### Requirement: Agent Portal HTML
The system SHALL render a static HTML portal at `site/_build/agent/index.html` that consumes the Agent Catalog to present a unified entry-point for humans and agents.

- The portal SHALL render quick links to global artifacts (DocFacts, NavMap, Schemas, Graphs, Observability).
- The portal SHALL display packages and modules as navigable cards with deep links to rendered pages and source anchors.
- The portal SHALL include a single search box that filters modules and symbols client-side using the combined indices from the catalog.
- The portal SHALL be viewable offline and SHALL NOT require any external build tooling.

#### Scenario: Portal renders from catalog
- WHEN `docs/_build/agent_catalog.json` exists and is valid
- THEN `site/_build/agent/index.html` is produced
- AND opening the page shows header, quick links, and at least one package/module card

#### Scenario: Client-side search works
- WHEN a user types a module or symbol substring into the search box
- THEN the list filters to matching modules/symbols without a full page reload

#### Scenario: Fallback when catalog missing
- WHEN the catalog is missing
- THEN portal generation fails with a clear error message and exits non-zero


### Requirement: Orchestrator Integration
The documentation artifacts orchestrator SHALL run the Agent Catalog generator and Agent Portal renderer after existing steps (docstrings → navmap → testmap → observability → schemas).

#### Scenario: Agent steps executed
- WHEN `make artifacts` (or `python -m tools.docs.build_artifacts`) runs successfully
- THEN the orchestrator prints status lines for agent catalog and portal
- AND the files `docs/_build/agent_catalog.json` and `site/_build/agent/index.html` are present

#### Scenario: Failure stops pipeline
- WHEN agent catalog validation fails
- THEN the orchestrator stops and returns a non-zero exit code


### Requirement: Deterministic Provenance and Paths
The system SHALL include `repo.sha` and deterministic relative paths for all artifacts and module files.

#### Scenario: Provenance recorded
- WHEN the catalog is generated on any branch or commit
- THEN `repo.sha` equals the current `git rev-parse --short HEAD`
- AND `generated_at` is an ISO-8601 UTC timestamp


### Requirement: Schema Validation
The system SHALL define and emit a JSON Schema for the Agent Catalog and SHALL validate each generated catalog against it.

#### Scenario: Schema present and validation enforced
- WHEN the generator runs
- THEN `docs/_build/schema_agent_catalog.json` is written or updated
- AND the produced catalog validates against it


### Requirement: Link Policy and Environment Overrides
The system SHALL respect `DOCS_LINK_MODE` and GitHub environment variables to compute deep links, and SHALL fallback gracefully when variables are absent.

#### Scenario: Editor mode without GitHub variables
- GIVEN no GitHub variables are set
- WHEN the generator runs with `DOCS_LINK_MODE=editor`
- THEN editor links are populated and GitHub links MAY be omitted

#### Scenario: GitHub mode with missing variables
- GIVEN `DOCS_LINK_MODE=github` but `DOCS_GITHUB_ORG` or `DOCS_GITHUB_REPO` is missing
- THEN generation fails with a clear configuration error and exits non-zero


### Requirement: Accessibility and Minimal Dependencies
The portal SHALL be accessible with basic keyboard navigation and SHALL render core content without JavaScript enabled.

#### Scenario: No-JS fallback
- WHEN JavaScript is disabled in the browser
- THEN the portal shows package and module listings with static links
- AND the search box may be disabled or show guidance

### Requirement: Link Policy Resolution Precedence
The system SHALL resolve link policy using the precedence: CLI > environment > defaults; GitHub mode requires org/repo/sha.

#### Scenario: CLI overrides env
- GIVEN `DOCS_LINK_MODE=editor` in env
- WHEN the CLI requests `--mode github --org X --repo Y --sha Z`
- THEN the catalog is generated in GitHub mode with those values

#### Scenario: Missing GitHub variables
- WHEN `mode=github` but `org`/`repo`/`sha` is missing
- THEN generator exits with configuration error and no catalog is written


### Requirement: agentctl CLI Behaviors
The CLI SHALL implement deterministic commands with stable exit codes.

#### Scenario: list-modules success
- WHEN `agentctl list-modules --package kgfoundry` runs
- THEN exit code is 0 and a table of modules is printed

#### Scenario: search returns JSON
- WHEN `agentctl search --q "vector store" --k 10` runs
- THEN exit code is 0 and a JSON array of results is returned

#### Scenario: configuration error
- WHEN required inputs are missing
- THEN exit code is 2 and an actionable message is printed


### Requirement: Analytics JSON
The system SHALL produce `docs/_build/analytics.json` summarizing portal usage and generation outcomes.

#### Scenario: Analytics present
- WHEN the portal is generated
- THEN `analytics.json` exists with `generated_at`, `portal.sessions`, and `errors.broken_links`


### Requirement: Sharding Thresholds and Root Index
The system SHALL shard catalogs when thresholds are exceeded and provide a root index that references shards.

#### Scenario: Sharding by size or count
- GIVEN catalog size exceeds 20MB or modules > 2000
- WHEN generation runs
- THEN per-package shards are written and root index references them

#### Scenario: Transparent resolution
- WHEN clients load the root index
- THEN they transparently load referenced shards on demand


### Requirement: Roles and Permissions (Hosted Mode)
Hosted deployments SHALL enforce RBAC with `viewer`, `contributor`, and `admin` roles.

#### Scenario: Role denial
- WHEN a viewer attempts to access admin features
- THEN access is denied and logged


### Requirement: Performance Budgets
The portal SHALL meet documented performance budgets on representative hardware.

#### Scenario: Cold-load budget
- WHEN opening the portal locally with a large catalog
- THEN first contentful render occurs in < 1.5s

#### Scenario: Search latency
- WHEN performing a first search
- THEN results return in < 300ms (subsequent < 150ms)


### Requirement: Fuzzy Remap Algorithm for Anchors
The system SHALL remap anchors using a deterministic fallback order.

#### Scenario: Remap order
- WHEN resolving a stale anchor
- THEN the system attempts: (1) match `symbol_id`; (2) name+arity match; (3) nearest-text search; or fails with a clear note


### Requirement: Semantic Index Quality Targets
The system SHALL produce a semantic index that meets minimum quality targets.

#### Scenario: Rerank quality
- WHEN evaluating on an internal benchmark set
- THEN hybrid search must achieve MRR@10 ≥ configured target (e.g., 0.65)


### Requirement: Stable ID Uniqueness
Symbol IDs SHALL be stable across runs and collision-resistant.

#### Scenario: Collision detection
- WHEN generating IDs for all symbols
- THEN collisions are not observed; any collision triggers a generation error with diagnostics


### Requirement: Code Coverage Mapping
The catalog SHALL optionally map symbols to tests and code coverage when available.

### Requirement: Editor-Activated Stdio API (Optional, No Daemon)
The system SHALL expose an optional, file-backed stdio process (MCP or JSON-RPC over stdio) over the catalog; it SHALL be spawned by the editor/agent per session and SHALL terminate at session end.

#### Scenario: Spawn and handshake
- WHEN the editor spawns the stdio process
- THEN the process responds to a `capabilities` call and loads the catalog into memory

#### Scenario: Query methods
- WHEN invoking `find_callers(symbol_id)`, `find_callees(symbol_id)`, `search(q)`, `open_anchor(symbol_id, mode)`, `change_impact(symbol_id)`, `explain_ranking(doc_id)`
- THEN results are returned from the in-memory catalog; repeated calls are faster due to warm cache

#### Scenario: Session termination
- WHEN stdin closes or an explicit shutdown request is sent
- THEN the process exits with code 0 and releases resources; no TCP ports are opened during its lifetime


### Requirement: OpenAPI 3.2 and SDKs
The system SHALL publish an OpenAPI 3.2 description for the API façade and generate Python/TS SDKs; errors SHALL follow RFC9457 Problem Details.

#### Scenario: Spec present
- WHEN the server is built
- THEN `docs/_build/agent_api_openapi.json` exists and validates against OpenAPI 3.2

#### Scenario: Problem Details
- WHEN an error occurs (e.g., missing symbol)
- THEN the response contains `type`, `title`, `status`, and `detail`


### Requirement: Agent Hints
The catalog SHALL include `agent_hints` per module/symbol with intent tags, safe operations, tests to run, performance budgets, and breaking-change notes; the portal SHALL render these on cards.

#### Scenario: Hints rendered
- WHEN a symbol has `agent_hints`
- THEN the portal shows tags, the safe-ops checklist, and tests-to-run actions


### Requirement: Change Impact Shard
The catalog SHALL include (or reference) `change_impact` data per module/symbol; the portal SHALL provide an “Edit here” panel summarizing impact.

#### Scenario: Impact card
- WHEN opening a module page
- THEN the portal shows impacted files, tests to run, owners to ping, and suggested commit skeletons


### Requirement: Exemplars
The catalog SHALL include `exemplars` per symbol with copy-ready snippets, counter-examples, and negative prompts; the portal SHALL expose “Insert exemplar” actions.

#### Scenario: Insert exemplar
- WHEN viewing a symbol with exemplars
- THEN a button copies a selected exemplar snippet to the clipboard


### Requirement: CST Fingerprint and Remap Order
Anchors SHALL include an explicit `remap_order` array; CST token-trigram fingerprinting SHALL be used before nearest-text fallbacks.

#### Scenario: Docstring/formatting churn
- WHEN only docstrings/formatting change
- THEN anchors remap using `symbol_id` or CST fingerprint without manual intervention


### Requirement: Ordering Semantics
Ordered data SHALL be represented as arrays; property order SHALL NOT be relied upon.

#### Scenario: Schema compliance
- WHEN validating catalog objects
- THEN arrays are used for ordered sequences (e.g., `remap_order`, `exemplars`), and validators do not assume object key order
#### Scenario: Coverage present
- WHEN coverage data is provided
- THEN the catalog includes coverage summaries per module/symbol and links to failing tests where applicable
### Requirement: Symbol Graph and Quality Signals in Catalog
The catalog SHALL include per-module and per-symbol graph information and quality signals enabling agents to reason about structure and health.

- Graphs: `graph.imports` (module → module) and `graph.calls` (symbol → symbol) MUST be present when derivable; MAY be empty otherwise.
- Quality: `quality.mypy_status`, `quality.ruff_rules`, `quality.doc_parity_ok`, `quality.coverage_pct`, `quality.doctest_status` MUST be present when inputs exist.

#### Scenario: Graph and quality present
- WHEN inputs (DocFacts, indices, static analysis) are available
- THEN `graph.imports` and `graph.calls` are populated
- AND `quality` fields are populated per symbol/module

#### Scenario: Graceful omission
- WHEN an input is missing (e.g., doctest not run)
- THEN the corresponding `quality` field is omitted or marked `unknown` without failing catalog generation


### Requirement: Stable Anchors and IDs
The catalog SHALL include `symbol_id` (stable content hash of normalized AST) and anchors with line spans, and SHALL support fuzzy remap across minor line drift.

#### Scenario: Anchor survives line drift
- GIVEN a symbol shifts by N lines but its AST is unchanged
- WHEN an agent resolves the anchor using `symbol_id`
- THEN the link target remaps to the correct current lines


### Requirement: Typed Clients and Agent CLI
The system SHALL provide typed clients (Python, TypeScript) and a local `agentctl` CLI to query the catalog.

#### Scenario: Python client query
- WHEN importing the Python client
- THEN `list_modules(pkg)` and `find_callers(symbol_id)` return typed results matching the schema

#### Scenario: agentctl list-modules
- WHEN running `agentctl list-modules --package kgfoundry`
- THEN a table of modules is printed with links and counts


### Requirement: Semantic Index and Hybrid Search
The portal and clients SHALL support hybrid search over lexical indices and a semantic FAISS index when present.

#### Scenario: Lexical fallback
- WHEN the semantic index is absent
- THEN search operates purely on lexical indices

#### Scenario: Vector rerank
- WHEN the semantic index is available
- THEN results are reranked by vector similarity with configurable weights


### Requirement: Faceted Search and Breadcrumbs
The portal SHALL expose facets (package, stability, churn, coverage, parity) and show breadcrumb navigation for context.

#### Scenario: Facet filtering
- WHEN a user selects `package=kgfoundry` and `parity=fail`
- THEN only matching modules/symbols are displayed

#### Scenario: Breadcrumb rendering
- WHEN viewing a deep module path
- THEN breadcrumbs reflect the package/module hierarchy


### Requirement: Usage Examples and Performance Metrics
The portal SHALL display example snippets and performance metrics per module when provided.

#### Scenario: Examples rendered
- WHEN example code is available for a module
- THEN an Examples section is shown with copyable snippets

#### Scenario: Metrics displayed
- WHEN metrics exist (complexity, timing)
- THEN a compact metrics block is shown on the module card


### Requirement: Dependency Visualization
The portal SHALL present dependency graphs inline (SVG) or via links to focused graph views.

#### Scenario: Inline graph
- WHEN a module has import edges
- THEN a small inline SVG is rendered with a link to a larger view


### Requirement: Tutorials and Feedback
The portal SHALL include offline tutorials and an optional local-only feedback mechanism.

#### Scenario: Tutorials accessible
- WHEN opening the portal
- THEN a Tutorials section links to onboarding/playbooks

#### Scenario: Feedback (local-only)
- WHEN a user submits feedback locally
- THEN it is saved to a local JSON file and redacted of sensitive terms


### Requirement: Security and Governance (Optional Hosted Mode)
When hosted, the portal MAY enable RBAC and audit logs; privacy guardrails MUST apply to any collected data.

#### Scenario: RBAC enforcement
- GIVEN hosted mode with roles configured
- WHEN a viewer accesses admin-only features
- THEN access is denied and audited

#### Scenario: Audit trail
- WHEN important actions occur (e.g., feedback submission)
- THEN an audit log entry is recorded


### Requirement: Performance and Caching
The portal and catalog generation SHALL implement content-addressed caching and meet stated load budgets.

#### Scenario: Cached generation
- WHEN inputs are unchanged
- THEN catalog/portal generation reuses cached shards and runs faster

#### Scenario: Load budget
- WHEN opening the portal on a large catalog
- THEN the initial render completes within the documented performance budget


### Requirement: Analytics and Review Annotations
The system SHALL emit analytics JSON for usage and enable CI review summaries for changes.

#### Scenario: Analytics written
- WHEN the portal is generated
- THEN `docs/_build/analytics.json` is written with anonymized usage summaries

#### Scenario: CI annotations
- WHEN a PR changes important documentation surfaces
- THEN the pipeline posts a summary (coverage, parity, high-churn modules) to the PR


### Requirement: Versioning and Sharding
The catalog SHALL use semantic versioning and optionally shard by package with a root index.

#### Scenario: Sharded catalog
- WHEN the catalog exceeds the shard threshold
- THEN a root `agent_catalog.json` references per-package shard files and clients resolve them transparently
