## Why
The docstring builder toolchain has accrued ad-hoc typing, logging, and subprocess handling that now block strict Ruff/Mypy adoption and risk silent corruption in DocFacts outputs. Key modules (`tools/docstring_builder/normalizer.py`, `policy.py`, `render.py`, and bundled plugins) still move `dict[str, Any]` payloads around, catch `Exception`, and rely on `print` statements, making it impossible to guarantee schema fidelity or trace failures. With DocFacts 2.0 rolling out and junior developers tasked with the implementation, we need explicit instructions, typed models, deterministic rendering, and first-class observability so downstream consumers (navmap, agent catalog, README generators) can trust the artifacts.

## What Changes
- [x] **ADDED**: Baseline typed contracts in `tools/docstring_builder/models.py` covering DocFacts entries, Docstring IR, CLI JSON envelopes, Problem Details, and the builder’s exception taxonomy, plus the Draft 2020-12 CLI schema at `schema/tools/docstring_builder_cli.json`.
- [x] **MODIFIED**: Documentation and implementation plan now reference these contracts as the authoritative shapes for the upcoming refactor, including the rollout strategy, testing plan, and migration notes.
- [ ] **REMOVED**: Legacy dynamic dictionaries, blind exception handling, and `print` statements (to be replaced module-by-module using the typed contracts).
- [ ] **RENAMED**: _None._
- [ ] **BREAKING**: No public API break expected; CLI JSON response will add a versioned envelope while legacy keys remain during the feature-flag period.

- **Affected specs (capabilities):** `docstring-tooling`
- **Affected code paths:**
  - `tools/docstring_builder/models.py` (new canonical typed definitions)
  - `tools/docstring_builder/normalizer.py`, `/policy.py`, `/render.py`, `/cli.py` (to migrate from `Any` to the typed IR/CLI envelopes)
  - `tools/docstring_builder/plugins/` package (Protocol enforcement, structured errors)
  - `tools/_shared/logging.py`, `tools/_shared/proc.py` (planned shared infrastructure)
  - `tools/docstring_builder/docfacts.py`, `/ir.py` (bridge to typed DocFacts + schema validation)
  - CLI manifest/observability writers (`docstring_builder/cli.py` downstream consumers)
- **Data contracts:**
  - `docs/_build/schema_docfacts.json` (existing DocFacts schema remains source of truth)
  - `schema/tools/docstring_builder_cli.json` (new CLI machine-output schema)
  - Future plugin metadata schema under `schema/tools/docstring_builder_plugin.json`
- **Dependencies:**
  - `jsonschema` runtime validation support (available via existing tooling utilities)
  - Access to CI artifacts for DocFacts shadow validation
  - Coordination with DocFacts 2.0 maintainers for schema/version alignment
- **Rollout plan:**
  1. Treat `tools/docstring_builder/models.py` + `schema/tools/docstring_builder_cli.json` as canonical contracts (complete); add validators that load these definitions at runtime behind the `DOCSTRINGS_TYPED_IR=1` feature flag.
  2. Run the builder in shadow mode on CI (`docstring_builder --json --validate-only`) comparing legacy payloads to the typed/validated pipeline.
  3. Enable the typed pipeline by default while retaining compatibility shims and deprecation warnings for one minor release.
  4. Remove the shim once downstream plugins confirm adoption and schema parity remains stable.

## Acceptance
- [ ] Scenarios under `specs/docstring-tooling/spec.md` pass as written (verified by automated tests and manual CLI dry runs).
- [ ] Quality gates all green (`uv run ruff format && uv run ruff check --fix`, `uv run pyrefly check`, `uv run mypy --config-file mypy.ini`, `uv run pytest -q`, `make artifacts && git diff --exit-code`).
- [ ] DocFacts payloads and CLI JSON outputs validate against their schemas; failure paths emit Problem Details JSON logged at ERROR level.
- [ ] Structured logs, metrics, and trace spans demonstrated via tests or recorded runbook snippets.
- [ ] Developer handoff notes (checklist + migration guide) published for plugin authors.

## Out of Scope
- Regenerating docstrings for the full repository (tracked under DocFacts 2.0 rollout).
- Replacing downstream consumers (navmap, README generator) beyond required type/schema updates.
- Introducing new LLM-backed summarizers beyond the existing optional plugin.
- Reworking non-docstring tooling CLIs (e.g., navmap, agent catalog) beyond shared utility adoption.

## Risks / Mitigations
- **Risk:** Plugin authors fail to update to the new Protocol and TypedDict contracts.
  - **Mitigation:** Provide migration guide, lint fixer, and compatibility shims with deprecation warnings for one minor release.
- **Risk:** Schema validation could surface latent data inconsistencies that break CI.
  - **Mitigation:** Shadow validation with `--json --validate-only` mode in CI before flipping defaults; add targeted fixtures to cover known oddities.
- **Risk:** Enforcing autoescape in Jinja renderer might change output formatting.
  - **Mitigation:** Add golden-file tests and diff previews; expose escape policy configuration with safe defaults.
- **Risk:** Junior developers may struggle to locate required schemas or understand rollout toggles.
  - **Mitigation:** Document file paths, feature flags, and step-by-step validation commands in tasks/design; pair with onboarding checklist.

## Assumptions / Prerequisites
- Developers have run `scripts/bootstrap.sh` and can execute `uv` commands locally.
- Existing DocFacts schema (`docs/_build/schema_docfacts.json`) reflects the latest agreed contract.
- Access to CI pipelines capable of running the builder in validation-only mode.


## Alternatives Considered
- **Alt A — Incremental lint fixes only:** Rejected because it does not establish typed contracts or shared infrastructure, leaving systemic risk untouched.
- **Alt B — Full rewrite of docstring builder in Rust:** Rejected for scope/time; incremental hardening within Python meets quality goals while leveraging existing investment.

