## Context
Phase 1 delivered typed shims, local stubs with drift checks, deterministic docstring builder runs (manifest + policy engine + plugins), and pre-commit enforcement. The builder already captures parameter kinds/display names, but we lack tests and enforcement to keep that metadata authoritative, and several docstring-heavy modules still contain placeholder content. The CLI remains flag-heavy, stub governance isn’t surfaced through `doctor`, observability is minimal, and the workflow is under-documented. Phase 2 combines the tooling enhancements and metadata synchronisation so our docstring pipeline becomes authoritative and easy to operate.

## Goals / Non-Goals
-- **Goals**
  1. Validate and enforce harvested metadata within DocFacts while regenerating docstrings for high-impact modules using that authoritative information.
  2. Restructure the docstring-builder CLI into subcommands with deterministic exit codes and configuration precedence.
  3. Surface stub governance through the CLI/CI, and improve observability (metrics JSON, HTML drift previews, editor integrations).
  4. Harden security (path normalisation, avoid unsafe evaluation) and introduce a feature flag + deprecation plan for `sitecustomize` patches.
  5. Update documentation with a docstring regeneration checklist, plugin/stub guides, policy configuration, observability outputs, and troubleshooting with `doctor`.
- **Non-Goals**
  - No changes to navmap/test-map/schema formats—only how we present diffs.
  - No wholesale rewrite of existing docstring content beyond the targeted modules unless required for accuracy.
  - No introduction of external services; observability remains file-based.

## Decisions
1. **Metadata authoritative source**: keep `ParameterHarvest` as the source of truth, add regression tests for `kind`/`display_name`, ensure DocFacts and the renderer consume these fields, and fail tests when fidelity regresses.
2. **Docstring regeneration**: Use the builder to refresh targeted modules; where metadata lacks descriptions, add concise manual text.
3. **CLI subcommands**: Implement shared runner functions, expose subcommands, enforce exit code constants, and add `--config` precedence.
4. **Stub governance**: Embed drift checking in `docstring-builder doctor --stubs`, integrate into CI, and optionally provide PEP-561 packages.
5. **Observability**: Emit JSON metrics, HTML drift previews, and editor shortcuts; provide guidance for reviewers.
6. **Security & sitecustomize**: Normalise paths, remove unsafe evaluation, add regression tests, introduce `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE`, and document the deprecation timeline.
7. **Documentation**: Expand contributor docs with a regeneration checklist, plugin/stub authoring, policy config, observability outputs, and `doctor` troubleshooting.

## Risks / Mitigations
- **Docstring churn**: Regeneration might cause large diffs. *Mitigation*: focus on targeted modules, review diffs carefully, keep DocFacts synced.
- **Metadata regressions**: Missing parameter kinds could slip through without tests. *Mitigation*: add unit tests covering positional-only, keyword-only, varargs, kwargs and wire DocFacts consumers into CI.
- **CLI complexity**: More subcommands require clear documentation. *Mitigation*: provide concise `--help` output and examples in docs.
- **Stub drift noise**: Frequent upstream changes may trigger drift alerts. *Mitigation*: keep expected symbol lists scoped to what we use and provide actionable output.
- **Security hardening regressions**: Strict path checks might break legitimate workflows. *Mitigation*: add regression tests and document expected path patterns.
- **Sitecustomize toggle failures**: Disabling patches may expose hidden issues. *Mitigation*: run the toggle in CI and fix failures before removing patches.

## Migration Plan
1. Add metadata regression tests, ensure DocFacts/renderer leverage existing fields, and wire consumers to fail fast.
2. Regenerate docstrings for the targeted modules; commit docstrings + DocFacts.
3. Restructure the CLI, add subcommands, exit codes, and config precedence; migrate drift checker into `doctor`.
4. Add observability artefacts, editor snippets, and documentation for interpreting them.
5. Normalize/validate paths, audit for unsafe evaluation, and add regression tests.
6. Introduce the `sitecustomize` feature flag/warning and add CI coverage with the flag disabled.
7. Refresh documentation (CONTRIBUTING/AGENTS) with the regeneration checklist, plugin/stub guides, policy config examples, and troubleshooting steps.
8. Validate via mypy, Ruff, CLI E2E tests, pyrefly, pre-commit, DocFacts consumers, and new CI jobs.

## Open Questions
- Do we extend metadata enrichment to return/raise sections in this phase or defer?
- Should HTML drift previews be generated only on CI, or gated behind a CLI flag for local runs?
- How long should the `sitecustomize` deprecation window last before we remove the patches entirely?
