## Context
Phase 1 delivered typed shims, local stubs with drift checks, deterministic docstring builder runs (manifest + policy engine + plugins), pre-commit enforcement, and a modern CLI with subcommands/exit codes/config precedence. The builder captures parameter kinds/display names and emits DocFacts, observability logs, and drift previews.

Phase 2 evolves the system into an authoritative, auditable, and performant pipeline: formal DocFacts schema/versioning with provenance, richer metadata, consolidated drift previews with trend metrics, stronger policy rules, plugin-driven content fidelity, parallel execution with machine-readable outputs, security hardening, and a pragmatic path off `sitecustomize`.

## Goals / Non-Goals
- **Goals**
  1. Formalize DocFacts (JSON Schema, versioning, provenance) and enrich metadata.
  2. Strengthen policy gates (examples, summary mood, dataclass/attrs parity) and expose diagnostics via `doctor`.
  3. Improve performance and outputs (parallel `--jobs`, `--json` result mode, `--baseline` comparisons).
  4. Enhance observability UX: consolidated docstrings drift HTML and trend metrics; PR-reviewable artifacts.
  5. Expand plugins for better content fidelity (dataclass/attrs, exception inference, parameter seeders; optional LLM post-processor behind a flag).
  6. Harden security (path normalization, symlink traversal) and complete the `sitecustomize` deprecation plan with CI coverage.
  7. Update documentation and editor integrations for a streamlined workflow.
- **Non-Goals**
  - No changes to navmap/test-map/schema formats—only how we present diffs.
  - No wholesale rewrite of existing docstring content beyond the targeted modules unless required for accuracy.
  - No introduction of external services; observability remains file-based.

## Decisions
1. **DocFacts schema/versioning**: Create a JSON Schema with a `docfactsVersion` and validate on each run; include provenance (builder version, config hash, commit).
2. **Metadata enrichment**: Keep `ParameterHarvest` authoritative; ensure renderer uses `display_name` and DocFacts include kinds, defaults, decorators, async/generator flags, ownership, and line spans.
3. **Docstring regeneration**: Refresh targeted modules; add concise manual descriptions where metadata is insufficient.
4. **Performance & outputs**: Add `--jobs` for parallel processing, `--json` output for CI, and `--baseline` for stable comparisons; keep stable ordering and deterministic output.
5. **Policy upgrades**: Implement rules for examples, imperative summaries, and dataclass/attrs parity; expose active exceptions and expiries via `doctor --policy`.
6. **Plugins**: Ship dataclass/attrs description plugin, stronger exception inference, and parameter-description seeders; gate optional LLM rewriter behind an explicit flag.
7. **Observability UX**: Add docstrings drift HTML and trend metrics; link previews in the manifest and surface them in CI output.
8. **Security & sitecustomize**: Normalize/validate paths with explicit diagnostics; add CI leg with `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=0` and document the removal plan.
9. **Documentation**: Update CONTRIBUTING/AGENTS with the regeneration checklist, plugin/stub guides, policy config, observability outputs, drift previews, and editor tasks.

## Risks / Mitigations
- **Docstring churn**: Regeneration might cause large diffs. *Mitigation*: focus on targeted modules, review diffs carefully, keep DocFacts synced.
- **Metadata regressions**: Fidelity could slip without tests. *Mitigation*: add unit tests for all parameter kinds and DocFacts parity; CI schema validation.
- **Parallelism flakiness**: Concurrency could mask ordering or caching bugs. *Mitigation*: stable sort, per-file isolation, deterministic outputs; CI stress runs.
- **Schema churn**: Schema updates may break consumers. *Mitigation*: version DocFacts; provide drift preview and migration guidance; guard with validation.
- **Stub drift noise**: Frequent upstream changes may trigger drift alerts. *Mitigation*: keep expected symbol lists scoped to what we use and provide actionable output.
- **Security hardening regressions**: Strict path checks might break legitimate workflows. *Mitigation*: add regression tests and document expected path patterns.
- **Sitecustomize toggle failures**: Disabling patches may expose hidden issues. *Mitigation*: run the toggle in CI and fix failures before removing patches.

## Migration Plan
1. Implement DocFacts schema/versioning + provenance; extend builder output; add schema validation and CI gate.
2. Add metadata regression tests and renderer parity tests; extend DocFacts tests for enriched fields and idempotence.
3. Implement performance and outputs (`--jobs`, `--json`, `--baseline`) with deterministic ordering; add E2E tests.
4. Build docstrings drift HTML and trend metrics; link previews from manifest; update observability.
5. Implement policy rules and `doctor --policy`; document exceptions and expiries.
6. Implement plugins (dataclass/attrs, exception inference, parameter seeders); optionally wire LLM rewriter behind a flag.
7. Security: normalize/validate paths with explicit diagnostics; add symlink traversal tests.
8. Sitecustomize: add CI matrix leg with shim disabled; document deprecation timeline.
9. Regenerate targeted modules; commit docstrings + DocFacts; validate via mypy, Ruff, pyrefly, pre-commit, DocFacts consumers, navmap, and CI jobs.

## Open Questions
- Should signature rendering be enabled by default for public APIs or gated per-package?
- How to default `--baseline` in local vs CI runs (merge-base vs main vs last successful build)?
- Scope and guardrails for optional LLM rewriting (strict opt-in, file allowlist, dry-run reports only?).
- What removal window for `sitecustomize` is acceptable given downstream dependencies?
