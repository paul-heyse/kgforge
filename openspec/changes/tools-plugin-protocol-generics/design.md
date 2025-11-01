## Context
The current docstring-builder plugin stack defines a base `DocstringBuilderPlugin` Protocol whose `apply` method accepts and returns a discriminated union (`DocstringPayload`). Stage-specific Protocols (`HarvesterPlugin`, `TransformerPlugin`, `FormatterPlugin`) narrow that signature, which pyrefly flags as an invalid override. To keep the build green we added suppressions that now mask regressions and block stricter static analysis. Built-in plugins and the orchestration manager therefore operate with casts and `Any` escapes that complicate future refactors.

## Goals / Non-Goals
- **Goals**
  - Model plugin stages with precise generics so pyrefly and mypy can validate the full execution pipeline without ignores.
  - Preserve runtime compatibility for existing plugins, including the legacy adapter and default plugin catalogue.
  - Document the stricter API so contributors can migrate confidently.
- **Non-Goals**
  - Redesign plugin discovery/registration mechanisms beyond typing alignments.
  - Introduce new plugin stages or change runtime behaviour of existing plugins.
  - Address remaining suppressions unrelated to plugin typing (handled in other proposals).

## Decisions
1. **Generic base Protocol** — Redefine `DocstringBuilderPlugin` as `Protocol[InputT, OutputT]` with `apply(self, context, payload: InputT) -> OutputT`. Stage-specific Protocols specialise both parameters to the appropriate payload type (`HarvestResult`, `SemanticResult`, `DocstringEdit`).
2. **Typed manager pipelines** — Update plugin management helpers (`_invoke_apply`, `_ensure_plugin_instance`, registration) to be generic in the payload type, removing casts and ensuring ordering remains type-safe.
3. **Legacy adapter translation** — Adjust `LegacyPluginAdapter` to adopt the generic Protocol by advertising identical input/output payload types while still invoking legacy `run` functions. Emit a runtime warning for un-migrated plugins and guarantee structured error logging when wrapping failures.
4. **Stub and documentation refresh** — Sync `stubs/tools/docstring_builder/plugins/*.pyi` and related exports to describe the new signature, providing examples for each stage so external plugins stay type-clean.
5. **Expanded validation harness** — Introduce table-driven pytest coverage, doctest examples, and import-linter assertions that confirm the refactored pipeline preserves layering, logging, and schema alignment without relying on suppressions.

## Risks / Trade-offs
- **Third-party plugin breakage** — Plugins outside the repo must update annotations (and potentially method signatures) to match the new generics. Mitigated by documenting the migration path and providing a temporary compatibility type alias.
- **Manager regression** — Changing helper signatures could introduce runtime bugs if not covered by tests. Mitigate with targeted unit tests and integration smoke tests (`uv run pyrefly check`, `uv run mypy`).
- **Stubs drift** — Failure to update `.pyi` files would reintroduce type mismatches. Mitigate via checklists in tasks and reviewer gates.

## Migration
1. Introduce the generic Protocols and adjust built-in plugins in the same commit to keep the tree bisectable.
2. Update orchestrator, legacy compatibility layers, and built-in plugins to consume the typed payloads end-to-end while emitting structured logs and Problem Details envelopes on failure.
3. Run lint/type/test gates (including the new plugin tests and import-linter check); address regressions before landing.
4. Publish migration guidance for external plugin authors in the dev docs (linked from `openspec/AGENTS.md`).

