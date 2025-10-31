# src-hardening-phases-0-2

This change packages the staged hardening plan for the `src/` tree:

- **Phase 0** – establish deterministic quality gates, surface all suppressed
  lints/types, and require the four-item design note before any execution.
- **Phase 2** – eliminate `Any` flows across catalog/search/storage modules,
  enforce schema-backed boundaries, and split high-complexity functions into
  typed units with targeted tests.
- **Phase 3** – run acceptance gates, capture telemetry, and complete the
  rollout with archived execution notes.

See `proposal.md` for the why/what/impact narrative and `tasks.md` for the
sequenced execution checklist. Requirements are captured under
`specs/src-hardening-roadmap/spec.md`.

