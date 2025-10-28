---
title: NavMap Visibility Policy
status: draft
version: 1
---

# NavMap Visibility Policy

This document captures the normative rules that govern the NavMap metadata
toolchain. The companion machine-parsable schema lives at
`docs/policies/visibility-policy.json`.

The policy applies to every Python module under `src/` that declares public
exports.

## 1. Module Surface

- `__all__` is the canonical list of exported symbols.
- `__navmap__["exports"]` MUST be present and equal to `set(__all__)`.
- Each module MAY provide module-level metadata fields (`owner`, `stability`,
  `since`, `deprecated_in`). When present they act as defaults for symbols that
  omit the same keys.

## 2. Sections

- `__navmap__["sections"]` lists ordered groups of symbols by visibility. Each
  section is an object `{"id": "<kebab-case>", "symbols": ["Name", ...]}`.
- The first section MUST be `public-api`.
- Section ids MUST match `^[a-z0-9]+(?:-[a-z0-9]+)*$`.

## 3. Symbols

- Every exported symbol MUST appear in at least one section.
- For every symbol in a section an inline comment `# [nav:anchor Symbol]` MUST
  exist adjacent to the definition line.
- Symbol names MUST be valid Python identifiers (`^[A-Za-z_]\w*$`).
- Each symbol MUST define:
  - `stability` ∈ {`stable`, `beta`, `experimental`, `deprecated`}
  - `owner`: non-empty string (team handle preferred)
  - `since`: PEP 440 compliant version string
  - `deprecated_in`: optional PEP 440 compliant version string with
    `deprecated_in >= since` when provided

## 4. Links

- Editor links use the VS Code `vscode://file/<path>` scheme.
- GitHub permalinks MUST target a fixed commit (`blob/<sha>#Lstart-Lend`).
- Link mode is configured via `DOCS_LINK_MODE` (`editor`, `github`, or `both`).

## 5. Anchors and Sections in Source

- Section markers in source take the form `# [nav:section slug]` and MUST be
  present for every section listed in `__navmap__`.
- Anchor and section line numbers stored in `navmap.json` MUST match the source
  of truth in the module.

## 6. Enforcement

- `tools/navmap/build_navmap.py` emits the policy version, link mode, module
  defaults, and per-symbol metadata.
- `tools/navmap/check_navmap.py` validates this policy, including PEP 440
  version checks.
- `tools/navmap/repair_navmaps.py` can auto-insert missing anchors, sections,
  and minimal `__navmap__` scaffolding.
- Round-trip verification ensures source annotations and JSON stay in sync.

## 7. Change Management

- Policy updates MUST bump the `version` front-matter field.
- Tooling reads `visibility-policy.json`; update both documents atomically.

