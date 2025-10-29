---
title: NavMap Visibility Policy
status: stable
version: 2
---

# NavMap Visibility Policy

# Visibility Policy (NavMap)

This repository treats **public API** as the union of:
1) `__all__` in each module, and
2) `__navmap__["exports"]` (may reference `__all__` but must match set-wise).

**Required per exported symbol**:
- `owner`: team handle (e.g., `@core-search`)
- `stability`: `stable` \| `beta` \| `experimental` \| `deprecated`
- `since`: PEP 440 version string (e.g., `0.8.0`)
- `deprecated_in` (optional): PEP 440; if present, must be **>=** `since`.

**Sections**
- First section **must** be `public-api`.
- Section IDs must be **kebab-case** (e.g., `domain-models`).
- Each section symbol must have a nearby inline anchor: `# [nav:anchor Name]`.

**Links**
- Editor: `vscode://file/<path>:<line>:<col>` (official scheme).
- GitHub: `https://github.com/<org>/<repo>/blob/<sha>/<path>#Lstart-Lend` (permalink with `#L` anchors).

Violations are reported by `tools/navmap/check_navmap.py`.
