# 0001 – Record architecture decisions

Date: 2025-10-26

## Status
Accepted

## Context
We need persistent, reviewable decisions for the architecture (layers, boundaries, invariants).

## Decision
Use ADRs (Markdown files in `docs/architecture/adr/`) to capture decisions.
Use Import Linter to encode layering rules; fail CI on violations.
Use C4 diagrams (PlantUML/Mermaid) to visualize context and containers.

## Consequences
- New contributors and AI agents understand *why* choices were made.
- Changes across layers require updating rules and ADRs to stay aligned.
