## Why
Our documentation build still reports multiple gallery cross-reference warnings (missing
captions for `sphx_glr_gallery_*` pages) and MkDocs repeatedly lists large sections of the
site as “not included in nav”. These warnings point to real usability issues: gallery links
render as broken references, and MkDocs navigation gives no visibility into the generated
API pages.

## What Changes
- Ensure every Sphinx-Gallery example exposes a canonical title or caption so the generated
  `sphx_glr_*` pages can be referenced without warnings.
- Update `docs/gallery/index.rst` and `docs/gallery/sg_execution_times.rst` to link to the
  gallery pages in a warning-free way (e.g., by using `:doc:` entries that match the actual
  generated docnames).
- Decide on a MkDocs navigation strategy: either add curated entries for the AutoAPI pages
  (at least the top-level index pages) or suppress the warnings explicitly with a documented
  rationale.
- Document the steps so future examples automatically comply with the required metadata.

## Impact
- Affected specs: none (content/navigation clean-up only).
- Affected files: `docs/gallery/*.rst`, scripts under `examples/` (metadata headers), and
  `mkdocs.yml` for nav configuration or warning suppression.
- After the change, running `tools/update_docs.sh` should produce zero gallery `ref.ref`
  warnings, and `mkdocs build` should no longer spam the console with “pages not in nav”
  messages unless intentionally suppressed.

