## 1. Discovery & Planning
- [x] 1.1 Run `tools/update_docs.sh` to capture the current gallery (`ref.ref`) warnings and
      copy them into a scratch document (docname + warning text).
- [x] 1.2 Inspect the generated gallery pages under `docs/_build/html/gallery/` to see which
      titles/captions are missing; note any script lacking the proper metadata headers.
- [x] 1.3 Review `mkdocs.yml` and the log output from `mkdocs build` to understand which
      directories trigger the “not included in nav” warning.

## 2. Implementation
- [x] 2.1 Update each example script in `examples/` to include a Sphinx-Gallery-compatible
      metadata block (Title/Tags/Time/GPU/Network). Ensure the title is unique and renders as
      the page heading.
- [x] 2.2 Adjust `docs/gallery/index.rst` and `docs/gallery/sg_execution_times.rst` so they
      reference the generated pages using known docnames (e.g., `gallery/00_quickstart`).
- [x] 2.3 Decide on the MkDocs navigation approach:
      - Option A: add curated entries (top-level AutoAPI indexes, gallery landing page) to
        `mkdocs.yml` so the documentation becomes discoverable from the nav.
      - Option B: if we intentionally keep those sections out of nav, set a documented config
        to suppress the warning.
- [x] 2.4 Update `mkdocs.yml` and any supporting scripts (e.g., gallery validation) to reflect
      the chosen approach.

## 3. Validation
- [x] 3.1 Run `tools/update_docs.sh` and confirm the gallery warnings disappear.
- [x] 3.2 Run `mkdocs build` manually to ensure the nav warnings are gone (or explicitly
      suppressed with a comment explaining why).
- [x] 3.3 Spot-check the rendered gallery index and a couple of example pages to confirm the
      titles/captions display correctly and the links work.

