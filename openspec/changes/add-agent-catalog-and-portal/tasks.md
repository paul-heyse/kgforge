## 1. Implementation
- [ ] 1.1 Build Agent Catalog generator
    - [ ] 1.1.1 Create `tools/docs/build_agent_catalog.py` reading existing artifacts:
        - `docs/_build/docfacts.json`, `site/_build/navmap/navmap.json`
        - `site/_build/by_file.json`, `site/_build/by_module.json`, `site/_build/symbols.json`
        - `docs/_build/graphs/*.svg`, `docs/_build/schema_*.json`, `docs/_build/observability*.json`, `site/_build/search/index.json`
    - [ ] 1.1.2 Compute provenance: `git rev-parse --short HEAD` for `repo.sha`; ISO timestamp for `generated_at`; probe tool versions when available.
    - [ ] 1.1.3 Build link policy from env: `DOCS_LINK_MODE=editor|github`, `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`.
    - [ ] 1.1.4 Map packages/modules to paths using Sphinx indices; attach DocFacts symbols per module.
    - [ ] 1.1.5 Write `docs/_build/agent_catalog.json` and emit `docs/_build/schema_agent_catalog.json`.
    - AC:
        - Generator runs locally with no errors and produces both files.
        - Paths and anchors resolve to existing targets (spot check a few entries).

- [ ] 1.2 Render Agent Portal
    - [ ] 1.2.1 Create `tools/docs/render_agent_portal.py` that reads the catalog and writes `site/_build/agent/index.html`.
    - [ ] 1.2.2 Render: header, quick links, search, package/module cards, symbol lists, action buttons (open-in-editor/GitHub).
    - [ ] 1.2.3 Use minimal inline CSS and vanilla JS; no external build.
    - AC:
        - Opening the HTML locally shows populated sections and working links.
        - Search filters module/symbol lists client-side.

- [ ] 1.3 Orchestrator integration
    - [ ] 1.3.1 Update `tools/docs/build_artifacts.py` to run `build_agent_catalog.py` then `render_agent_portal.py` after schemas.
    - [ ] 1.3.2 Print status messages; fail fast if schema validation or IO errors occur.
    - AC:
        - `make artifacts` prints agent steps and produces the new files.

- [ ] 1.4 Documentation (human + agent)
    - [ ] 1.4.1 Add `docs/agent_portal_readme.md` explaining the catalog schema, link modes, and usage for agents and humans.
    - [ ] 1.4.2 Link the readme from the portal page and (optionally) MkDocs nav.
    - AC:
        - Readme builds cleanly; portal links to it.

## 2. Testing & Validation
- [ ] 2.1 Schema validation tests
    - [ ] 2.1.1 Unit test: catalog conforms to `schema_agent_catalog.json`.
    - [ ] 2.1.2 Negative test: missing required keys should fail with a clear message.
- [ ] 2.2 Link sanity tests
    - [ ] 2.2.1 Spot check a sample of module/page/fjson/editor/GitHub links exist or resolve.
- [ ] 2.3 Portal smoke test
    - [ ] 2.3.1 HTML contains quick links and at least one package/module card; search input present.
- [ ] 2.4 Orchestrator test
    - [ ] 2.4.1 `make artifacts` (or `python -m tools.docs.build_artifacts`) completes with agent steps.

## 3. Acceptance
- Agent Catalog JSON present and valid; Portal HTML present and navigable.
- Links open correct targets for both `editor` and `github` modes (when configured).
- Orchestrator integrates both steps without impacting existing artifact generation.

