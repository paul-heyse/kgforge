# Agent playbook: Enable agent-first documentation

> This playbook assumes Python ≥ 3.11 and a repo that uses either `src/<pkg>` or `<pkg>` at the root.

## 0) Install
```bash
pip install -U pip
pip install -e ".[docs]"   # adds Sphinx stack + docstring tools
```

## 1) Drop these files into the repo root (keeping directories)
- `docs/conf.py`, `docs/index.md`
- `docs/_scripts/build_symbol_index.py`
- `docs/architecture/adr/0001-record-architecture-decisions.md`
- `docs/_static/` (empty keep)
- `tools/gen_readmes.py`, `tools/make_importlinter.py`, `tools/detect_pkg.py`
- `.pre-commit-config.yaml`, `Makefile`
- (optional) `optional/mkdocs.yml`, `docs/_scripts/mkdocs_gen_api.py`

## 2) (Optional) Create an Import Linter configuration
```bash
python tools/make_importlinter.py
```

## 3) Generate/fix docstrings
```bash
make docstrings
```

## 4) Generate package-level READMEs and TOCs
```bash
# Choose link type:
export DOCS_LINK_MODE=editor   # or: github
export DOCS_EDITOR=vscode      # or: pycharm
export DOCS_GITHUB_ORG=your-org DOCS_GITHUB_REPO=your-repo
python tools/gen_readmes.py
# optional TOC update if doctoc is installed
doctoc src/$(python tools/detect_pkg.py) || true
```

## 5) Build docs (human + machine-readable)
```bash
make html
make json
python docs/_scripts/build_symbol_index.py
```

- HTML output in `docs/_build/html`
- JSON corpus in `docs/_build/json`
- Symbol index in `docs/_build/symbols.json`

## 6) Live reload
```bash
make watch
```
Open the local URL; click a symbol's **[source]** or **Open in VS Code** links (from the right-side links on symbol pages).

## 7) Environment toggles (no code changes needed)
- `DOCS_LINK_MODE=editor|github` (default: `editor`)
- `DOCS_EDITOR=vscode|pycharm` (default: `vscode`)
- `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA` if using GitHub link mode
- `SPHINX_THEME=pydata_sphinx_theme|furo|...`

## 8) CI hooks (optional)
Create `.github/workflows/docs.yml` to run `make html json` and (optionally) `lint-imports` and `linkcheck`.

## 9) What agents can index
- `docs/_build/json/**.fjson`, `globalcontext.json`, `searchindex.json`
- `docs/_build/symbols.json`
- ADRs under `docs/architecture/adr/`
- Package-level READMEs under `src/<pkg>/**/README.md`

## 10) Troubleshooting
- If symbols don't link: ensure `griffe` is installed and your package path is detected.
- If editor links don't open: confirm your OS has a handler for the URI scheme (`vscode://` or `pycharm://`).
- If you're not using `src/` layout, the scripts auto-fallback to root.
