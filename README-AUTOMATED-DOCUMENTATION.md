# Automated Documentation System

This repo ships with a fully automated documentation stack that keeps
API references, module docstrings, and package READMEs in sync with the codebase.
This guide explains the moving pieces, how they fit together, and where to look
if you need to adjust behavior.

> **TL;DR**
>
> 1. Install extras: `pip install -e ".[docs]"`
> 2. Regenerate artifacts: `tools/update_docs.sh`
> 3. Pre-commit enforces docstring formatting, coverage, and linting automatically.

---

## Key Configuration Files

| File | Purpose |
| ---- | ------- |
| [`docs/conf.py`](docs/conf.py) | Sphinx configuration. Performs static parsing with AutoAPI, adds editor/GitHub deep links, and builds HTML/JSON outputs. |
| [`tools/gen_readmes.py`](tools/gen_readmes.py) | Generates package-level `README.md` files that list public modules/classes/functions with summaries and source links. |
| [`tools/update_navmaps.py`](tools/update_navmaps.py) | Refreshes module-level docstrings with a `NavMap:` section summarizing public API entry points. |
| [`tools/update_docs.sh`](tools/update_docs.sh) | One-touch wrapper that runs docstring generation, nav-map refresh, README generation, and Sphinx builds. |
| [`Makefile`](Makefile) | Defines developer tasks (`make docstrings`, `make readmes`, `make html/json/symbols`, etc.). |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Pre-commit hooks for Ruff, Black, Mypy, docformatter, pydocstyle, and interrogate. |
| [`Automated Documentation Start.md`](Automated Documentation Start.md) | Original runbook that describes the goals, tooling, and usage patterns. |

---

## What Each File Does

### Sphinx Configuration (`docs/conf.py`)
- Auto-detects the package (`src/<pkg>` or `<pkg>` layout) via `tools/detect_pkg.py`.
- Uses **AutoAPI** to statically analyze the source tree (no imports).
- Adds [linkcode](https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html) hooks with:
  - Editor deep links (`vscode://file/...`) when `DOCS_LINK_MODE=editor` (default).
  - GitHub permalinks (`.../blob/<SHA>/...#Lstart-Lend`) when `DOCS_LINK_MODE=github`.
- Builds HTML and machine-readable JSON (`make html`, `make json`).
- Enables extensions such as `myst_parser`, `sphinx.ext.viewcode`, `sphinx.ext.graphviz`, `sphinx.ext.inheritance_diagram`, and `sphinxcontrib.mermaid`.

### README Generator (`tools/gen_readmes.py`)
- Uses **Griffe** to walk each package and collect public modules/classes/functions.
- Writes `README.md` with:
  - A Doctoc-compatible TOC placeholder.
  - Per-symbol entries (summary + `[open]`/`[view]` links).
  - `DOCS_LINK_MODE` toggles between editor links and GitHub permalinks.
- Re-run manually (`python tools/gen_readmes.py`) or automatically via `make readmes`.

### NavMap Updater (`tools/update_navmaps.py`)
- Called during `make docstrings`.
- Rewrites module docstrings to append a concise `NavMap:` section summarizing public API entry points.
- Summaries are truncated to satisfy Ruff’s 100-character limit.

### Docs Update Script (`tools/update_docs.sh`)
- Run from repo root to perform the full pipeline:
  1. `make docstrings` (doq → nav map → docformatter/pydocstyle/interrogate)
  2. `make readmes`
  3. `make html`, `make json`, `make symbols`
  4. Optional MkDocs build if `mkdocs` is installed
- Gracefully skips MkDocs if not present.

### Makefile Tasks
- `make docstrings`: seeds and formats docstrings, updates NavMaps, runs docformatter/pydocstyle/interrogate.
- `make readmes`: regenerates package READMEs.
- `make html/json/symbols`: Sphinx builders for HTML, JSON corpus, and `docs/_build/symbols.json`.
- `make watch`: runs `sphinx-autobuild` (auto rebuild+serve).
- `make bootstrap`: creates `.kgforge-venv`, installs `pip install -e "[dev,docs]"`, sets up pre-commit.

---

## Library & Extension Overview

| Tool / Library | Role |
|----------------|------|
| [Sphinx](https://www.sphinx-doc.org/) | Core documentation builder. |
| [AutoAPI](https://github.com/readthedocs/sphinx-autoapi) | Static API extraction without importing modules. |
| [MyST-Parser](https://myst-parser.readthedocs.io/) | Markdown support in Sphinx. |
| [Griffe](https://mkdocstrings.github.io/griffe/) | Static analyzer powering nav maps, README links, symbol index, and mkdocstrings integration. |
| [docformatter](https://github.com/PyCQA/docformatter) | Docstring formatter invoked in `make docstrings` and pre-commit. |
| [pydocstyle](https://www.pydocstyle.org/en/stable/) | PEP 257 docstring linting. |
| [interrogate](https://interrogate.readthedocs.io/) | Docstring coverage enforcement (configured for 90%). |
| [doq](https://github.com/heavenshell/py-doq) | Docstring skeleton generator used in `make docstrings`. |
| [MkDocs + mkdocstrings](https://www.mkdocs.org/) | Optional Markdown-first site (configured in `mkdocs.yml`). |
| [Doctoc](https://github.com/thlorenz/doctoc) | Optional CLI that updates README TOCs (invoked by `make readmes` if available). |

### Sphinx Extensions Enabled
- `myst_parser`
- `sphinx.ext.autosummary`
- `sphinx.ext.intersphinx`
- `sphinx.ext.viewcode`
- `sphinx.ext.linkcode`
- `sphinx.ext.graphviz`
- `sphinx.ext.inheritance_diagram`
- `autoapi.extension`
- `sphinxcontrib.mermaid`

### Optional Extras
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) + mkdocstrings + mkdocs-gen-files (see `mkdocs.yml`).
- Graphviz system binary required for graphviz/inheritance diagrams.

---

## Pre-Commit Hooks

`.pre-commit-config.yaml` wires in:
- **Ruff** (`ruff --fix`) for linting and code-style checks.
- **Black** for Python formatting.
- **Mypy** for static type checking (`args: [src]`).
- **Docformatter** (custom wrapper) for docstring formatting; prints touched files.
- **pydocstyle** to enforce docstring conventions.
- **Interrogate** to enforce docstring coverage (90%).

Run all hooks manually with `pre-commit run --all-files`.

---

## Workflows & CI

- `tools/update_docs.sh` is the recommended local command for a full rebuild.
- CI (`.github/workflows/ci.yml`) runs Ruff, Black, pytest, the documentation pipeline (`tools/update_docs.sh`), and checks for a clean tree afterward.
- `make docstrings` is idempotent; it will rewrite docstrings, nav maps, and docformatter output each run.

---

## Things to Know

1. **Link Modes**
   - `DOCS_LINK_MODE=editor` (default) creates `vscode://` or `pycharm://` URLs.
   - `DOCS_LINK_MODE=github` + `DOCS_GITHUB_ORG/REPO` produce commit-stable permalinks.
2. **Nav Maps**
   - Only modules with existing docstrings are rewritten; others are left untouched.
   - Summaries are truncated to fit Ruff’s 100-character line limit.
3. **README Links**
   - Each entry includes both an editor link (`open`) and a relative Markdown link (`view`).
   - Run Doctoc after regeneration if you want actual TOC entries populated.
4. **Symbol Index**
   - `make symbols` (or `python docs/_scripts/build_symbol_index.py`) writes `docs/_build/symbols.json` for agent consumption.
5. **Type Checking**
   - `mypy.ini` is configured with `mypy_path = src` and ignores for third-party packages lacking stubs (duckdb, faiss, etc.).
6. **Docstring Templates**
   - Located under `tools/doq_templates/numpy/` and used by `doq` during `make docstrings` to emit NumPy-style docstrings.
7. **Environment Variables**
   - `DOCS_EDITOR` for editor scheme (`vscode`, `pycharm`).
   - `SPHINX_AUTOBUILD_PORT` to change the live server port.
   - `SPHINX_THEME` to override the HTML theme.

---

## Quick Commands

```bash
# regenerate everything (docstrings, nav maps, READMEs, Sphinx outputs)
tools/update_docs.sh

# individual tasks
make docstrings
make readmes
make html
make json
make symbols
make watch
```

For questions or modifications, see `Automated Documentation Start.md` or the
comments inside each tooling script.
