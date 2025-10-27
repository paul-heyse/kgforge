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
| [`docs/_scripts/build_symbol_index.py`](docs/_scripts/build_symbol_index.py) | Emits `docs/_build/symbols.json` for agent consumption. |
| [`tools/detect_pkg.py`](tools/detect_pkg.py) | Detects primary/all packages for doc generation (used by Sphinx/Makefile/tools). |

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
- Summaries are truncated (~60 chars) so each bullet stays within the 100-char line length enforced by Ruff/Black/docformatter.

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
- `make bootstrap`: creates `.venv`, installs `pip install -e "[dev,docs]"`, sets up pre-commit.

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

- **Ruff (imports)** – runs `ruff --select I --fix` to normalize import ordering before other checks.
- **Ruff (lint+fix)** – runs `ruff --fix` with the full ruleset enabled in `pyproject.toml`.
- **Ruff (format)** – applies Ruff’s formatter (`ruff format`) so commits always match the project style.
- **Black** – downstream safety net; should be a no-op if Ruff formatting has already run.
- **Mypy** – static type checking with strict mode configured via `mypy.ini` (invoked as `mypy src`).
- **Docformatter** (custom wrapper) for docstring formatting; prints touched files.
- **pydocstyle** to enforce docstring conventions.
- **Interrogate** to enforce docstring coverage (90%).

Run all hooks manually with `pre-commit run --all-files`.

---

## Formatting & Type Checking Pipeline

- Ruff handles import sorting, linting (including automated fixes), and code formatting. The three Ruff hooks run sequentially so that formatting happens before Black and before mypy executes.
- Black runs after Ruff to catch any drift (for example if Ruff is upgraded and emits slightly different formatting).
- Mypy is run in the same pre-commit pass and stops a commit if type checking fails. The configuration in `mypy.ini` enables strict options and whitelists third-party modules that do not ship type stubs.

Because these run on every commit, local commands like `tools/update_docs.sh` maintain code style and type safety as part of their workflow (the script itself calls tooling that reruns Ruff/Black/Mypy when you later commit).

### Style configuration highlights

- Line length is 100 in both Ruff and Black; docformatter also wraps at 100.
- Ruff’s formatter is authoritative; Black acts as a safety net and should be a no-op.
- Mypy targets Python 3.13 with strict options; `mypy_path = src` for imports.

---

## Workflows & CI

- `tools/update_docs.sh` is the recommended local command for a full rebuild.
- If CI is configured, it should run Ruff, Black, pytest, the documentation pipeline (`tools/update_docs.sh`), and assert a clean working tree.
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

# formatting & linting
make fmt
make lint
```

For questions or modifications, see the comments inside each tooling script.
