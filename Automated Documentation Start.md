Awesome—here’s a **turn‑key, agent‑first documentation system** you can drop into your repo today. I’ve already generated a complete starter kit with all files and scripts; you can download it and follow the step‑by‑step runbook below.

Kit downloaded at agent-docs-kit.zip

> What you’ll get:
>
> * Sphinx configuration that **parses your code statically** (no imports), builds HTML + **JSON** docs, and adds **per‑symbol, line‑precise** “Open in editor” links.
> * Scripts to **generate/fill docstrings**, **auto‑build package READMEs** with an always‑current **TOC** and **deep links to symbol starts**, and a **symbol index JSON** for your AI agents.
> * Optional MkDocs variant (if you prefer Markdown‑native authoring for the site).
> * Pre‑commit hooks, a Makefile, and an (optional) Import Linter setup to **enforce architecture**.

---

## 1) What this solves (aligned to your scope)

* **Docstrings:** seed/convert/normalize and enforce coverage & style.
* **Navmaps:** complete API reference generated directly from your code (static analysis).
* **Package‑level READMEs:** materialized automatically, TOCs auto‑updatable, with **line‑precise deep links**.
* **Symbol pages + stable anchors:** every object has a canonical doc anchor; **view source** jumps to the right lines.
* **Line‑precise “Open in editor” links:** `vscode://file/<abs>:line:col` (or GitHub permalinks on demand).
* **Instant local rebuilds:** live reload on save.
* **Agent‑ready artifacts:** Sphinx **JSON** corpus + a compact **symbols.json** (fqname → file + line range + summary).
* **Architecture encoding (optional):** Import Linter contracts + ADRs + C4/Mermaid diagrams.

---

## 2) Files included in the ZIP (what to add to your repo)

```
docs/
  conf.py                          # Sphinx config (static parsing, JSON builder, editor links)
  index.md                         # MyST index with toctrees
  _scripts/
    build_symbol_index.py          # builds docs/_build/symbols.json for agents
    mkdocs_gen_api.py              # (optional) MkDocs API generator
  architecture/
    adr/0001-record-architecture-decisions.md
    diagrams/.gitkeep
  _static/.gitkeep
tools/
  detect_pkg.py                    # auto-detects your top-level package (src/<pkg> or <pkg>)
  gen_readmes.py                   # generates package-level READMEs with deep links
  make_importlinter.py             # writes a .importlinter with the detected package
optional/
  mkdocs.yml                       # MkDocs alternative site config
.pre-commit-config.yaml            # docstring lint/format + coverage hooks
Makefile                           # one-touch tasks: docstrings, readmes, html/json, watch, symbols
README-AGENT.md                    # on-repo runbook (same steps as below)
```

> All file contents are already created in the ZIP. You can open them to see implementation details.

---

## 3) One‑time setup (AI‑agent friendly runbook)

> These steps are robust to *any* repo layout (it autodetects whether you use `src/<pkg>` or `<pkg>` at the repo root).

1. **Unpack the kit** at the repo root (preserving directories):

```bash
unzip agent-docs-kit.zip -d .
```

2. **Install the docs + docstring tooling** (either pip or uv):

```bash
# pip
pip install -U pip
pip install -e ".[docs]"

# or uv (if you use it)
uv pip install -e ".[docs]"
```

3. **(Optional)** Create Import Linter contracts (for layers)

```bash
python tools/make_importlinter.py
# later in CI: lint-imports
```

4. **Generate/normalize docstrings** (seed → format → enforce coverage)

```bash
make docstrings
# Re-run as needed while authoring.
```

5. **Generate package‑level READMEs** (+ TOCs) with deep links

Choose link style:

* Editor links (local, instant):
  `export DOCS_LINK_MODE=editor` (default), `export DOCS_EDITOR=vscode` (or `pycharm`)
* GitHub permalinks (commit‑stable):
  `export DOCS_LINK_MODE=github`, `export DOCS_GITHUB_ORG=<org>`, `export DOCS_GITHUB_REPO=<repo>`

Then:

```bash
make readmes
# Optional: doctoc updates TOCs if installed (Makefile calls it if present).
```

6. **Build docs** (human + machine)

```bash
make html
make json
make symbols
# HTML → docs/_build/html/
# JSON corpus → docs/_build/json/
# Symbol index → docs/_build/symbols.json
```

7. **Live reload while you edit**

```bash
make watch
# A local server opens; refreshes on saves.
```

> On any symbol page, click **[source]** to jump to the exact lines, or the custom link to **Open in VS Code**.

---

## 4) pyproject.toml updates (minimal, safe changes)

I inspected your attached `pyproject.toml`. You already consolidated the documentation tooling into the single `docs` extra, so no other extras are required. Just make sure `griffe` (for static parsing) and any optional packages such as `graphviz` are included under `[project.optional-dependencies].docs`. If you use Graphviz or inheritance diagrams, remember to install the Graphviz system binary in addition to the Python package.

---

## 5) What each file does (so agents can reason about it)

* **`docs/conf.py`**

  * Auto‑detects the package (`src/<pkg>` or `<pkg>`).
  * Uses **AutoAPI** to generate API pages **statically** (no imports).
  * Adds **`viewcode`** with line numbers and **`linkcode`** that produces:

    * editor deep links (`vscode://file/<abs_path>:line:col`) by default, or
    * GitHub permalinks (`.../blob/<SHA>/path#Lstart-Lend`) if `DOCS_LINK_MODE=github`.
  * Builds **JSON** (for the AI corpus) and HTML.

* **`tools/gen_readmes.py`**
  Walks the package with **Griffe** and writes a `README.md` per package that lists top‑level classes/functions with **deep links** (editor or GitHub). Run `doctoc` to refresh TOCs.

* **`docs/_scripts/build_symbol_index.py`**
  Writes `docs/_build/symbols.json`: list of `{fqname, kind, file, lineno, endlineno, doc}`. Great for a RAG/agent index.

* **`.pre-commit-config.yaml`**
  `pydocstyle`, `docformatter`, and `interrogate` to keep docstrings **present, consistent, and covered**.

* **`tools/make_importlinter.py`**
  Writes a `.importlinter` tailored to your package, with a default layered contract (`presentation → domain → infrastructure`). Adjust to your design.

* **`Makefile`**
  Tasks: `docstrings`, `readmes`, `html`, `json`, `symbols`, `watch`.

* **ADRs & diagrams**
  An ADR stub (MADR‑style) plus a diagrams folder ready for C4/Mermaid.

* **MkDocs (optional)**
  `optional/mkdocs.yml` + `docs/_scripts/mkdocs_gen_api.py` for a Markdown‑native site. (Sphinx remains the default.)

---

## 6) How to switch link styles (no code changes)

* **Editor links (local & instant)** *(recommended for development)*

  ```
  export DOCS_LINK_MODE=editor
  export DOCS_EDITOR=vscode      # or pycharm
  ```

  Clicking **Open in VS Code** from docs/READMEs opens the file at the **correct start line**.

* **GitHub permalinks (commit‑stable)** *(great for sharing with teammates)*

  ```
  export DOCS_LINK_MODE=github
  export DOCS_GITHUB_ORG=<org> DOCS_GITHUB_REPO=<repo>
  # optionally override commit with DOCS_GITHUB_SHA
  ```

---

## 7) Suggested CI (optional)

Add `.github/workflows/docs.yml`:

```yaml
name: docs
on: [push, pull_request]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -U pip && pip install -e ".[docs]"
      - run: python tools/gen_readmes.py
      - run: sphinx-build -W -b html docs docs/_build/html
      - run: sphinx-build -W -b json docs docs/_build/json
      # optional:
      - run: lint-imports
      - run: sphinx-build -b linkcheck docs docs/_build/linkcheck
```

---

## 8) What your AI agents should index

* `docs/_build/json/` (Sphinx JSON builder output: `*.fjson`, `globalcontext.json`, `searchindex.json`)
* `docs/_build/symbols.json` (compact symbol map: fqname → file + line range + summary)
* `src/**/README.md` (package READMEs with deep links)
* `docs/architecture/adr/**` (the “why” behind your architecture)

---

## 9) Notes & tips

* Your `pyproject.toml` already provides an extensive docs extra. Adding **`griffe`** (and the docstring tools optional extra) is the key change for this system. 
* If you plan to use `sphinx.ext.graphviz` or `inheritance_diagram`, install **Graphviz** on the system.
* Everything works **entirely locally**; no service dependencies required.
* The configuration **does not import your package**—it uses static analysis (AutoAPI/Griffe), so it’s side‑effect safe and works on partial builds.

---

## 10) Quick start (3 commands)

```bash
unzip agent-docs-kit.zip -d .
pip install -e ".[docs]"
make docstrings readmes html json symbols
```

You’ll have:

* a browsable site in `docs/_build/html/` (with symbol anchors and editor deep links), and
* an agent corpus in `docs/_build/json/` + `docs/_build/symbols.json`.

---

If you want me to tailor the files to a specific package name, editor (VS Code vs PyCharm), or linking style (GitHub default), say the word—I can emit a second ZIP with everything pre‑inlined.


# Detailed reference and background

Below is a **single, end‑to‑end, “agent‑first” documentation system** for Python that:

* **Creates/fills docstrings** and enforces coverage & style
* **Auto‑builds nav maps** and **package‑level README pages**
* **Auto‑updates tables of contents** in those READMEs
* Generates **symbol pages with stable anchors** (per object)
* Adds **deep links to the exact start line** of each symbol
* Provides **line‑precise “open in editor” links** (e.g., VS Code)
* **Rebuilds instantly** on local edits (no network required)
* Emits **machine‑readable JSON** (Sphinx JSON + optional symbol index)
* Optionally **enforces architecture** (layers/contracts + ADRs + diagrams)

It is **Sphinx‑first**, entirely local, and includes a MkDocs alternative. Citations are included at the points where behavior depends on external tools or formats.

---

## 0) Outputs (what this system produces)

1. **Human docs**: HTML site with a complete API reference, per‑symbol pages, “View source” links (line‑exact), diagrams, and architecture notes (ADRs). Sphinx supports this natively (HTML builder). ([Sphinx][1])

2. **Machine docs**: Sphinx **JSON builder** (`.fjson` pages + `globalcontext.json` + `searchindex.json`) for indexing/RAG. ([Sphinx][1])

3. **Symbol‑index JSON (optional)**: A compact map from fully qualified name → `{path, lineno, endlineno, kind, signature, doc summary}`, generated by **Griffe**; perfect for AI agent navigation. ([mkdocstrings][2])

4. **Per‑package README.md** inside your source tree, with a live TOC and **deep links** either to:

   * **GitHub commit‑stable line URLs** `#Lstart-Lend`, or
   * local **`vscode://file/...:line:column`** URIs for immediate open‑in‑editor. ([GitHub Docs][3])

5. **Stable anchors** for all documented symbols—for navigation **within the docs**—plus “view source” links landing at the correct start line. Sphinx `viewcode`/`linkcode` and mkdocstrings provide this. ([Sphinx][4])

6. **Architecture enforcement artifacts**: import‑layer contracts (Import Linter), ADRs, and optional C4 diagrams (PlantUML/Mermaid). ([Import Linter][5])

---

## 1) Repo layout (works for any package name)

```
repo/
├─ pyproject.toml
├─ src/yourpkg/...
├─ docs/
│  ├─ conf.py
│  ├─ index.md
│  ├─ _scripts/
│  │  ├─ gen_api.py              # MkDocs variant (optional)
│  │  └─ build_symbol_index.py   # Optional symbol-index JSON
│  ├─ architecture/
│  │  ├─ adr/                    # ADRs (Markdown)
│  │  └─ diagrams/               # .puml or .mmd (optional)
│  └─ _static/ (optional)
├─ tools/
│  ├─ gen_readmes.py             # Package-level README generator
│  └─ source_linker.py           # (optional) for Pydoc-Markdown
├─ .importlinter                 # Architecture contracts (optional)
├─ .pre-commit-config.yaml
└─ Makefile (or noxfile.py)
```

> Use the **src/** layout; it keeps imports clean and tooling predictable. (Sphinx autodoc warns about import side‑effects; we rely on *static* analyzers—AutoAPI/Griffe—to avoid imports.) ([Sphinx][6])

---

## 2) Install (single command block you can paste)

```bash
# --- Sphinx stack (human + JSON builds, static API parse, Markdown authoring)
pip install "sphinx>=7" furo myst-parser \
  sphinx-autoapi sphinx-autodoc-typehints \
  sphinxcontrib-mermaid

# Optional Sphinx helpers
pip install sphinx-autobuild  # live-reload dev server
pip install sphinx-needs      # JSON export of requirements/needs

# --- Docstring automation & enforcement (in-code docs)
pip install doq pyment interrogate pydocstyle docformatter

# --- Architecture & diagrams (optional but recommended)
pip install import-linter     # layer contracts
# If you embed Graphviz diagrams or inheritance diagrams:
pip install graphviz

# --- MkDocs alternative (optional)
pip install mkdocs mkdocs-material mkdocstrings[python] \
            mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index griffe

# --- README TOC auto-update (pick one)
npm install -g doctoc         # GitHub-compatible Markdown TOCs
# or:
pip install md-toc
```

* **AutoAPI** generates a complete API reference **without importing** (static parse). ([Sphinx AutoAPI][7])
* **MyST‑Parser** adds Markdown authoring to Sphinx. ([Sphinx][8])
* **Type hints rendering** via `sphinx-autodoc-typehints`. ([GitHub][9])
* **`viewcode`** adds “[source]” pages; **`viewcode_line_numbers = True`** adds inline line numbers (Sphinx 7.2+). ([Sphinx][4])
* **Mermaid** diagrams in Sphinx via `sphinxcontrib-mermaid`. ([GitHub][10])
* **sphinx‑autobuild** watches and hot‑reloads docs in a dev server. ([GitHub][11])
* **Sphinx‑Needs** can export `needs.json` for machine consumption. ([Sphinx-Needs][12])
* **Docstrings**: doq (generate), Pyment (convert/fill), interrogate (coverage), pydocstyle/docformatter (style/format). ([GitHub][13])
* **MkDocs alternative**: mkdocstrings (Python handler) built on **Griffe** static analysis; can show source and compute locations. ([mkdocstrings][14])
* **README TOC**: **doctoc** generates GitHub‑compatible anchors; **md‑toc** is a Python alternative. ([GitHub][15])

> If you embed Graphviz diagrams or `inheritance_diagram`, install Graphviz and enable `sphinx.ext.graphviz`. ([Sphinx][16])

---

## 3) Sphinx configuration (agent‑first, static, line‑precise)

**`docs/conf.py`** — drop‑in, zero‑import config that uses **AutoAPI** and **Griffe** to compute line locations (so `linkcode` doesn’t import your package):

```python
# docs/conf.py
import os, sys
from pathlib import Path

# --- Project metadata
project = os.environ.get("PROJECT_NAME", "Your Project")
author  = os.environ.get("PROJECT_AUTHOR", "Your Team")

# --- Paths
DOCS_DIR = Path(__file__).resolve().parent
ROOT     = DOCS_DIR.parent
SRC_DIR  = ROOT / "src"

# --- Detect the top-level package automatically (first src/* with __init__.py)
def _detect_pkg():
    for p in (SRC_DIR if SRC_DIR.exists() else ROOT).glob("*/__init__.py"):
        return p.parent.name
    raise RuntimeError("No package found under src/ or project root")
PKG = os.environ.get("DOCS_PKG", _detect_pkg())

sys.path.insert(0, str(ROOT))   # only for Sphinx internals (not used by linkcode)

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "autoapi.extension",            # static API docs (no import)
    # "sphinx_needs",               # optional: requirements export
    # "autodoc_pydantic",          # optional: if you use Pydantic models
    "sphinxcontrib-mermaid",
]

html_theme = "furo"
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# --- AutoAPI (static parse of Python)
autoapi_type = "python"
autoapi_dirs = [str(SRC_DIR)]
autoapi_add_toctree_entry = True
autoapi_options = [
    "members", "undoc-members", "show-inheritance", "special-members",
    "imported-members"
]

# --- Docstrings and type hints
autodoc_typehints = "description"  # render type hints in descriptions

# --- Cross-links to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {"objects.inv": None}),
}

# --- Show code with line numbers in viewcode pages (Sphinx >= 7.2)
viewcode_line_numbers = True

# --- Build deep links *without importing* the code (use Griffe)
from griffe.loader import GriffeLoader
_loader = GriffeLoader(search_paths=[str(SRC_DIR)])
_root = _loader.load(PKG)

def _lookup(module: str, fullname: str):
    """Find (abs_path, start_line, end_line) via Griffe."""
    node = _root
    # descend to module/package
    if module and module != PKG:
        for part in module.split(".")[1:]:
            node = node.members.get(part, node)
    # descend to object
    for part in (fullname or "").split("."):
        if not part:
            continue
        node = node.members.get(part, node)
    file_rel = getattr(node, "relative_package_filepath", None)
    start    = getattr(node, "lineno", None)
    end      = getattr(node, "endlineno", None)
    if not (file_rel and start):
        return None
    abs_path = (SRC_DIR / file_rel).resolve()
    return str(abs_path), int(start), int(end or start)

# Editor deep-link scheme: VS Code recommended; can switch via env
EDITOR = os.getenv("DOCS_EDITOR", "vscode")  # "vscode" | "pycharm" | "none"

def linkcode_resolve(domain, info):
    """Return a deep link for each symbol."""
    if domain != "py" or not info.get("module"):
        return None
    res = _lookup(info["module"], info.get("fullname", ""))
    if not res:
        return None
    path, start, end = res
    if EDITOR == "vscode":
        # vscode://file/ABS:line:col
        return f"vscode://file/{path}:{start}:1"
    elif EDITOR == "pycharm":
        # works if pycharm CLI launcher is on PATH
        return f"pycharm://open?file={path}&line={start}"
    # fallback: let viewcode provide an internal source page
    return None
```

* **AutoAPI** parses source and generates API pages, avoiding import side‑effects. ([Sphinx AutoAPI][7])
* **Napoleon** parses Google/NumPy docstrings. ([Sphinx][17])
* **`viewcode_line_numbers = True`** adds line numbers to rendered source pages. ([Sphinx][4])
* **`linkcode`** adds a per‑symbol **external** link that we set to **`vscode://file/...:line`** or a PyCharm link; VS Code’s URL scheme supports opening files at a line/column. ([Sphinx][18])

> If you prefer GitHub links in docs (instead of editor links), replace `linkcode_resolve` to compute `https://github.com/<org>/<repo>/blob/<SHA>/<rel>#Lstart-Lend`. GitHub’s **permalinks** are stable and recommended (`press “Y”` in the UI). ([GitHub Docs][3])

---

## 4) Authoring in Markdown (MyST) + API pages

**`docs/index.md`** (MyST):

````md
# Your Project

```{toctree}
:maxdepth: 2
:caption: Guide
getting-started.md
how-to/index
explanations/index
````

```{toctree}
:maxdepth: 2
:caption: Reference
api/index
architecture/index
```

````

> MyST enables Markdown authoring in Sphinx. :contentReference[oaicite:23]{index=23}  
> AutoAPI adds an API toctree under `api/` automatically. :contentReference[oaicite:24]{index=24}

---

## 5) Build targets (human + machine + live)

```bash
# Human-readable docs
sphinx-build -b html docs/ docs/_build/html

# Machine-readable docs (JSON corpus for agents)
sphinx-build -b json docs/ docs/_build/json

# Live local preview with hot reload
sphinx-autobuild docs/ docs/_build/html
````

* **JSON builder** outputs `*.fjson`, `globalcontext.json`, `searchindex.json`. ([Sphinx][1])
* **sphinx‑autobuild** watches sources and reloads the browser. ([GitHub][11])

---

## 6) Auto‑create/fill docstrings + enforce coverage & style

Run locally and in CI:

```bash
# Generate skeletons (Google style) for missing docstrings
doq -d google -r src/yourpkg

# Convert/harmonize styles (e.g., to NumPy)
pyment -w -o numpydoc -r src/yourpkg

# Enforce docstring coverage: fail if < 90%
interrogate -i src/yourpkg --fail-under 90

# Lint and format docstrings
pydocstyle src/yourpkg
docformatter -r -i src/yourpkg
```

* **doq** (generate), **Pyment** (create/convert), **interrogate** (coverage), **pydocstyle**/**docformatter** (style). ([GitHub][13])

---

## 7) Package‑level READMEs with **always‑current** TOC and deep links

**Goal:** A `README.md` inside each `src/yourpkg/subpkg/` with:

* an updated **TOC**
* one entry per top‑level class/function with a link **to its start line**.

Pick **(A)** local editor links (open in VS Code), or **(B)** commit‑stable GitHub URLs.

### tools/gen_readmes.py (A: VS Code local deep links)

```python
# tools/gen_readmes.py
import pathlib
from griffe.loader import GriffeLoader

SRC_DIR = pathlib.Path("src")
PKG = "yourpkg"  # or read from env

loader = GriffeLoader(search_paths=[str(SRC_DIR)])
root = loader.load(PKG)

def vs_link(path, line):  # vscode URL scheme
    return f"vscode://file/{path}:{line}:1"

def write_pkg_readme(node):
    pkg_dir = SRC_DIR / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"
    lines = [f"# `{node.path}`\n\n", "## API (open in editor)\n"]
    for child in node.members.values():
        if child.kind.value in {"class", "function"} and child.lineno:
            abs_path = (SRC_DIR / child.relative_package_filepath).resolve()
            lines.append(f"- **`{child.path}`** → [open]({vs_link(abs_path, child.lineno)})\n")
    readme.write_text("".join(lines), encoding="utf-8")

for m in root.members.values():
    if m.is_package:
        write_pkg_readme(m)
```

**Update TOCs** after generation:

```bash
# GitHub-compatible anchors
doctoc src/yourpkg   # or: md_toc -p -i src/yourpkg/README.md
```

* VS Code supports `vscode://file/ABS:line:column` links. ([Visual Studio Code][19])
* **Doctoc** writes GitHub‑compatible TOCs. ([GitHub][15])

### tools/gen_readmes.py (B: GitHub **commit‑stable** deep links)

```python
import subprocess, pathlib
from griffe.loader import GriffeLoader

OWNER, REPO = "your-org", "your-repo"
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def gh_link(rel, start, end):
    return f"https://github.com/{OWNER}/{REPO}/blob/{COMMIT}/{rel}#L{start}-L{end}"

# ... use gh_link(child.relative_package_filepath, child.lineno, child.endlineno)
```

* GitHub exposes **permalinks** to files/lines by using the **commit SHA** (press **Y** in the UI or copy permalink). ([GitHub Docs][3])

---

## 8) Symbol pages, stable anchors, and **line‑precise** “View source”

You already get:

* **Per‑symbol anchors** in the docs site (Sphinx pages / sections).
* “[source]” links to **exact line ranges** via `viewcode`/`linkcode`. ([Sphinx][4])

If you prefer MkDocs:

* **mkdocstrings[python]** renders API pages from your code (via Griffe). Set `show_source: true` to display source; it supports options like signatures, inheritance diagrams, etc. ([mkdocstrings][20])

**`mkdocs.yml` (excerpt)**

```yaml
site_name: Your Project
theme: { name: material }
plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
```

> mkdocstrings uses **Griffe** for static parsing; it exposes **lineno/endlineno** to build source links. ([mkdocstrings][2])

---

## 9) Auto‑generated **nav maps** (two ways)

**Sphinx (recommended with AutoAPI):** AutoAPI generates a complete API tree and adds it to the **toctree** automatically. ([Sphinx AutoAPI][7])

**MkDocs alternative:** generate index pages during build with **mkdocs‑gen‑files** (plugin) and place `::: yourpkg.subpkg` blocks that mkdocstrings expands to full pages.

**`docs/_scripts/gen_api.py` (MkDocs)**

```python
from pathlib import Path
import mkdocs_gen_files
from griffe.loader import GriffeLoader

PKG, SRC = "yourpkg", Path("src")
root = GriffeLoader(search_paths=[str(SRC)]).load(PKG)

out = Path("api")
with mkdocs_gen_files.open(out / "index.md", "w") as f:
    f.write("# API Reference\n")

def write_node(node):
    rel = node.path.replace(".", "/")
    page = out / rel / "index.md"
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(f"# `{node.path}`\n\n::: {node.path}\n")
for m in root.members.values():
    if m.is_package or m.is_module:
        write_node(m)
```

Enable the plugin:

```yaml
plugins:
  - gen-files:
      scripts:
        - docs/_scripts/gen_api.py
  - mkdocstrings:
      default_handler: python
```

* **mkdocs‑gen‑files** programmatically writes Markdown at build time; **mkdocs‑section‑index** makes folders clickable; **literate‑nav** lets you drive nav from `SUMMARY.md`. ([Oprypin][21])

---

## 10) Architecture: encode and enforce it (so agents comply)

1. **Contracts** with **Import Linter** (for layers/forbidden imports). Put this in `.importlinter`:

```ini
[importlinter]
root_package = yourpkg

[importlinter:contract:layers]
name = Respect layered architecture
type = layers
layers =
    yourpkg.presentation
    yourpkg.domain
    yourpkg.infrastructure
```

Run `lint-imports` locally/CI; the build fails if rules are violated. ([Import Linter][5])

2. **ADRs** (decision log) under `docs/architecture/adr/`—each ADR captures context, decision, and consequences. ([Architectural Decision Records][22])

3. **C4 diagrams** with C4‑PlantUML and/or Mermaid (Sphinx supports Mermaid via `sphinxcontrib-mermaid`). ([GitHub][23])

---

## 11) Machine‑readable corpora for AI agents

* **Sphinx JSON builder** (per‑page `.fjson`, `globalcontext.json`, `searchindex.json`). Use as an ingestion set for RAG. ([Sphinx][1])
* **Sphinx‑Needs** (optional) to export **`needs.json`** with requirements/design items:
  `sphinx-build -b needs docs/ docs/_build/needs` (or set `needs_build_json` in `conf.py`). ([Sphinx-Needs][12])
* **Symbol index** (optional) with **Griffe**:

**`docs/_scripts/build_symbol_index.py`**

```python
import json
from pathlib import Path
from griffe.loader import GriffeLoader

SRC, PKG = Path("src"), "yourpkg"
root = GriffeLoader(search_paths=[str(SRC)]).load(PKG)
items = []
def walk(n):
    for m in n.members.values():
        entry = {
            "path": m.path,
            "kind": m.kind.value,
            "file": getattr(m, "relative_package_filepath", None),
            "lineno": getattr(m, "lineno", None),
            "endlineno": getattr(m, "endlineno", None),
            "doc": (m.docstring.value.split("\n\n")[0] if getattr(m, "docstring", None) else "")
        }
        items.append(entry)
        walk(m)
walk(root)
out = Path("docs/_build/symbols.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
```

Griffe exposes `lineno/endlineno` and file paths for each symbol. ([mkdocstrings][2])

---

## 12) Pre‑commit & CI (copy/paste)

**`.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/PyCQA/pydocstyle
    rev: 6.3.0
    hooks: [{ id: pydocstyle }]
  - repo: https://github.com/PyCQA/docformatter
    rev: 1.7.5
    hooks: [{ id: docformatter, args: ["-r", "-i", "src/yourpkg"] }]
  - repo: local
    hooks:
      - id: interrogate
        name: interrogate (docstring coverage)
        entry: bash -c "interrogate -i src/yourpkg --fail-under 90"
        language: system
        pass_filenames: false
      - id: doctoc
        name: doctoc (update TOCs)
        entry: doctoc src/yourpkg
        language: node
        pass_filenames: false
```

**GitHub Actions** (key steps):

```yaml
name: docs
on: [push, pull_request]
jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -U pip && pip install -r docs/requirements.txt
      - run: lint-imports          # optional: enforce architecture
      - run: python tools/gen_readmes.py && npx doctoc src/yourpkg
      - run: sphinx-build -W -b html docs/ docs/_build/html
      - run: sphinx-build -W -b json docs/ docs/_build/json
      - run: sphinx-build -b linkcheck docs/ docs/_build/linkcheck
```

* **linkcheck builder** validates external links; keep it in CI. ([Sphinx][1])

---

## 13) Instant local rebuilds (no network)

* Sphinx: `sphinx-autobuild docs docs/_build/html` (watch + hot reload). ([GitHub][11])
* MkDocs: `mkdocs serve` (live reload; you can add extra watch paths). ([MkDocs][24])

---

## 14) Optional MkDocs variant (if you prefer Markdown‑native authoring)

**Pros:** Simpler authoring, mkdocstrings + Griffe static parsing, “show source” toggle, and programmable pages via mkdocs‑gen‑files.

**`mkdocs.yml`**

```yaml
site_name: Your Project
theme:
  name: material
  features: [navigation.indexes]
plugins:
  - search
  - gen-files:
      scripts:
        - docs/_scripts/gen_api.py
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
  - literate-nav
  - section-index
```

* mkdocstrings’ Python handler + **Griffe** provide static API rendering and source links. ([mkdocstrings][14])
* **mkdocs‑gen‑files**/**literate‑nav**/**section‑index** automate pages and navigation. ([Oprypin][21])

---

## 15) Notes on “symbol‑stable” vs “line‑stable” links

* **Docs site anchors** (section IDs per symbol) are **stable** across edits; the site’s **“View source”** links recompute to current lines on every build (Sphinx `linkcode`/`viewcode`; mkdocstrings `show_source`). ([Sphinx][18])
* **GitHub** line links are **line‑stable per commit**; use **permalinks** (commit SHA) to keep them from rotting. You must regenerate when lines move (our scripts do). ([GitHub Docs][3])
* **Open in editor** (`vscode://file/…:line:col`) is **local, instant**, and avoids hosting entirely. ([Visual Studio Code][19])

---

## 16) Architecture docs & diagrams (optional, but agent‑useful)

* **ADRs**: add `docs/architecture/adr/0001-record-architecture-decisions.md`. ([Architectural Decision Records][22])
* **C4 diagrams** in PlantUML (`.puml`) or Mermaid blocks; Sphinx supports Mermaid via `sphinxcontrib-mermaid`. ([GitHub][23])
* **Import Linter** contracts to enforce layering/forbidden imports; run `lint-imports` in CI. ([Import Linter][5])

---

## 17) Makefile (or nox) to orchestrate everything

**`Makefile`**

```make
PKG ?= yourpkg

init:
\tpip install -U pip
\tpip install -r docs/requirements.txt

docstrings:
\tdoq -d google -r src/$(PKG)
\tpyment -w -o numpydoc -r src/$(PKG)
\tpydocstyle src/$(PKG) && docformatter -r -i src/$(PKG)
\tinterrogate -i src/$(PKG) --fail-under 90

readmes:
\tpython tools/gen_readmes.py
\tdoctoc src/$(PKG)

html:
\tsphinx-build -W -b html docs docs/_build/html

json:
\tsphinx-build -W -b json docs docs/_build/json

watch:
\tsphinx-autobuild docs docs/_build/html
```

---

## 18) What an AI agent needs to do (step‑by‑step, regardless of repo shape)

1. **Detect package name** under `src/*/__init__.py` (or use `DOCS_PKG`).
2. **Install** the tooling (Section 2).
3. **Write** `docs/conf.py` from Section 3 **verbatim**.
4. **Create** `docs/index.md` from Section 4.
5. **Generate docstrings** and **enforce coverage** (Section 6).
6. **Generate READMEs** and **update TOCs** (Section 7) using either VS Code URIs or GitHub permalinks.
7. **Build docs**: `make html` and `make json`.
8. **(Optional)** build `symbols.json` (Section 11).
9. **(Optional)** add `.importlinter` and ADRs (Sections 10 & 16).
10. **Run live server** for instant local rebuilds (Section 13).

---

## 19) Why this is “best‑in‑class” for **AI agent performance**

* **Complete, static API coverage** via AutoAPI/mkdocstrings—no imports or runtime needed. ([Sphinx AutoAPI][7])
* **Stable per‑symbol anchors** + **line‑exact source links** (recomputed each build) so agents can jump from docs → code deterministically. ([Sphinx][4])
* **Machine corpora** ready for indexing (Sphinx JSON and optional symbols index). ([Sphinx][1])
* **Architecture encoded** (contracts + ADRs + diagrams), preventing agents from proposing designs that violate layering. ([Import Linter][5])
* **Local‑only path** with editor deep links and live rebuilds—no lag, no dependency on hosting. ([GitHub][11])

---

### References (selected)

* Sphinx builders (HTML/JSON), viewcode, linkcode, intersphinx, autosummary/inheritance/graphviz: ([Sphinx][1])
* MyST‑Parser (Markdown in Sphinx): ([Sphinx][8])
* AutoAPI (static API docs): ([Sphinx AutoAPI][7])
* sphinx‑autodoc‑typehints: ([GitHub][9])
* sphinx‑autobuild: ([GitHub][11])
* Sphinx‑Needs “needs.json”: ([Sphinx-Needs][12])
* Docstring automation (doq, Pyment, interrogate, pydocstyle, docformatter): ([GitHub][13])
* MkDocs + mkdocstrings + Griffe + gen‑files + literate‑nav + section‑index: ([mkdocstrings][14])
* VS Code URL scheme (open file at line/column): ([Visual Studio Code][19])
* GitHub **permalink to code** / **line ranges**: ([GitHub Docs][3])
* Import Linter, ADRs, C4‑PlantUML: ([Import Linter][5])

---

### TL;DR (copy this into your tracker)

* [ ] Install the dependencies in §2.
* [ ] Add **`docs/conf.py`** in §3 (copy verbatim).
* [ ] Add **`docs/index.md`** in §4.
* [ ] Run **docstring generation & enforcement** in §6.
* [ ] Generate **package READMEs** and **TOCs** (§7).
* [ ] Build **HTML** + **JSON** (§5).
* [ ] (Optional) Generate **symbols.json** (§11).
* [ ] (Optional) Add **.importlinter** and **ADRs/C4** (§10 & §16).
* [ ] Add **pre‑commit** and **CI** glue (§12).
* [ ] Use **`sphinx-autobuild`** (or `mkdocs serve`) during development (§13).

If you want me to tailor the file stubs to a particular package name, I can output the final `conf.py`, `Makefile`, `tools/*.py`, and `mkdocs.yml` with your identifiers inlined.

[1]: https://www.sphinx-doc.org/en/master/usage/builders/index.html?utm_source=chatgpt.com "Builders — Sphinx documentation"
[2]: https://mkdocstrings.github.io/griffe/?utm_source=chatgpt.com "Overview - Griffe - mkdocstrings"
[3]: https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files?utm_source=chatgpt.com "Getting permanent links to files"
[4]: https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html?utm_source=chatgpt.com "sphinx.ext.viewcode – Add links to highlighted source code"
[5]: https://import-linter.readthedocs.io/en/stable/contract_types.html?utm_source=chatgpt.com "Contract types — Import Linter 2.5.2 documentation"
[6]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html?utm_source=chatgpt.com "sphinx.ext.autodoc – Include documentation from docstrings"
[7]: https://sphinx-autoapi.readthedocs.io/?utm_source=chatgpt.com "Sphinx AutoAPI 3.6.1 documentation"
[8]: https://www.sphinx-doc.org/en/master/usage/markdown.html?utm_source=chatgpt.com "Markdown — Sphinx documentation"
[9]: https://github.com/tox-dev/sphinx-autodoc-typehints?utm_source=chatgpt.com "tox-dev/sphinx-autodoc-typehints"
[10]: https://github.com/mgaitan/sphinxcontrib-mermaid?utm_source=chatgpt.com "Mermaid diagrams in yours sphinx powered docs"
[11]: https://github.com/sphinx-doc/sphinx-autobuild?utm_source=chatgpt.com "sphinx-doc/sphinx-autobuild"
[12]: https://sphinx-needs.readthedocs.io/en/latest/builders.html?utm_source=chatgpt.com "Builders - Sphinx-Needs 6.0.1 documentation"
[13]: https://github.com/heavenshell/py-doq?utm_source=chatgpt.com "heavenshell/py-doq: Docstring generator"
[14]: https://mkdocstrings.github.io/usage/?utm_source=chatgpt.com "Usage - mkdocstrings"
[15]: https://github.com/thlorenz/doctoc?utm_source=chatgpt.com "thlorenz/doctoc"
[16]: https://www.sphinx-doc.org/en/master/usage/extensions/graphviz.html?utm_source=chatgpt.com "Add Graphviz graphs"
[17]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html?utm_source=chatgpt.com "Support for NumPy and Google style docstrings"
[18]: https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html?utm_source=chatgpt.com "sphinx.ext.linkcode – Add external links to source code"
[19]: https://code.visualstudio.com/docs/configure/command-line?utm_source=chatgpt.com "Command Line Interface (CLI)"
[20]: https://mkdocstrings.github.io/python/reference/api/?utm_source=chatgpt.com "API reference - mkdocstrings-python"
[21]: https://oprypin.github.io/mkdocs-gen-files/index.html?utm_source=chatgpt.com "Manual - mkdocs-gen-files - GitHub Pages"
[22]: https://adr.github.io/?utm_source=chatgpt.com "Architectural Decision Records (ADRs) | Architectural ..."
[23]: https://github.com/plantuml-stdlib/C4-PlantUML?utm_source=chatgpt.com "C4-PlantUML combines the benefits ..."
[24]: https://www.mkdocs.org/user-guide/cli/?utm_source=chatgpt.com "Command Line Interface"
