Love it — let’s lock in **ReDoc** and turn this into a complete, junior-friendly, end-to-end build that you can drop into any repo.

Below you’ll get:

* a **minimal file layout** to copy
* a **production-grade `mkdocs.yml`** (Material + mkdocstrings/Griffe + D2 + ReDoc + cross-refs + strict build polish)
* **two tiny generators** that build (1) *one-page-per-module* summaries with “nearest-neighbor” links (imports / imported-by / exports / inheritance) and (2) *folder-cluster D2 diagrams* with clickable nodes
* **API page in ReDoc** with deep links
* **AI-onboarding helpers** (an `/llms.txt` file, repo/source links, etc.)

I’ll explain each piece right where you paste it so a junior dev can implement it solo.

---

# 0) Prereqs (one-time)

```bash
# Core
pip install mkdocs mkdocs-material

# Autodoc + structure (mkdocstrings+Griffe)
pip install mkdocstrings mkdocstrings-python griffe

# Programmatic page generation and nav helpers
pip install mkdocs-gen-files mkdocs-autorefs mkdocs-section-index

# Diagrams (D2) – plugin + install the D2 CLI
pip install mkdocs-d2-plugin
# Install D2 CLI (Linux/macOS): https://d2lang.com -> get >= 0.6.3

# ReDoc (embed as a tag)
pip install mkdocs-redoc-tag

# UX / polish
pip install mkdocs-redirects mkdocs-minify-plugin \
           mkdocs-git-revision-date-localized-plugin mkdocs-glightbox

# (Optional) LLM onboarding file
pip install mkdocs-llmstxt
```

* **MkDocs** and **Material** are the base SSG + theme. Configure Git repo integration to enable *Edit/View Source* buttons. ([MkDocs][1])
* **mkdocstrings (Python handler)** uses **Griffe** to read your package structure & docstrings; we’ll lean on it (no stubs to maintain). ([mkdocstrings][2])
* **mkdocs-gen-files** lets us **generate pages during build**, and `set_edit_path` maps a generated page to the real source file so “Edit this page” jumps into code. ([Oprypin][3])
* **autorefs** auto-links headings across pages (great for generated pages). ([PyPI][4])
* **D2 plugin** renders `.d2` into SVG at build time — we’ll make clickable nodes. Install the D2 CLI. ([LandMaj][5])
* **ReDoc** via `mkdocs-redoc-tag` adds a `<redoc/>` tag to any Markdown page; it syncs with Material dark mode. ([GitHub][6])

---

# 1) Layout to copy

```
mkdocs.yml
docs/
  index.md
  architecture/index.md
  api/index.md        # our ReDoc page
  modules/index.md    # landing page for “one page per module”
  diagrams/index.md   # landing for folder-cluster diagrams
  _scripts/
    gen_module_pages.py
    gen_d2_diagrams.py
openapi/
  openapi.yaml
```

> Keep code in a package (e.g. `src/your_package/…`). MkDocs expects docs in `docs/` and a single `mkdocs.yml`. ([MkDocs][7])

---

# 2) `mkdocs.yml` — paste this and adjust `site_name`, `repo_url`, `PACKAGE`

```yaml
site_name: Your Project
site_url: https://your.domain/docs/       # required by llmstxt
repo_url: https://github.com/your-org/your-repo
edit_uri: edit/main/                       # Material uses this for edit/view
strict: true                               # fail on warnings in CI

theme:
  name: material
  features:
    - navigation.sections
    - navigation.top
    - navigation.tracking
    - navigation.footer
    - content.action.edit
    - content.action.view                  # show Edit/View buttons

markdown_extensions:
  - admonition
  - attr_list
  - pymdownx.details
  - pymdownx.superfences                   # enables Mermaid fences
  - meta                                   # for localized dates if you want

plugins:
  - search                                 # always re-add if you set plugins
  - autorefs                               # cross-page heading links
  - section-index                          # make section headings clickable
  - gen-files:
      scripts:
        - docs/_scripts/gen_module_pages.py
        - docs/_scripts/gen_d2_diagrams.py # must run AFTER module pages
  - d2                                     # render .d2 to SVG
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_signature: true
            show_root_heading: true
            inherited_members: true
            # Optional (sponsor-only in mkdocstrings-python 1.7+):
            # show_inheritance_diagram: true
            # inheritance_diagram_direction: LR
            # extensions:
            #   - griffe_inherited_docstrings: {merge: true}
  - redoc-tag                              # <redoc src="..."/>
  - redirects                              # generate redirect stubs
  - minify                                 # minify html/css/js
  - git-revision-date-localized            # "last updated" footer
  - glightbox                              # image lightbox
  - llmstxt:
      markdown_description: "How AI agents should navigate this site."
      # optional: sections: see plugin README

nav:
  - Overview: index.md
  - Architecture:
      - architecture/index.md
  - API:
      - api/index.md
  - Modules:
      - modules/index.md
  - Diagrams:
      - diagrams/index.md
```

**Why these settings**

* Material’s **content.action.edit/view** show the edit/view buttons when `repo_url` + `edit_uri` are set. A typical `edit_uri` for GitHub is `edit/main/docs/`, but you can map generated pages to **code paths** via `set_edit_path` (we’ll do this) so the edit button jumps to source. ([Squidfunk][8])
* **Mermaid** works out of the box with Material via `pymdownx.superfences` (mkdocstrings’ *optional* inheritance diagram feature uses Mermaid). ([Squidfunk][9])
* **D2 plugin** handles compiling `.d2` to SVG during build. ([LandMaj][5])
* **mkdocstrings (python)** injects API sections with `:::`; options like `show_source`, `inherited_members` are standard. *`show_inheritance_diagram` is an Insiders feature in mkdocstrings-python 1.7+.* ([mkdocstrings][10])
* **autorefs** auto-resolves cross-page heading links by title — perfect for generated pages. ([PyPI][4])
* **redirects/minify/git-revision-date** harden and polish the build. ([GitHub][11])
* **llmstxt** writes `/llms.txt` (and optionally a *full* variant) so agents know how to traverse the docs; it requires `site_url`. ([GitHub][12])
* `strict: true` (or `mkdocs build --strict`) **fails on warnings** in CI. ([MkDocs][13])

---

# 3) One-page-per-module summaries with “nearest neighbors”

Create `docs/_scripts/gen_module_pages.py` — this runs at build time via **mkdocs-gen-files** and writes one summary page for every module under your top-level package `PACKAGE`:

```python
# docs/_scripts/gen_module_pages.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json

import griffe
import mkdocs_gen_files

PACKAGE = "your_package"   # <-- change this to your top-level import

# Optional: map code -> API operations for cross-linking into ReDoc
# Example file docs/_scripts/api_usage.json:
# { "your_package.modA.func_x": ["getPetById"], "your_package.modB.Client": ["createOrder","listOrders"] }
API_USAGE_FILE = Path(__file__).with_name("api_usage.json")
api_usage: dict[str, list[str]] = {}
if API_USAGE_FILE.exists():
    api_usage = json.loads(API_USAGE_FILE.read_text())

root = griffe.load(PACKAGE, search_paths=["."], submodules=True)  # load full package tree
modules: dict[str, griffe.Module] = {}

# Collect objects and relationships
imports = defaultdict(set)     # module -> imported module paths
imported_by = defaultdict(set) # inverse edges
exports = defaultdict(set)     # module -> exported names (__all__)
classes = defaultdict(list)    # module -> class paths
funcs = defaultdict(list)      # module -> function paths
bases = defaultdict(list)      # class path -> base class paths

def visit(mod: griffe.Module) -> None:
    modules[mod.path] = mod

    # exports from __all__
    if mod.exports:
        for e in mod.exports:
            exports[mod.path].add(str(getattr(e, "name", e)))

    for name, obj in mod.members.items():
        # Imported names appear as Aliases with is_imported=True
        if isinstance(obj, griffe.Alias) and obj.is_imported:
            target = getattr(obj, "target_path", None) or getattr(obj, "target", None)
            if target:
                imports[mod.path].add(str(target))

        # Gather symbol indexes
        if isinstance(obj, griffe.Class):
            classes[mod.path].append(obj.path)
            for b in obj.bases:
                if getattr(b, "path", None):
                    bases[obj.path].append(b.path)
        elif isinstance(obj, griffe.Function):
            funcs[mod.path].append(obj.path)

        # Recurse into submodules
        if isinstance(obj, griffe.Module):
            visit(obj)

visit(root)

# Reverse edges (imported-by)
for src, outs in imports.items():
    for dst in outs:
        imported_by[dst].add(src)

# Emit one page per module
for mod_path, mod in modules.items():
    page_path = f"modules/{mod_path}.md"
    with mkdocs_gen_files.open(page_path, "w") as f:
        f.write(f"# {mod_path}\n\n")

        # short summary line from module docstring (optional)
        if mod.docstring and mod.docstring.value:
            first_para = mod.docstring.value.strip().split("\n\n")[0]
            f.write(first_para + "\n\n")

        # Relationships (“nearest neighbors”)
        outs = sorted(imports.get(mod_path, []))
        incs = sorted(imported_by.get(mod_path, []))
        exps = sorted(exports.get(mod_path, []))
        if outs or incs or exps:
            f.write("## Relationships\n\n")
            if outs:
                f.write("**Imports:** " + ", ".join(f"[{m}](../modules/{m}.md)" for m in outs) + "\n\n")
            if incs:
                f.write("**Imported by:** " + ", ".join(f"[{m}](../modules/{m}.md)" for m in incs) + "\n\n")
            if exps:
                f.write("**Exports (`__all__`):** " + ", ".join(f"`{n}`" for n in exps) + "\n\n")

        # API operations this module touches (manual mapping via api_usage.json)
        used_ops = []
        for sym in classes[mod_path] + funcs[mod_path]:
            used_ops.extend(api_usage.get(sym, []))
        used_ops = sorted(set(used_ops))
        if used_ops:
            f.write("## Related API operations\n\n")
            # ReDoc deep-link format is #operation/<operationId>
            f.write(", ".join(f"[{op}](../api/index/#operation/{op})" for op in used_ops) + "\n\n")

        # Symbol sections – mkdocstrings renders API & optional source
        if classes[mod_path] or funcs[mod_path]:
            f.write("## Contents\n\n")
        for c in sorted(classes[mod_path]):
            f.write(f"### {c}\n\n::: {c}\n\n")
        for fn in sorted(funcs[mod_path]):
            f.write(f"### {fn}\n\n::: {fn}\n\n")

    # Map Edit button to the actual source file (Material uses edit_uri)
    if getattr(mod, "relative_filepath", None):
        mkdocs_gen_files.set_edit_path(page_path, str(mod.relative_filepath))

# Simple landing page
with mkdocs_gen_files.open("modules/index.md", "w") as f:
    f.write("# Modules\n\n")
    for m in sorted(modules):
        f.write(f"- [{m}](./{m}.md)\n")
```

**What it’s doing / why it’s accurate**

* **Griffe** loads your package and gives you a model with `members`, `bases`, `docstring`, file paths, etc. It also distinguishes **imported** vs **exported** (via `__all__`) objects, which we use to build “nearest-neighbor” lists. ([mkdocstrings][14])
* We **reverse imports** to make an *Imported by* list (useful for discovery).
* Each symbol gets a `:::` block — **mkdocstrings** renders signatures, types, and (optionally) source, and supports inheritance options. ([mkdocstrings][2])
* `set_edit_path` points the **Edit** button to the real Python file — great for machine + human workflows. ([Oprypin][3])

> **Note on class inheritance diagrams:** `show_inheritance_diagram` is available in **mkdocstrings-python Insiders 1.7+**; if you have access, enable it as shown above. Otherwise, lean on the folder-level D2 diagrams for structural navigation. ([mkdocstrings][15])

---

# 4) Folder-cluster D2 diagrams with click-through

Create `docs/_scripts/gen_d2_diagrams.py` — it runs *after* module pages are generated and makes one `.d2` per top-level folder cluster:

```python
# docs/_scripts/gen_d2_diagrams.py
from collections import defaultdict
import mkdocs_gen_files

# Build folder -> modules from the virtual file list (module pages exist already)
by_folder: dict[str, list[str]] = defaultdict(list)
for path in list(mkdocs_gen_files.files()):
    if path.startswith("modules/") and path.endswith(".md") and path.count(".") >= 1:
        mod = path[len("modules/"):-3]              # "pkg.sub.mod"
        parts = mod.split(".")
        folder = parts[1] if len(parts) > 1 else parts[0]
        by_folder[folder].append(mod)

# Emit a D2 file per folder with clickable nodes to the module page
for folder, mods in by_folder.items():
    d2_path = f"diagrams/{folder}.d2"
    with mkdocs_gen_files.open(d2_path, "w") as d:
        d.write('direction: right\n')
        d.write(f'{folder}: "{folder}" {{\n')
        for m in sorted(mods):
            d.write(f'  "{m}": "{m}" {{ link: "../modules/{m}.md" }}\n')
        d.write('}\n')

# Landing page that lists available diagrams
with mkdocs_gen_files.open("diagrams/index.md", "w") as f:
    f.write("# Diagrams\n\n")
    for folder in sorted(by_folder):
        f.write(f"- [{folder}](./{folder}.d2)\n")
```

* The **mkdocs-d2-plugin** compiles `.d2` → SVG during the build, and `link:` makes nodes clickable to your module pages (navigation is two-way when module pages link back to their diagram). ([LandMaj][5])

---

# 5) API page (ReDoc), deep-links, and hiding chrome on that page

Create `docs/api/index.md`:

```markdown
---
hide:
  - navigation
  - toc
---

# HTTP API

<redoc src="../../openapi/openapi.yaml"/>
```

* The **mkdocs-redoc-tag** plugin adds the `<redoc …/>` tag, bundles the assets for offline use, and syncs dark mode with Material. ([GitHub][6])
* You can **deep-link** to specific operations with `#operation/<operationId>` (that’s exactly what the generator writes under “Related API operations”). ([GitHub][16])

---

# 6) Authoring tips & cross-refs (humans + agents)

* Use **autorefs**: you can link to any heading by **title** (no need to know the page URL) — super helpful for generated content. ([PyPI][4])
* Material initializes **Mermaid** for fenced `mermaid` code blocks; if you enable mkdocstrings’ inheritance diagrams (Insiders), they render as Mermaid too. ([Squidfunk][9])
* If you want a **source-code link** per symbol that jumps to GitHub lines instead of inlined source, add **mkdocstrings-sourcelink** (optional). ([AI2 Business][17])

---

# 7) Make it agent-friendly

Add some high-level breadcrumbs for agents:

* In `mkdocs.yml`, keep **clear nav** (Overview → Architecture → API → Modules → Diagrams).
* Enable `/llms.txt` by setting `site_url` and optionally `sections:` in the `llmstxt` plugin; you can even output a “full” variant with page content if you want. ([GitHub][12])

---

# 8) Run it

```bash
mkdocs serve            # dev server with live reload
mkdocs build --strict   # CI-grade build: fail on warnings
```

MkDocs’ `--strict` **aborts on warnings** — that’s what you want in CI. ([MkDocs][13])

---

## Extra notes for your repo

1. **Material repo integration**
   Make sure `repo_url` and `edit_uri` are set. Typical GitHub pattern is `repo_url: https://github.com/<org>/<repo>` and `edit_uri: edit/main/` (Material’s `content.action.*` then shows the buttons). Our generator maps each module page’s **Edit** to the **code file** via `set_edit_path`. ([Squidfunk][8])

2. **Griffe details**

* Griffe exposes `is_imported`/`is_exported` and module `exports` derived from `__all__` — that’s what powers the *Imports / Imported by / Exports* blocks. ([mkdocstrings][18])
* The convenience `griffe.load()` API takes `search_paths` and loads submodules; we used it instead of the older `GriffeLoader` import. ([mkdocstrings][14])

3. **D2 plugin**
   Install the D2 CLI (>= 0.6.3) on your build host; the plugin shells out to it when compiling diagrams. ([LandMaj][5])

4. **mkdocstrings options**
   `show_source`, `inherited_members`, signature options, etc., are part of the Python handler; *inheritance diagrams* are Insiders. ([mkdocstrings][10])

5. **ReDoc deep links**
   We link into ReDoc with `#operation/<operationId>`. Keep your `operationId`s URL-safe (good practice anyway). ([GitHub][16])

---

## What you’ll see after first build

* **Overview** and **Architecture** are your hand-written story.
* **API** is a full ReDoc UI, with deep links from module pages to operations.
* **Modules** lists one page per module with:

  * **Relationships** (Imports / Imported by / Exports) as active links
  * All **classes & functions** rendered via mkdocstrings (with source if enabled)
  * Optional **Related API operations** block that deep-links into ReDoc
* **Diagrams** shows **folder clusters**; each node is clickable and points to the module page; module pages link back — two-way navigation.

---

## (Optional) One small convenience: map code → API ops

If you want the *Related API operations* block to be auto-filled, drop a tiny JSON next to the generator:

```jsonc
// docs/_scripts/api_usage.json
{
  "your_package.client.Client": ["getPetById", "createOrder"],
  "your_package.orders.service.place_order": ["createOrder"]
}
```

You can also generate this file from comments/annotations or from endpoint wrappers in your code — up to you.

---

## Done ✅

This stack gives you:

* **Layered, clickable knowledge** (Overview → Architecture → Folders → Modules → Symbols)
* **Cross-linked API & code** (ReDoc deep links + module pages)
* **Diagrams with two-way nav**, plus **Edit/View Source** everywhere
* **Agent-ready surface** with an `/llms.txt` map, consistent anchors, and strict builds

If you share your top-level package name and repo URL, I’ll tailor the two generator scripts to your exact paths and, if you have mkdocstrings Insiders, switch on inheritance diagrams automatically.

[1]: https://www.mkdocs.org/getting-started/?utm_source=chatgpt.com "Getting Started"
[2]: https://mkdocstrings.github.io/python/usage/?utm_source=chatgpt.com "Usage - mkdocstrings-python"
[3]: https://oprypin.github.io/mkdocs-gen-files/index.html "Manual - mkdocs-gen-files"
[4]: https://pypi.org/project/mkdocs-autorefs/0.5.0/?utm_source=chatgpt.com "mkdocs-autorefs"
[5]: https://landmaj.github.io/mkdocs-d2-plugin/?utm_source=chatgpt.com "mkdocs-d2-plugin"
[6]: https://github.com/blueswen/mkdocs-redoc-tag "GitHub - blueswen/mkdocs-redoc-tag: A MkDocs plugin supports adding Redoc to the page."
[7]: https://www.mkdocs.org/user-guide/writing-your-docs/?utm_source=chatgpt.com "Writing Your Docs"
[8]: https://squidfunk.github.io/mkdocs-material/setup/adding-a-git-repository/?utm_source=chatgpt.com "Adding a git repository - Material for MkDocs - GitHub Pages"
[9]: https://squidfunk.github.io/mkdocs-material/reference/diagrams/?utm_source=chatgpt.com "Diagrams - Material for MkDocs - GitHub Pages"
[10]: https://mkdocstrings.github.io/python/reference/api/?utm_source=chatgpt.com "API reference - mkdocstrings-python"
[11]: https://github.com/mkdocs/mkdocs-redirects?utm_source=chatgpt.com "Open source plugin for Mkdocs page redirects"
[12]: https://github.com/pawamoy/mkdocs-llmstxt "GitHub - pawamoy/mkdocs-llmstxt: MkDocs plugin to generate an /llms.txt file (https://llmstxt.org/)."
[13]: https://www.mkdocs.org/user-guide/cli/?utm_source=chatgpt.com "Command Line Interface"
[14]: https://mkdocstrings.github.io/griffe/reference/api/loaders/ "Loaders - Griffe"
[15]: https://mkdocstrings.github.io/python/usage/configuration/general/?utm_source=chatgpt.com "General - mkdocstrings-python"
[16]: https://github.com/Redocly/redoc/issues/1625?utm_source=chatgpt.com "support operationRef or operationId references in the ..."
[17]: https://ai2business.github.io/mkdocstrings-sourcelink/ "Introduction - mkdocstrings-sourcelink"
[18]: https://mkdocstrings.github.io/griffe/guide/users/navigating/?utm_source=chatgpt.com "Navigating - Griffe - mkdocstrings"
