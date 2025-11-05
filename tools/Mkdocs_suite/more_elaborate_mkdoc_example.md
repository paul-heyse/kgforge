Absolutely—here’s a “best-in-class” **one-page-per-module** setup that adds everything you asked for:

* **Mermaid inheritance** (inline on each module page)
* **Autorefs** examples (cross-page, cross-symbol)
* **API linkage** (deep-link to ReDoc operations)
* **D2 diagrams** that are clickable back to documentation (two-way connection)
* **GitHub-ready**: MkDocs Material config + GitHub Pages workflow

I’ll give you: (1) the `mkdocs.yml` you can drop in, (2) an upgraded module-page generator that emits Mermaid + inline D2, (3) a tiny `api_usage.json` example, (4) the ReDoc page, and (5) a GH Actions workflow.

---

# 1) `mkdocs.yml` (Material + mkdocstrings + autorefs + D2 + ReDoc)

````yaml
site_name: Your Project
site_url: https://<your-gh-username>.github.io/<your-repo>/
repo_url: https://github.com/<your-gh-username>/<your-repo>
edit_uri: edit/main/          # maps to docs/ by default; we’ll remap generated pages
strict: true

theme:
  name: material
  features:
    - navigation.sections
    - navigation.tracking
    - navigation.top
    - content.action.edit
    - content.action.view
    - content.code.copy

markdown_extensions:
  - admonition
  - attr_list
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.superfences        # enables Mermaid fences
  - pymdownx.snippets

plugins:
  - search
  - autorefs                    # lets [SymbolName][] cross-link across pages
  - section-index
  - d2                          # renders ```d2``` or .d2 files into SVG
  - redoc-tag                   # <redoc src="..."/>
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_signature: true
            inherited_members: true
  - gen-files:
      scripts:
        - docs/_scripts/gen_module_pages.py   # (included below)

nav:
  - Overview: index.md
  - API:
      - api/index.md            # ReDoc (loads docs/api/openapi.yaml)
  - Modules:
      - modules/index.md
  - Diagrams:
      - diagrams/index.md
````

> Material handles Mermaid automatically via `pymdownx.superfences`.
> `autorefs` lets you write `[your_package.mod.ClassName][]` and it resolves to the `mkdocstrings` section for that symbol.

---

# 2) Upgraded generator: **Mermaid + Autorefs + API links + inline D2**

Save as **`docs/_scripts/gen_module_pages.py`**.

````python
# docs/_scripts/gen_module_pages.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json
import textwrap

import griffe
import mkdocs_gen_files

# === CONFIG ===
PACKAGE = "your_package"             # top-level import name
SEARCH_PATHS = ["src", "."]          # adjust for your repo
MAX_REL_NEIGHBORS = 12               # how many neighbors to show per module
# Optional: map code -> API operations for deep links to ReDoc
# docs/_scripts/api_usage.json: {"your_package.modA.func_x": ["getPetById"], ...}
API_USAGE_FILE = Path(__file__).with_name("api_usage.json")

# === Load optional code->API map ===
api_usage: dict[str, list[str]] = {}
if API_USAGE_FILE.exists():
    api_usage = json.loads(API_USAGE_FILE.read_text())

# === Walk package with griffe ===
root = griffe.load(PACKAGE, search_paths=SEARCH_PATHS, submodules=True)
modules: dict[str, griffe.Module] = {}

imports = defaultdict(set)     # module -> imported module paths
imported_by = defaultdict(set) # inverse edges
exports = defaultdict(set)     # module -> exported names (__all__)
classes = defaultdict(list)    # module -> class paths
funcs = defaultdict(list)      # module -> function paths
bases = defaultdict(list)      # class path -> list[base path]

def visit(mod: griffe.Module) -> None:
    modules[mod.path] = mod

    # exports from __all__
    if mod.exports:
        for e in mod.exports:
            exports[mod.path].add(str(getattr(e, "name", e)))

    for name, obj in mod.members.items():
        # imports
        if isinstance(obj, griffe.Alias) and obj.is_imported:
            target = getattr(obj, "target_path", None) or getattr(obj, "target", None)
            if target:
                imports[mod.path].add(str(target))

        # symbols
        if isinstance(obj, griffe.Class):
            classes[mod.path].append(obj.path)
            for b in obj.bases:
                if getattr(b, "path", None):
                    bases[obj.path].append(b.path)
        elif isinstance(obj, griffe.Function):
            funcs[mod.path].append(obj.path)

        # descend into submodules
        if isinstance(obj, griffe.Module):
            visit(obj)

visit(root)

# reverse edges
for src, outs in imports.items():
    for dst in outs:
        imported_by[dst].add(src)

def fq_to_short(name: str) -> str:
    """Short class label for diagrams (last segment)."""
    return name.split(".")[-1]

def mermaid_inheritance(mod_path: str) -> str:
    """
    Build a Mermaid classDiagram for classes defined in this module.
    We show both local and external bases, with Base <|-- Derived edges.
    """
    local = set(classes.get(mod_path, []))
    if not local:
        return ""
    lines = ["```mermaid", "classDiagram"]
    declared = set()

    def ensure_decl(n: str):
        if n not in declared:
            lines.append(f'    class {fq_to_short(n)}')
            declared.add(n)

    for cls in sorted(local):
        ensure_decl(cls)
        for b in bases.get(cls, []):
            ensure_decl(b)
            # Mermaid: Base <|-- Derived
            lines.append(f'    {fq_to_short(b)} <|-- {fq_to_short(cls)}')
    lines.append("```")
    return "\n".join(lines)

def inline_d2_neighborhood(mod_path: str) -> str:
    """
    Small D2 graph inline: the module and a limited set of neighbors, clickable.
    Each node links to the corresponding module page.
    """
    outs = sorted(imports.get(mod_path, []))[:MAX_REL_NEIGHBORS//2]
    incs = sorted(imported_by.get(mod_path, []))[:MAX_REL_NEIGHBORS//2]
    if not outs and not incs:
        return ""
    def mlink(m: str) -> str:
        return f'../modules/{m}.md'
    lines = [
        "```d2",
        "direction: right",
        f'"{mod_path}": "{mod_path}" {{ link: "./{mod_path}.md" }}'
    ]
    for m in outs:
        lines.append(f'"{m}": "{m}" {{ link: "{mlink(m)}" }}')
        lines.append(f'"{mod_path}" -> "{m}"')
    for m in incs:
        lines.append(f'"{m}": "{m}" {{ link: "{mlink(m)}" }}')
        lines.append(f'"{m}" -> "{mod_path}"')
    lines.append("```")
    return "\n".join(lines)

# Emit pages
for mod_path, mod in modules.items():
    page_path = f"modules/{mod_path}.md"
    with mkdocs_gen_files.open(page_path, "w") as f:
        # Title
        f.write(f"# {mod_path}\n\n")

        # Summary (first paragraph of module docstring)
        if mod.docstring and mod.docstring.value:
            first_para = mod.docstring.value.strip().split("\n\n")[0]
            f.write(first_para + "\n\n")

        # Relationships
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

        # Inline D2 neighborhood (two-way nav: diagram -> page, and this page links the diagram)
        d2_block = inline_d2_neighborhood(mod_path)
        if d2_block:
            f.write("## Neighborhood (clickable)\n\n")
            f.write(d2_block + "\n\n")
            # Optional “full” .d2 page for this module (write a file too)
            d2_file_path = f"diagrams/modules/{mod_path}.d2"
            with mkdocs_gen_files.open(d2_file_path, "w") as d2f:
                # Slightly larger version with all neighbors
                lines = ["direction: right", f'"{mod_path}": "{mod_path}" {{ link: "../modules/{mod_path}.md" }}']
                for m in sorted(imports.get(mod_path, [])):
                    lines.append(f'"{m}": "{m}" {{ link: "../modules/{m}.md" }}')
                    lines.append(f'"{mod_path}" -> "{m}"')
                for m in sorted(imported_by.get(mod_path, [])):
                    lines.append(f'"{m}": "{m}" {{ link: "../modules/{m}.md" }}')
                    lines.append(f'"{m}" -> "{mod_path}"')
                d2f.write("\n".join(lines))
            f.write(f"> See the full diagram: [{mod_path}](../diagrams/modules/{mod_path}.d2)\n\n")

        # Mermaid inheritance (local class graph)
        mermaid = mermaid_inheritance(mod_path)
        if mermaid:
            f.write("## Inheritance (Mermaid)\n\n")
            f.write(mermaid + "\n\n")

        # API operations this module touches (from api_usage.json)
        used_ops = []
        for sym in classes[mod_path] + funcs[mod_path]:
            used_ops.extend(api_usage.get(sym, []))
        used_ops = sorted(set(used_ops))
        if used_ops:
            f.write("## Related API operations\n\n")
            f.write("These deep-link into the ReDoc page.\n\n")
            f.write(", ".join(f"[{op}](../api/index/#operation/{op})" for op in used_ops) + "\n\n")

        # Contents (mkdocstrings renders sections + anchors)
        if classes[mod_path] or funcs[mod_path]:
            f.write("## Contents\n\n")

        # Autorefs demo: show how to refer to symbols by name (mkdocs-autorefs resolves)
        sample_refs: list[str] = []
        for c in sorted(classes[mod_path])[:3]:
            sample_refs.append(f"[`{c}`][]")
        for fn in sorted(funcs[mod_path])[:3]:
            sample_refs.append(f"[`{fn}`][]")
        if sample_refs:
            f.write("**Autorefs examples:** " + ", ".join(sample_refs) + "\n\n")

        # Render classes/functions with mkdocstrings
        for c in sorted(classes[mod_path]):
            f.write(f"### {c}\n\n::: {c}\n\n")
        for fn in sorted(funcs[mod_path]):
            f.write(f"### {fn}\n\n::: {fn}\n\n")

    # Map Edit button to the actual source file (Material uses edit_uri)
    if getattr(mod, "relative_filepath", None):
        mkdocs_gen_files.set_edit_path(page_path, str(mod.relative_filepath))

# Landing pages
with mkdocs_gen_files.open("modules/index.md", "w") as f:
    f.write("# Modules\n\n")
    for m in sorted(modules):
        f.write(f"- [{m}](./{m}.md)\n")

with mkdocs_gen_files.open("diagrams/index.md", "w") as f:
    f.write("# Diagrams\n\n")
    f.write("Per-module graphs live under this folder.\n\n")
````

What this adds over your baseline:

* **Mermaid inheritance**: builds a `classDiagram` from Griffe’s class bases (`Base <|-- Derived`).
* **Autorefs**: emits sample links like <code>[`your_package.mod.ClassName`][]</code>, which `mkdocs-autorefs` resolves to the mkdocstrings section.
* **API linkage**: reads `api_usage.json` and deep-links to ReDoc sections (`#operation/<operationId>`).
* **Inline D2 neighborhood**: small clickable diagram **inside** the module page + a “full” `.d2` file per module; nodes link back to module pages → **two-way** connection.
* **Edit button mapping**: `set_edit_path` points the page’s **Edit** action to the module’s Python file on GitHub.

---

# 3) Minimal `api_usage.json` (code ↔ API binding)

Save next to the script as **`docs/_scripts/api_usage.json`**:

```json
{
  "your_package.search.client.SearchClient": ["search.execute", "search.suggest"],
  "your_package.ingest.pipeline.run": ["ingest.files"]
}
```

Update the keys to your real class/function paths and the values to your ReDoc operationIds.

---

# 4) ReDoc page (deep-link target for the module pages)

**`docs/api/index.md`**

```markdown
# API Reference

<redoc src="./openapi.yaml"/>
```

Put the bundled OpenAPI file at `docs/api/openapi.yaml`.
(If you’re generating it elsewhere, copy/symlink it here, or set `src="../api/openapi.yaml"` and place it in `docs/api/`.)

---

# 5) GitHub: GitHub Pages workflow (publish on push to `main`)

**`.github/workflows/docs.yml`**

```yaml
name: docs

on:
  push:
    branches: [ main ]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pages: write
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install mkdocs mkdocs-material mkdocs-redoc-tag mkdocs-gen-files mkdocs-typer mkdocs-d2-plugin mkdocstrings mkdocstrings-python

      - name: Sanity check source parses
        run: python -m compileall -q src

      - name: Build site
        run: mkdocs build --strict

      - name: Deploy to gh-pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

This catches syntax errors early (`compileall`), builds with `--strict`, and publishes to `gh-pages`. In your repository settings, enable Pages and point it at the `gh-pages` branch.

---

## How it looks (and why it scales)

* Each **module page** opens with a short summary, a **Relationships** section, a **clickable D2 neighborhood**, and a **Mermaid inheritance** chart for local classes.
* Then **Related API operations** (from `api_usage.json`) deep-link you into ReDoc.
* **Contents** renders every class/function with mkdocstrings (signatures, types, and source links).
* The **Autorefs examples** show how to link to symbols using just their fully-qualified names; you can use the same syntax anywhere in your docs.
* The **inline D2** provides immediate local context; the full `.d2` file per module gives a bigger neighborhood view. Both link **back** to the module pages (two-way).

---

### Tips & tweaks

* If module pages get long, enable Material’s `navigation.tabs` and `toc.integrate` to make in-page navigation persistent.
* For larger inheritance graphs, consider limiting to “local base classes only” or splitting the Mermaid into “internal vs external bases” blocks.
* If you want **per-symbol** D2 nodes linking to class headings, you can add IDs like `(#classname)`—mkdocstrings’ heading IDs are deterministic; if you prefer safety, keep links at the module level (as above).

---

If you tell me your real top-level package name and a couple of representative modules, I’ll pre-fill `PACKAGE`, the `SEARCH_PATHS`, and hand you a small `api_usage.json` that matches your current operations so your first build renders perfectly.
