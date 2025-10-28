Here’s a **clear, end-to-end narrative** of what changes land in the repo to implement recommendations **2–6**, how they fit together, what artifacts they produce, and how AI programming agents (and humans) will use them. I’ll reference concrete file paths and named functions/config fields so it’s obvious where each piece lives, but I’ll keep everything non-code as requested.

---

# Big picture

We’re adding five documentation capabilities that turn your docs from “static reference” into a **living, execution-aware map** of the codebase:

1. **Executable docs** — docstrings are smoke-tested and a small, curated **Examples Gallery** is rendered into the docs site without executing GPU/IO at Sphinx time.

2. **Code↔Tests cross-links** — a generated index answers “which tests touch this symbol?” and exposes that both as a docs page and as machine-readable JSON.

3. **Observability catalog** — static scans enumerate logging events, metrics, traces, and config surfaces with file:line breadcrumbs. These tables appear in the docs and as JSON.

4. **Schema docs** — all Pydantic/Pandera models are exported to JSON Schema and published in the docs; CI checks drift so new contracts can’t sneak in undocumented.

5. **Import & call-graphs** — package-level import graphs and UML-style class diagrams are built to SVG and embedded in the docs.

All five capabilities plug into your existing Sphinx pipeline (see `docs/conf.py`) and your docs orchestrator script. They create a repeatable flow: **generate → build → gate**. Agents benefit because each artifact is small, structured, and cross-linked to source, so they can jump directly to the right file and line.

---

# 2) Executable docs: doctest + examples gallery

## What changes in the repo

* **New authoring surface** under `examples/` for tiny, hermetic scripts (e.g., `examples/00_quickstart.py`, `examples/20_search_smoke.py`). These are written specifically to be import-safe, CPU-only, and sub-second where possible.
* **Docstring smoke tests** are enabled via pytest’s xdoctest integration; this runs your **inline `>>>` examples** embedded in code docstrings.
* **Gallery integration** is switched on in `docs/conf.py` (via the `sphinx_gallery_conf` section). The gallery **renders** example pages into `docs/_build/html/gallery/` and auto-adds “download as .py/.ipynb” buttons, but does **not execute** examples during Sphinx (execution happens in the doctest phase instead).

## How it runs

* Locally and in CI, the docs orchestrator (`tools/update_docs.sh`) runs the doctest smoke first, then builds Sphinx. That ensures failing examples never reach publication.
* GPU/IO or non-hermetic lines are explicitly marked with doctest skip directives in the authoring files, so the smoke remains fast and deterministic.
* The gallery pages automatically back-link to module pages (AutoAPI) because `doc_module=("kgfoundry", …)` is set in the Sphinx config.

## Why agents care

* Agents can **see and retrieve runnable minimal examples** alongside the API reference without accidentally triggering GPUs or network. The “download” affordances let them spin local experiments easily.
* When a symbol’s docstring example fails, CI stops the docs build. That failure is **immediately actionable** at the symbol level.

---

# 3) Code ↔ test cross-links (“who tests what?”)

## What changes in the repo

* A generator script (`tools/docs/build_test_map.py`) walks `tests/` and **heuristically maps symbols to tests** by scanning Python imports, attribute references, and exact mentions.
* The output is written to `docs/_build/test_map.json` (machine), and a new docs page at `docs/reference/test-matrix.md` **includes** that JSON so it renders as a table within the site.
* Optionally, your symbol index builder (if you maintain `docs/_scripts/build_symbol_index.py`) now **enriches each symbol entry** with a `tested_by` field by loading the test map JSON.

## How it runs

* The generator is called **before** Sphinx in `tools/update_docs.sh`. That way the Test Matrix page and any symbol enrichment are available during the build.
* There’s an optional CI guard that scans the test map after generation and **warns/fails** when too many public surfaces appear unreferenced in tests (a tuneable threshold and allowlist).

## Why agents care

* On any API page, the agent can see **nearby test files** that reference that symbol (and line hints). This is the shortest path to trustworthy examples and assertions.
* Machine consumers (your agents) can also load `docs/_build/test_map.json` to find the right test surfaces to update when proposing changes.

---

# 4) Observability catalog (metrics, logs, traces, config)

## What changes in the repo

* A scanner (`tools/docs/scan_observability.py`) runs static AST/regex passes over `src/` to collect:

  * **Log events**: file, line, level, and the initial message template.
  * **Metrics**: Prometheus or OTel counter/gauge/histogram declarations.
  * **Traces**: spans and attribute/event calls.
  * **Configuration surfaces**: locations that declare or use Pydantic `BaseSettings` (or equivalent).
* It emits machine outputs to `docs/_build/metrics.json`, `docs/_build/log_events.json`, `docs/_build/traces.json`, plus a human summary `docs/_build/config.md`.
* New docs pages under `docs/reference/observability/` **include** those files verbatim (rendered as tables/blocks) and, thanks to your existing linkcode setup in `docs/conf.py`, users can click through to **exact file/line** for each row.

## How it runs

* The scanner runs **before** Sphinx in `tools/update_docs.sh` so the data exists when pages are built.
* The docs build publishes these as part of your Reference section so they’re always current with the codebase.

## Why agents care

* When tasks involve reliability, performance, or ops behavior, agents can scan a **canonical list of metrics/logs/traces** and jump to the exact call site to modify semantics, labels, or error handling. It cuts out code spelunking.

---

# 5) Schema docs for contracts (Pydantic/Pandera → JSON Schema)

## What changes in the repo

* An exporter (`tools/docs/export_schemas.py`) imports the top-level packages (`kgfoundry`, `kgfoundry_common`, `kg_builder`, etc.), finds all **Pydantic models** and **Pandera schema models**, and **writes a JSON Schema file per model** under `docs/reference/schemas/` (e.g., `kgfoundry.api.Request.json`).
* The docs include a **Schemas** index (`docs/reference/schemas/index.md`) that toctrees all generated `.json`. If you prefer the richer `jsonschema` directive, that remains compatible.
* A **contract validity test** in `tests/docs/` ensures every exported schema passes Draft 2020-12 validation.
* CI adds a **drift check**: if models change but schema files weren’t regenerated, the docs job fails with a clear message.

## How it runs

* The exporter is invoked **before** Sphinx in `tools/update_docs.sh`. Sphinx then renders the schemas into the site.
* The drift check runs in CI after export to catch missed updates in PRs.

## Why agents care

* Agents get **machine-precise contracts** for inputs/outputs in one place. This lets them propose changes in API layers, validation paths, or serialization code with confidence.
* Drift checks prevent “silent contract shifts” that would otherwise confuse downstream tooling or clients.

---

# 6) Import graphs & UML diagrams

## What changes in the repo

* A builder (`tools/docs/build_graphs.py`) discovers top-level packages in `src/`, then:

  * Runs **pydeps** to produce a package import graph (DOT), converts it to **SVG**, and stores it in `docs/_build/graphs/<pkg>-imports.svg`.
  * Runs **pyreverse** (from pylint) to emit **UML-style class diagrams** to `docs/_build/graphs/<pkg>-uml.svg`.
* A new page at `docs/reference/graphs/index.md` embeds the generated SVGs.

## How it runs

* The builder runs **before** Sphinx in `tools/update_docs.sh`.
* CI ensures the system dependency `graphviz` is installed so DOT→SVG conversion works. The Sphinx build simply embeds the pre-rendered SVG (so Sphinx itself doesn’t need graphviz).

## Why agents care

* For non-trivial packages, being able to **glance at the import surface** and the **main class relationships** is the fastest way to orient before proposing invasive refactors or hooking into the right seams.

---

# Where everything integrates

## Orchestration order

1. **Doctest** (fast smoke) — prevents the docs build from publishing stale or broken examples.
2. **Cross-links** (test map) — the JSON and Test Matrix page are generated.
3. **Observability** — metrics/logs/traces/config outputs are generated.
4. **Schemas** — JSON Schema is exported (CI drift guard installed).
5. **Graphs** — SVGs produced for imports/UML.
6. **Sphinx build** — HTML and JSON builders consume all the above and produce a site that includes the new reference sections and gallery.

This ordering guarantees **inputs exist before rendering**, and that **gates fire** (doctest, drift checks) before anything renders or ships.

## Pre-commit vs CI

* **Pre-commit** runs a quick doctest and the fast generators (test map, observability, schemas) so contributors get immediate feedback without waiting on the full CI.
* **CI** runs the complete orchestrator, installs system deps (graphviz), validates JSON Schemas, and performs the drift check. If anything changes the docs artifacts without committing updates (or a schema drifts), the job fails.

## Sphinx navigation

* The main index (`docs/index.md`) gains two new navigation clusters:

  * **Reference** → Test Matrix, Observability (metrics/logs/traces/config), Schemas, Graphs.
  * **Gallery** → the examples gallery index.
* Your existing configuration (AutoAPI, Napoleon, intersphinx, linkcode, theme) remains intact; these additions augment the nav without disrupting your current structure.

---

# Developer & agent workflows

## Developer mental model

* **Create** or update a symbol → write/update its docstring example → ensure it’s tiny and hermetic → run doctest locally.
* **Touch code paths** that affect contracts → run the schema exporter; commit any changed JSON.
* **Add metrics/logging** → run the observability scanner; confirm they appear in the docs.
* **Add tests** for new public symbols → confirm they show up in the Test Matrix.
* **Touch package structure** → regenerate graphs; inspect the new import/UML maps to ensure dependency health.

## Agent mental model

* Land on any API page → follow the **“Tested by”** references to the closest assertions → open the source link to the exact file/line.
* If the task is reliability-focused → open **Observability** pages to see log/metric/trace surfaces and jump to the callsite.
* If the task is data-shape-focused → open **Schemas** to get the exact contract JSON.
* If the task is architectural → open **Graphs** to see import seams and main class relationships.
* If the task is example-driven → open **Gallery** for copy-pasteable, minimal “how-to” snippets.

---

# Operational notes & safeguards

* **Performance & determinism**: doctest runs before Sphinx and is the only place code executes; docs renders do not execute examples. Heavy or non-hermetic lines are explicitly marked to be skipped in doctest, keeping the smoke fast, stable, and CPU-only.
* **Safety gates**: doctest failures block docs publish; schema drift blocks merges (if enabled); optional “untested public surfaces” gate reduces the risk of undocumented/untested expansion.
* **Cross-link fidelity**: your existing linkcode settings in `docs/conf.py` ensure file:line links resolve correctly from observability rows and module pages.
* **Style coherence**: the gallery and doctest align with your ongoing move to **NumPy-style docstrings** (see `docs/explanations/numpy-docstring-migration.md`), so examples in docstrings and gallery entries reflect the same conventions.
* **Generated artifact hygiene**: everything large or ephemeral lands under `docs/_build/…`; you keep those paths uncommitted (unless you intentionally commit schemas), and CI’s “docs drift” check ensures sources and artifacts stay in sync.

---

# What “done” looks like (at a glance)

* The docs site now includes:

  * **Gallery** of example pages with download buttons.
  * **Test Matrix** that maps symbols to tests.
  * **Observability** tables (metrics, logs, traces, config).
  * **Schemas** (one JSON per data model) under a dedicated index.
  * **Graphs** (imports and UML) per top-level package.
* The CI “Docs” job:

  * Runs doctest first (fast), then all generators, then Sphinx HTML/JSON.
  * Fails on schema drift or missing updates to generated artifacts.
* Agents and maintainers can:

  * Hop from any symbol to its tests, logs, metrics, or schema.
  * See and download runnable, minimal examples without triggering heavy paths.
  * Understand structural dependencies rapidly via graphs.





Here’s a repo-ready, fully integrated plan to ship **recommendations 2–6** in **kgfoundry**. I based this on your current automated docs pipeline (Sphinx HTML+JSON, navmap build/check, symbol index, pre-commit/CI hooks) and wired each recommendation so it drops cleanly into what you already have. 

---

# Executable docs (xdoctest + sphinx-gallery)

## What you’ll ship

* **Doctest on docstrings** (fast smoke): `pytest --xdoctest`, GPU/IO marked `+SKIP`.
* **Examples gallery** at `/examples` → rendered into docs as HTML pages + downloadable notebooks.
* **Strict but safe gates** in pre-commit + CI.

## Files & patches

### 1) `pyproject.toml` (deps + pytest config)

```toml
[project.optional-dependencies]
docs = [
  "sphinx>=7.3",
  "myst-parser",
  "pydata-sphinx-theme",
  "sphinxcontrib-mermaid",
  "sphinx-jsonschema",
  "xdoctest>=1.1.4",
  "sphinx-gallery>=0.17.1",
  # graphs (used later but install with docs)
  "pydeps>=1.12.20",
  "pylint>=3.2",            # for pyreverse
  "graphviz>=0.20.3",
]

[tool.pytest.ini_options]
addopts = "-q"
# Run doctests in docstrings; examples directory is handled by sphinx-gallery
xdoctest_optionflags = "ELLIPSIS IGNORE_WHITESPACE NORMALIZE_WHITESPACE"
```

> If you already have `[tool.pytest.ini_options]`, merge `addopts` and the `xdoctest_*` lines.

### 2) Examples layout

```
examples/
  00_quickstart.py
  10_data_contracts_minimal.py
  20_hybrid_search_smoke.py
  _utils.py               # lightweight helpers (no network, no GPU)
```

**Authoring rules (docstring header in each example):**

```python
"""
Title: Quickstart — build & search a tiny index
Tags: getting-started, smoke
Time: <10s
GPU: no
Network: no
"""
# doctest: +SKIP  ← put on any line that would pull GPUs/IO or take >~2s
```

### 3) `docs/conf.py` (enable gallery)

Add to your existing Sphinx config:

```python
extensions += [
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../..", "examples"][1],   # repo root relative to docs/conf.py
    "gallery_dirs": "gallery",                   # docs/_build/html/gallery/*
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "backreferences_dir": "gen_modules/backrefs",
    "doc_module": ("kgfoundry",),
    "run_stale_examples": False,
    # Don't execute examples in Sphinx; we only render; doctest is run by pytest.
    "plot_gallery": False,
}
```

> Your Sphinx already ships with AutoAPI, MyST, linkcode, graphviz/mermaid—leave those as-is. 

### 4) Pre-commit hook (quick doctest smoke)

In `.pre-commit-config.yaml` (order it after ruff/mypy):

```yaml
-   repo: local
    name: doctest (xdoctest via pytest)
    entry: bash -lc 'pytest -q --xdoctest -k ""'
    language: system
    pass_filenames: false
    always_run: true
```

### 5) CI: ensure doctest + gallery build

Add to your docs workflow:

```yaml
- name: Doctest smoke (xdoctest)
  run: pytest -q --xdoctest

- name: Build docs (HTML+JSON+gallery)
  run: |
    tools/update_docs.sh   # your orchestrator already builds html+json
```

> Keep gallery **non-executing** in Sphinx; examples execute only under pytest/xdoctest, which honors `+SKIP`.

### 6) `tools/update_docs.sh` (callout)

Add a tiny gate just before Sphinx:

```bash
run "$BIN/pytest" -q --xdoctest
```

**Definition of Done**

* `pytest -q --xdoctest` passes locally/CI.
* `docs/_build/html/gallery/*` exists; pages link to raw `.py` and `.ipynb` (auto by sphinx-gallery).
* Any GPU/IO example lines are marked `# doctest: +SKIP`.

---

# Code ↔ test cross-links (“who tests what?”)

## What you’ll ship

* A machine map `docs/_build/test_map.json`: **symbol → {tests, line ranges}**.
* A human page `/reference/test-matrix.html`.
* (Optional) Add `tested_by` into your `symbols.json` for agents.

## Files & patches

### 1) `tools/docs/build_test_map.py`

Static analysis (no imports): discover references by AST+string heuristics.

```python
from __future__ import annotations
import ast, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
OUT = ROOT / "docs" / "_build" / "test_map.json"

def find_symbols():
    # naive inventory: fully-qualified names from symbols.json if present; else from src tree
    syms = []
    sym_index = ROOT / "docs" / "_build" / "symbols.json"
    if sym_index.exists():
        for row in json.loads(sym_index.read_text()):
            syms.append(row["path"])
    else:
        for py in SRC.rglob("*.py"):
            mod = ".".join(py.relative_to(SRC).with_suffix("").parts)
            syms.append(mod)  # minimal; functions/classes resolved by usage below
    return sorted(set(syms))

def scan_tests(symbols):
    out = {s: [] for s in symbols}
    pattern = re.compile(r"([a-zA-Z_][\w\.]+)")
    for tfile in TESTS.rglob("test_*.py"):
        code = tfile.read_text("utf-8", errors="ignore")
        # quick hits (module.function or from module import name)
        hits = {m.group(1) for m in pattern.finditer(code)}
        for s in symbols:
            base = s.split(".")[0]
            if s in hits or base in hits:
                # find line spans (rough): where the symbol string occurs
                lines = []
                for i, ln in enumerate(code.splitlines(), 1):
                    if s in ln or (("." in s) and s.split(".")[-1] in ln):
                        lines.append(i)
                if lines:
                    out[s].append({"file": str(tfile.relative_to(ROOT)), "lines": lines[:5]})
    # drop empties
    return {k: v for k, v in out.items() if v}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    syms = find_symbols()
    OUT.write_text(json.dumps(scan_tests(syms), indent=2), "utf-8")

if __name__ == "__main__":
    main()
```

### 2) Sphinx page that renders the matrix

`docs/reference/test_matrix.md`:

````md
# Test Matrix (symbol → tests)

```{include} ../_build/test_map.json
:literal:
````

````

### 3) (Optional) enrich the symbol index
If you want agents to see tests right next to each symbol:
- Modify `docs/_scripts/build_symbol_index.py` after it collects each symbol:
```python
test_map = json.loads(Path("docs/_build/test_map.json").read_text()) if Path("docs/_build/test_map.json").exists() else {}
entry["tested_by"] = test_map.get(entry["path"], [])
````

### 4) Orchestration & CI

* In `tools/update_docs.sh`, right before Sphinx:

```bash
run "$BIN/python" tools/docs/build_test_map.py
run "$BIN/python" docs/_scripts/build_symbol_index.py  # rebuild symbols to include tested_by
```

**Definition of Done**

* `docs/_build/test_map.json` exists with hits for public symbols.
* New page `/reference/test-matrix.html` renders.
* (Optional) `docs/_build/symbols.json` entries carry `"tested_by"`.

---

# Observability catalog (metrics/logs/traces/config)

## What you’ll ship

* Machine files:

  * `docs/_build/metrics.json`
  * `docs/_build/log_events.json`
* Human pages:

  * `/reference/observability/metrics.md`
  * `/reference/observability/logs.md`
  * `/reference/observability/config.md`

## Files & patches

### 1) `tools/docs/scan_observability.py`

Lightweight AST/regex sweep for Prometheus/OpenTelemetry/logging and Pydantic Settings:

```python
from __future__ import annotations
import ast, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"
OUT.mkdir(parents=True, exist_ok=True)

METRICS, LOGS = [], []
CONFIG_LINES = []

def scan_file(py: Path):
    code = py.read_text("utf-8", errors="ignore")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        # logging.getLogger("name") → logger.name; logger.info("msg …")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # logs
            if node.func.attr in {"debug","info","warning","error","exception","critical"}:
                LOGS.append({
                    "file": str(py),
                    "lineno": node.lineno,
                    "level": node.func.attr,
                    "message_template": ast.get_source_segment(code, node.args[0]) if node.args else None
                })
        # OpenTelemetry (Counter/Histogram) or prometheus_client.* instruments
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            tgt = getattr(node.value.func, "attr", "")
            mod = getattr(getattr(node.value.func, "value", None), "id", "")
            if mod in {"prometheus_client"} or tgt in {"create_counter","create_histogram","Counter","Histogram","Gauge","UpDownCounter"}:
                METRICS.append({
                    "file": str(py),
                    "lineno": node.lineno,
                    "decl": ast.get_source_segment(code, node.value)[:200],
                })

    # Pydantic BaseSettings, Pandera schemas in plain text (quick links)
    if "BaseSettings" in code or "pandera" in code:
        CONFIG_LINES.append({"file": str(py)})

def main():
    for py in SRC.rglob("*.py"):
        scan_file(py)
    (OUT / "metrics.json").write_text(json.dumps(METRICS, indent=2), "utf-8")
    (OUT / "log_events.json").write_text(json.dumps(LOGS, indent=2), "utf-8")
    (OUT / "config.md").write_text(
        "# Config surfaces (quick index)\n\n" +
        "\n".join(f"- `{c['file']}`" for c in CONFIG_LINES),
        "utf-8"
    )

if __name__ == "__main__":
    main()
```

### 2) Reference pages

`docs/reference/observability/metrics.md`

````md
# Metrics (static scan)

```{include} ../../_build/metrics.json
:literal:
````

````

`docs/reference/observability/logs.md`
```md
# Log events (static scan)

```{include} ../../_build/log_events.json
:literal:
````

````

> Your Sphinx already resolves deep links to source via linkcode; keep that. :contentReference[oaicite:2]{index=2}

### 3) Orchestration & CI
Add before Sphinx build:
```bash
run "$BIN/python" tools/docs/scan_observability.py
````

**Definition of Done**

* JSONs exist, pages render, and rows link back to source via existing linkcode.

---

# Schema docs for contracts (Pydantic/Pandera → JSON Schema)

## What you’ll ship

* Exported JSON Schema files under `docs/reference/schemas/*.json`.
* Rendered schema reference pages (via **sphinx-jsonschema**).
* CI drift check: fail if models ≠ exported schemas.

## Files & patches

### 1) `tools/docs/export_schemas.py`

```python
from __future__ import annotations
import importlib, inspect, json, pkgutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PKG = "kgfoundry"  # adjust if your top-level package differs

OUT = ROOT / "docs" / "reference" / "schemas"
OUT.mkdir(parents=True, exist_ok=True)

def is_pydantic(cls):
    try:
        from pydantic import BaseModel
        return inspect.isclass(cls) and issubclass(cls, BaseModel)
    except Exception:
        return False

def is_pandera(cls):
    try:
        import pandera as pa
        return inspect.isclass(cls) and issubclass(cls, pa.SchemaModel)
    except Exception:
        return False

def iter_modules(package):
    mod = importlib.import_module(package)
    for m in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
        yield m.name

def main():
    for mod_name in iter_modules(PKG):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in vars(mod).items():
            if is_pydantic(obj):
                try:
                    schema = obj.model_json_schema()
                    (OUT / f"{obj.__module__}.{name}.json").write_text(json.dumps(schema, indent=2), "utf-8")
                except Exception:
                    pass
            if is_pandera(obj):
                try:
                    schema = obj.to_schema().to_json()
                    (OUT / f"{obj.__module__}.{name}.json").write_text(schema, "utf-8")
                except Exception:
                    pass

if __name__ == "__main__":
    main()
```

### 2) Render page

`docs/reference/schemas/index.md`

````md
# Data Contract Schemas

```{toctree}
:maxdepth: 1
:glob:

*.json
````

````
(and ensure `sphinx-jsonschema` is in `extensions` or use `{jsonschema}` directives as desired.)

### 3) Drift check (CI)
Add after schema export:
```bash
# normalize (jq) and compare when a golden directory exists
git diff --exit-code docs/reference/schemas || (echo "Schemas drifted; run tools/docs/export_schemas.py" && exit 1)
````

**Definition of Done**

* Schema files generated and included in docs build.
* CI fails when models change but schemas weren’t regenerated.

---

# Import & call-graph pages (map the terrain)

## What you’ll ship

* Import graphs per-package (from **pydeps**) and UML class diagrams (from **pyreverse**).
* A `/reference/graphs/*` section in the docs.

## Files & patches

### 1) `tools/docs/build_graphs.py`

```python
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd): subprocess.run(cmd, check=True)

def main():
    pkgs = {p.parts[0] for p in (p.relative_to(SRC) for p in SRC.rglob("__init__.py")) if len(p.parts) >= 1}
    for pkg in sorted(pkgs):
        # pydeps → import graph
        run([sys.executable, "-m", "pydeps", f"src/{pkg}", "--max-bacon=4", "--show-dot", "--noshow"])
        dot = Path(f"{pkg}.dot")
        if dot.exists():
            run(["dot", "-Tsvg", str(dot), "-o", str(OUT / f"{pkg}-imports.svg")])
            dot.unlink()
        # pyreverse (pylint) → UML
        run([sys.executable, "-m", "pylint.pyreverse", f"src/{pkg}", "-o", "svg", "-p", pkg])
        for svg in Path(".").glob("classes_*.svg"):
            svg.rename(OUT / f"{pkg}-uml.svg")
        for junk in Path(".").glob("*.dot"): junk.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
```

### 2) Docs page

`docs/reference/graphs/index.md`

````md
# Package Graphs

Below are generated import and class diagrams.

```{image} ../../_build/graphs/kgfoundry-imports.svg
:alt: Imports graph
:width: 100%
````

```{image} ../../_build/graphs/kgfoundry-uml.svg
:alt: UML diagram
:width: 100%
```

````

> Your Sphinx already enables `sphinx.ext.graphviz`; these SVGs are pre-rendered so you don’t need graphviz at build time, only at generation time. :contentReference[oaicite:3]{index=3}

### 3) Orchestration & CI
Run *before* Sphinx build:
```bash
run "$BIN/python" tools/docs/build_graphs.py
````

On CI (Ubuntu), install system deps:

```yaml
- name: System deps for graphs
  run: sudo apt-get update && sudo apt-get install -y graphviz
```

**Definition of Done**

* `docs/_build/graphs/*svg` exist and render under `/reference/graphs/`.

---

# Orchestrator & Makefile integration (single-command, deterministic)

Add these calls to **`tools/update_docs.sh`** (sequence relative to your existing stages):

```
# after docstrings & readmes & navmap build/check …
run "$BIN/pytest" -q --xdoctest
run "$BIN/python" tools/docs/build_test_map.py
run "$BIN/python" tools/docs/scan_observability.py
run "$BIN/python" tools/docs/export_schemas.py
run "$BIN/python" tools/docs/build_graphs.py
# existing:
run "$BIN/python" -m sphinx -b html docs docs/_build/html
run "$BIN/python" -m sphinx -b json docs docs/_build/json
run "$BIN/python" docs/_scripts/build_symbol_index.py
```

> This plugs directly into your **master docs script** alongside Sphinx HTML+JSON and symbol index builds. 

Add convenient Make targets:

```make
doctest:
	pytest -q --xdoctest

test-map:
	python tools/docs/build_test_map.py

obs-catalog:
	python tools/docs/scan_observability.py

schemas:
	python tools/docs/export_schemas.py

graphs:
	python tools/docs/build_graphs.py
```

---

# CI wiring (minimal deltas)

In your existing docs job:

```yaml
- name: Install docs extras
  run: uv pip install -e ".[docs]"

- name: System deps (graphs)
  run: sudo apt-get update && sudo apt-get install -y graphviz

- name: Full docs pipeline
  run: tools/update_docs.sh

- name: Ensure docs are in sync
  run: git diff --exit-code docs/ || (echo "Docs drifted"; exit 1)
```

---

# Pre-commit (fast feedback)

Append to `.pre-commit-config.yaml` (after lint/format/type steps you already run):

```yaml
- repo: local
  name: docs: build test map
  entry: python tools/docs/build_test_map.py
  language: system
  pass_filenames: false
  always_run: true

- repo: local
  name: docs: observability scan
  entry: python tools/docs/scan_observability.py
  language: system
  pass_filenames: false
  always_run: true

- repo: local
  name: docs: export schemas
  entry: python tools/docs/export_schemas.py
  language: system
  pass_filenames: false
  always_run: true
```

---

# “Repo check” note

I verified that **`README-AUTOMATED-DOCUMENTATION.md`** exists in `paul-heyse/kgfoundry` (your docs pipeline description), and the plan above nests neatly into that flow (Sphinx HTML+JSON, navmap build/check, symbol index, single orchestrator). The patches here assume that structure and only add new generation steps + Sphinx config to surface them. (If you’d like, I can open a branch with these exact file drops next.) 

---

## Acceptance checklist (for all 5 recommendations)

* [ ] `pytest -q --xdoctest` passes; GPU/IO examples are `+SKIP`.
* [ ] Gallery pages appear under `/gallery/*` with downloads.
* [ ] `docs/_build/test_map.json` exists and `/reference/test-matrix.html` renders.
* [ ] `docs/_build/metrics.json`, `docs/_build/log_events.json` exist; observability pages render.
* [ ] `docs/reference/schemas/*.json` generated and included; CI drift check in place.
* [ ] `docs/_build/graphs/*svg` exist and `/reference/graphs/index.html` renders.
* [ ] `tools/update_docs.sh` runs end-to-end locally and in CI with a clean `git diff`.



# ADDENDUM — Fully-specified implementation artifacts for recommendations 2–6

> This addendum provides the missing *concrete, repo-ready* details (precise files, code, config, commands, gating, and tests) to implement: **(2) executable docs**, **(3) code↔tests cross-links**, **(4) observability catalog**, **(5) schema docs**, **(6) import & call-graphs**. It assumes your repo has `docs/conf.py` and a Sphinx pipeline already producing HTML/JSON (and that `site/_build/` exists). The snippets below are drop-in additions or safe merges.

---

## 0) Shared conventions & environment (applies to all 2–6)

**Pins & env toggles**

* Python: 3.11+/3.12+/3.13 (CI matrix).
* Env for safe generation:

  * `AGENT_DOCS_FAST=1` (skip heavy steps)
  * `CUDA_VISIBLE_DEVICES=""`
  * `TOKENIZERS_PARALLELISM=false`
  * `PYTHONWARNINGS=ignore`
* System packages (CI): `graphviz` (for SVG rendering of .dot), `jq` (diff checks).
* Use `uv` for installs (aligns with your toolchain):

  * `uv pip install -e ".[docs]"` for local dev and CI docs job.

**`pyproject.toml` additions (merge these blocks)**

```toml
[project.optional-dependencies]
docs = [
  "sphinx>=7.3",
  "myst-parser>=3.0.1",
  "pydata-sphinx-theme>=0.16.1",
  "sphinx-jsonschema>=1.19.3",
  "sphinxcontrib-mermaid>=1.0.0",
  # Rec 2
  "xdoctest>=1.1.4",
  "sphinx-gallery>=0.17.1",
  # Rec 6
  "pydeps>=1.12.20",
  "pylint>=3.2.0",        # for pyreverse
  "graphviz>=0.20.3",
  # Rec 5 (validation)
  "jsonschema>=4.23.0",
  "pydantic>=2.8.0",      # if not already present
  "pandera>=0.20.0",      # if you use pandera
]

[tool.pytest.ini_options]
addopts = "-q"
# run doctests in code docstrings (xdoctest parses '>>>')
xdoctest_optionflags = "ELLIPSIS IGNORE_WHITESPACE NORMALIZE_WHITESPACE"
# fail fast for CI
# (optional) add: " -x" to addopts if you prefer
```

**Makefile additions (merge as targets)**

```make
.PHONY: doctest test-map obs-catalog schemas graphs docs-json docs-html

doctest:
	pytest -q --xdoctest

test-map:
	python tools/docs/build_test_map.py

obs-catalog:
	python tools/docs/scan_observability.py

schemas:
	python tools/docs/export_schemas.py

graphs:
	python tools/docs/build_graphs.py

docs-json:
	python -m sphinx -b json docs docs/_build/json

docs-html:
	python -m sphinx -b html docs docs/_build/html
```

**Docs orchestrator (append calls in order) — `tools/update_docs.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
BIN="${BIN:-python}"

# 1) fast smoke of doctests in code docstrings
$BIN -m pytest -q --xdoctest

# 2) generate cross-links (symbol → tests)
$BIN tools/docs/build_test_map.py

# 3) observability catalog (metrics/logs/config)
$BIN tools/docs/scan_observability.py

# 4) export JSON Schemas (Pydantic/Pandera)
$BIN tools/docs/export_schemas.py

# 5) graphs (imports & UML) → SVG
$BIN tools/docs/build_graphs.py

# 6) build Sphinx (HTML & JSON); gallery is configured in conf.py
$BIN -m sphinx -b html docs docs/_build/html
$BIN -m sphinx -b json docs docs/_build/json
```

**Pre-commit additions (append to `.pre-commit-config.yaml`)**

```yaml
- repo: local
  name: doctest (xdoctest via pytest)
  entry: bash -lc 'pytest -q --xdoctest'
  language: system
  pass_filenames: false
  always_run: true

- repo: local
  name: docs: build test map
  entry: python tools/docs/build_test_map.py
  language: system
  pass_filenames: false
  always_run: true

- repo: local
  name: docs: observability scan
  entry: python tools/docs/scan_observability.py
  language: system
  pass_filenames: false
  always_run: true

- repo: local
  name: docs: export schemas
  entry: python tools/docs/export_schemas.py
  language: system
  pass_filenames: false
  always_run: true
```

**CI (docs job) — add or extend `/.github/workflows/docs.yml` job**

```yaml
name: Docs
on:
  push:
    branches: [main]
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

jobs:
  docs:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    env:
      AGENT_DOCS_FAST: "1"
      CUDA_VISIBLE_DEVICES: ""
      TOKENIZERS_PARALLELISM: "false"
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }

      - uses: astral-sh/setup-uv@v4
      - name: Install system deps
        run: sudo apt-get update && sudo apt-get install -y graphviz jq

      - name: Install docs extras
        run: uv pip install -e ".[docs]"

      - name: Build complete docs pipeline
        run: bash tools/update_docs.sh

      - name: Ensure docs artifacts committed or ignored
        run: |
          git update-index --refresh
          git diff --exit-code docs/ || (echo "::error::Docs drifted; run tools/update_docs.sh locally and commit." && exit 1)

      - name: Upload built docs as artifact (optional)
        uses: actions/upload-artifact@v4
        with:
          name: docs-build
          path: docs/_build/html
```

---

## 2) Executable docs — xdoctest + sphinx-gallery

**`docs/conf.py` updates (exact lines to add/merge)**

```python
# -- Extensions ----------------------------------------------------------------
extensions = list(set(extensions + [
    "sphinx_gallery.gen_gallery",  # gallery pages
    # (keep your existing: 'myst_parser', 'sphinx.ext.napoleon', 'autoapi.extension', 'sphinx.ext.linkcode', etc.)
]))

# -- Gallery config -------------------------------------------------------------
# Resolve path to examples dir from this conf.py
from pathlib import Path as _P
_EXAMPLES_DIR = str((_P(__file__).resolve().parents[1] / "examples").resolve())

sphinx_gallery_conf = {
    "examples_dirs": _EXAMPLES_DIR,        # source scripts
    "gallery_dirs": "gallery",             # output under docs/_build/html/gallery
    "filename_pattern": r".*\\.py",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "doc_module": ("kgfoundry",),          # module linking for backrefs
    "run_stale_examples": False,
    "plot_gallery": False,                 # do not execute in Sphinx; execution goes via pytest/xdoctest
}
```

**Examples layout and minimal, hermetic examples**

```
examples/
  00_quickstart.py
  10_data_contracts_minimal.py
  20_search_smoke.py
  _utils.py
```

**`examples/_utils.py`**

```python
# Tiny helpers designed to be import-safe and hermetic in CI.
def tiny_corpus():
    return [
        {"id": "1", "text": "cats like naps"},
        {"id": "2", "text": "dogs enjoy walks"},
        {"id": "3", "text": "birds can fly"},
    ]
```

**`examples/00_quickstart.py`**

```python
"""
Title: Quickstart — minimal import smoke test
Tags: getting-started, smoke
Time: <2s
GPU: no
Network: no
"""

def show_version():
    """
    >>> import importlib
    >>> m = importlib.import_module("kgfoundry")
    >>> hasattr(m, "__version__")
    True
    """
    pass
```

**`examples/10_data_contracts_minimal.py`**

```python
"""
Title: Data contracts — schema export smoke
Tags: schema, pydantic
Time: <2s
GPU: no
Network: no
"""
# doctest: +SKIP  # remove SKIP once the model class is available at import time

def render_schema():
    """
    >>> from pydantic import BaseModel
    >>> class Mini(BaseModel):
    ...     a: int
    ...     b: str | None = None
    >>> s = Mini.model_json_schema()
    >>> "properties" in s
    True
    """
    pass
```

**`examples/20_search_smoke.py`**

```python
"""
Title: Search — tiny corpus smoke (no GPU)
Tags: search, smoke
Time: <2s
GPU: no
Network: no
"""
# doctest: +SKIP  # unskip when a CPU-only search path is import-safe and <2s

def tiny_search():
    """
    >>> from examples._utils import tiny_corpus
    >>> corpus = tiny_corpus()
    >>> len(corpus) > 0
    True
    """
    pass
```

**Doctest directives for heavy lines**

* Add `# doctest: +SKIP` to any statement that would:

  * allocate GPU,
  * perform network I/O,
  * take >2 seconds.

**Local run**

```bash
uv pip install -e ".[docs]"
pytest -q --xdoctest
python -m sphinx -b html docs docs/_build/html
```

---

## 3) Code ↔ test cross-links

**Generator — `tools/docs/build_test_map.py`**

```python
from __future__ import annotations
import ast, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
OUT = ROOT / "docs" / "_build" / "test_map.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

PKG_PREFIXES = ("kgfoundry", "kgfoundry_common", "kg_builder", "search_api", "embeddings_dense", "embeddings_sparse", "ontology", "orchestration")

def _symbol_candidates() -> set[str]:
    cands: set[str] = set()
    for py in SRC.rglob("*.py"):
        mod = ".".join(py.relative_to(SRC).with_suffix("").parts)
        if mod.startswith(PKG_PREFIXES):
            cands.add(mod)
    return cands

def _scan_one_test(path: Path, symbols: set[str]):
    txt = path.read_text("utf-8", errors="ignore")
    try:
        tree = ast.parse(txt)
    except SyntaxError:
        return {}

    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: names.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module)
        elif isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name):
                names.add(f"{n.value.id}.{n.attr}")
        elif isinstance(n, ast.Name):
            names.add(n.id)

    hits = {}
    for s in symbols:
        top = s.split(".")[0]
        if s in txt or top in names or s in names:
            # collect a few line refs for context
            lines = []
            for i, line in enumerate(txt.splitlines(), 1):
                if s in line or line.strip().startswith(("from ", "import ")) and top in line:
                    lines.append(i)
                    if len(lines) >= 5: break
            hits.setdefault(s, []).append({"file": str(path.relative_to(ROOT)), "lines": lines})
    return hits

def main():
    symbols = _symbol_candidates()
    cross = {}
    for t in TESTS.rglob("test_*.py"):
        for k, v in _scan_one_test(t, symbols).items():
            cross.setdefault(k, []).extend(v)
    OUT.write_text(json.dumps(cross, indent=2), "utf-8")

if __name__ == "__main__":
    main()
```

**Sphinx page — `docs/reference/test-matrix.md`**

````md
# Test Matrix (symbol → tests)

```{include} ../_build/test_map.json
:literal:
````

````

**(Optional) enrich symbol index (if you maintain `docs/_scripts/build_symbol_index.py`)**
```python
from pathlib import Path
import json

# ... after building 'entries' list of symbols:
test_map_path = Path("docs/_build/test_map.json")
test_map = json.loads(test_map_path.read_text()) if test_map_path.exists() else {}
for e in entries:
    e["tested_by"] = test_map.get(e["path"], [])
````

**CI guard (new public symbols referenced in tests) — add to docs job**

```bash
python tools/docs/build_test_map.py
python - << 'PY'
import json, sys
tm = json.load(open("docs/_build/test_map.json"))
# Heuristic: fail if there are 5+ public modules with zero tests pointing at them.
untested = [k for k,v in tm.items() if not v]
if len(untested) > 5:
    print("::warning::Public symbols appear untested:", untested[:10])
    sys.exit(1)
PY
```

**Local run**

```bash
make test-map
python -m sphinx -b html docs docs/_build/html
```

---

## 4) Observability catalog (metrics / logs / traces / config)

**Generator — `tools/docs/scan_observability.py`**

```python
from __future__ import annotations
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"
OUT.mkdir(parents=True, exist_ok=True)

metrics, logs, traces, configs = [], [], [], []

def parse_file(p: Path):
    code = p.read_text("utf-8", errors="ignore")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    # logging.* calls
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            level = n.func.attr
            if level in {"debug","info","warning","error","exception","critical"}:
                msg = None
                if n.args: msg = ast.get_source_segment(code, n.args[0])
                logs.append({
                    "file": str(p), "lineno": n.lineno, "level": level,
                    "message_template": (msg or "").strip()[:240],
                })
    # prometheus_client and OTel metrics
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            fn = n.value.func
            mod = None
            name = None
            if isinstance(fn, ast.Attribute):
                name = fn.attr
                if isinstance(fn.value, ast.Name):
                    mod = fn.value.id
            elif isinstance(fn, ast.Name):
                name = fn.id
            if (mod == "prometheus_client") or (name in {"Counter","Gauge","Histogram","Summary","UpDownCounter"}):
                metrics.append({
                    "file": str(p), "lineno": n.lineno,
                    "decl": (ast.get_source_segment(code, n.value) or "").strip()[:240],
                })
    # traces: span starts and attribute sets
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in {"start_as_current_span","start_span","set_attribute","add_event"}:
                traces.append({
                    "file": str(p), "lineno": n.lineno,
                    "call": (ast.get_source_segment(code, n) or "").strip()[:240],
                })
    # Config: Pydantic BaseSettings or env usage
    if "BaseSettings" in code or "pydantic_settings" in code:
        configs.append({"file": str(p)})

def main():
    for py in SRC.rglob("*.py"):
        parse_file(py)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    (OUT / "log_events.json").write_text(json.dumps(logs, indent=2), "utf-8")
    (OUT / "traces.json").write_text(json.dumps(traces, indent=2), "utf-8")
    # minimal config page
    (OUT / "config.md").write_text(
        "# Config surfaces (quick index)\n\n" +
        "\n".join(f"- `{c['file']}`" for c in configs),
        "utf-8"
    )

if __name__ == "__main__":
    main()
```

**Docs pages**
`docs/reference/observability/metrics.md`

````md
# Metrics (static scan)

```{include} ../../_build/metrics.json
:literal:
````

````
`docs/reference/observability/logs.md`
```md
# Log events (static scan)

```{include} ../../_build/log_events.json
:literal:
````

````
`docs/reference/observability/traces.md`
```md
# Traces (static scan)

```{include} ../../_build/traces.json
:literal:
````

````
`docs/reference/observability/config.md`
```md
# Configuration surfaces (Pydantic Settings & env usage)

```{include} ../../_build/config.md
````

````

**Local run**
```bash
make obs-catalog
make docs-html
````

---

## 5) Schema docs for contracts (Pydantic / Pandera → JSON Schema)

**Exporter — `tools/docs/export_schemas.py`**

```python
from __future__ import annotations
import importlib, inspect, json, pkgutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TOP_PACKAGES = ("kgfoundry", "kgfoundry_common", "kg_builder")

OUT = ROOT / "docs" / "reference" / "schemas"
OUT.mkdir(parents=True, exist_ok=True)

def is_pydantic_model(x):
    try:
        from pydantic import BaseModel
        return inspect.isclass(x) and issubclass(x, BaseModel)
    except Exception:
        return False

def is_pandera_model(x):
    try:
        import pandera as pa
        return inspect.isclass(x) and issubclass(x, pa.SchemaModel)
    except Exception:
        return False

def walk(pkgs):
    for pkg in pkgs:
        try:
            m = importlib.import_module(pkg)
        except Exception:
            continue
        if not hasattr(m, "__path__"):  # skip non-packages
            continue
        for info in pkgutil.walk_packages(m.__path__, prefix=pkg + "."):
            yield info.name

def main():
    exported = 0
    for modname in walk(TOP_PACKAGES):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in vars(mod).items():
            if is_pydantic_model(obj):
                try:
                    schema = obj.model_json_schema()
                    (OUT / f"{obj.__module__}.{name}.json").write_text(json.dumps(schema, indent=2), "utf-8")
                    exported += 1
                except Exception:
                    pass
            elif is_pandera_model(obj):
                try:
                    schema = obj.to_schema().to_json()
                    (OUT / f"{obj.__module__}.{name}.json").write_text(schema, "utf-8")
                    exported += 1
                except Exception:
                    pass
    print(f"exported={exported}")

if __name__ == "__main__":
    main()
```

**Render index — `docs/reference/schemas/index.md`**

````md
# Data Contract Schemas

```{toctree}
:maxdepth: 1
:glob:
*.json
````

````

**Schema validity test — `tests/docs/test_schemas_valid.py`**
```python
import json, pathlib
import jsonschema

SCHEMAS = pathlib.Path("docs/reference/schemas")

def test_all_jsonschemas_are_valid():
    for p in SCHEMAS.glob("*.json"):
        data = json.loads(p.read_text())
        jsonschema.Draft202012Validator.check_schema(data)
````

**CI drift check (append to docs job)**

```bash
python tools/docs/export_schemas.py
git diff --quiet -- docs/reference/schemas || (echo "::error::Schema files drifted; run export_schemas.py and commit." && exit 1)
```

**Local run**

```bash
make schemas
pytest -q tests/docs/test_schemas_valid.py
```

---

## 6) Import graphs (pydeps) & UML (pyreverse)

**Builder — `tools/docs/build_graphs.py`**

```python
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd): subprocess.run(cmd, check=True)

def packages():
    # infer top-level packages under src/
    pkgs = set()
    for init in SRC.rglob("__init__.py"):
        parts = init.relative_to(SRC).parts
        if parts: pkgs.add(parts[0])
    return sorted(pkgs)

def main():
    for pkg in packages():
        modpath = f"src/{pkg}"
        # pydeps: import graph (DOT -> SVG)
        run([sys.executable, "-m", "pydeps", modpath, "--max-bacon=4", "--show-dot", "--noshow"])
        dot = Path(f"{pkg}.dot")
        if dot.exists():
            run(["dot", "-Tsvg", str(dot), "-o", str(OUT / f"{pkg}-imports.svg")])
            dot.unlink(missing_ok=True)

        # pyreverse: UML classes (SVG)
        run([sys.executable, "-m", "pylint.pyreverse", modpath, "-o", "svg", "-p", pkg])
        for svg in Path(".").glob("classes_*.svg"):
            svg.rename(OUT / f"{pkg}-uml.svg")
        for junk in Path(".").glob("*.dot"): junk.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
```

**Docs page — `docs/reference/graphs/index.md`**

````md
# Package Graphs

Below are generated import graphs and UML diagrams.

```{toctree}
:maxdepth: 1
:glob:
*.md
````

```{image} ../../_build/graphs/kgfoundry-imports.svg
:alt: kgfoundry import graph
:width: 100%
```

```{image} ../../_build/graphs/kgfoundry-uml.svg
:alt: kgfoundry UML
:width: 100%
```

````

**CI system deps (already in job)**
```yaml
- name: System deps for graphs
  run: sudo apt-get update && sudo apt-get install -y graphviz
````

**Local run**

```bash
make graphs
make docs-html
```

---

## 7) Sphinx: ensure pages are included in nav

**`docs/index.md` (add toctrees if not already present)**

````md
# kgfoundry Documentation

```{toctree}
:maxdepth: 2
:caption: Reference

reference/test-matrix
reference/observability/metrics
reference/observability/logs
reference/observability/traces
reference/observability/config
reference/schemas/index
reference/graphs/index
````

```{toctree}
:maxdepth: 1
:caption: Gallery

gallery/index
```

````

---

## 8) Smoke tests for the pipeline

**`tests/docs/test_pipeline_artifacts.py`**
```python
from pathlib import Path
import json

def test_test_map_exists_and_nonempty():
    p = Path("docs/_build/test_map.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert isinstance(data, dict)

def test_observability_jsons_exist():
    base = Path("docs/_build")
    for name in ["metrics.json", "log_events.json", "traces.json"]:
        p = base / name
        assert p.exists(), f"missing {name}"

def test_graph_svgs_exist():
    g = Path("docs/_build/graphs")
    assert any(g.glob("*-imports.svg")), "no import graphs present"
    assert any(g.glob("*-uml.svg")), "no UML graphs present"
````

Run:

```bash
pytest -q tests/docs/test_pipeline_artifacts.py
```

---

## 9) Developer UX (local commands)

**Local one-liners**

```bash
# Full docs refresh (all generators + Sphinx)
uv pip install -e ".[docs]" && bash tools/update_docs.sh

# Just rebuild gallery + HTML
python -m sphinx -b html docs docs/_build/html

# Quick doctest smoke
pytest -q --xdoctest
```

---

## 10) Windows/macOS notes (if contributors use them)

* **Graphviz**

  * macOS: `brew install graphviz`
  * Windows (Admin PowerShell): `choco install graphviz` and ensure `dot.exe` is on `PATH`.
* **Paths** in `docs/conf.py` use `pathlib` — cross-platform safe.
* **No GPU imports**: keep examples/docstrings guarded by `# doctest: +SKIP` or lazy imports.

---

## 11) Git hygiene for generated files

* Ensure `docs/_build/**` is **not** accidentally committed unless your workflow requires publishing built HTML in-repo.
* Recommended `.gitignore` entries (append if needed):

```
docs/_build/
gallery/
*.dot
*.svg~
```

---

## 12) Quick verification checklist

* `pytest -q --xdoctest` passes locally.
* `tools/update_docs.sh` completes; produces:

  * `docs/_build/test_map.json`
  * `docs/_build/{metrics.json,log_events.json,traces.json}`
  * `docs/reference/schemas/*.json`
  * `docs/_build/graphs/*-imports.svg` and `*-uml.svg`
  * `docs/_build/html/gallery/*`
* `make docs-html` shows new nav sections:

  * **Reference →** Test Matrix, Observability (3 pages), Schemas, Graphs
  * **Gallery →** index + examples
* CI `Docs` job completes; fails if schemas drift or if docs drift from committed state (if you choose to commit artifacts).

