Awesome—here’s a concrete, repo-ready plan that turns your “best-in-class navmap” spec into working code for **kgfoundry**. It’s split into small, drop-in files plus tiny edits you can make quickly.

---

# 1) Add typed navmap primitives (one tiny module)

**Create** `src/kgfoundry_common/navmap_types.py`:

```python
from __future__ import annotations
from typing import Final, Literal, NotRequired, TypedDict

class NavSection(TypedDict):
    id: str                 # kebab-case slug, e.g. "public-api"
    title: str
    symbols: list[str]      # Python identifiers in this module

class SymbolMeta(TypedDict, total=False):
    since: str
    stability: Literal["frozen", "stable", "experimental", "internal"]
    side_effects: list[Literal["none", "fs", "net", "gpu", "db"]]
    thread_safety: Literal["reentrant", "threadsafe", "not-threadsafe"]
    async_ok: bool
    perf_budget_ms: float
    tests: list[str]                 # pytest nodeids/markers
    replaced_by: NotRequired[str]
    deprecated_msg: NotRequired[str]
    contracts: NotRequired[list[str]]
    coverage_target: NotRequired[float]

class NavMap(TypedDict, total=False):
    title: str
    synopsis: str
    exports: list[str]               # MUST mirror __all__
    sections: list[NavSection]
    see_also: list[str]
    tags: list[str]
    since: str
    deprecated: str
    symbols: dict[str, SymbolMeta]
    edit_scopes: dict[str, list[str]]   # {"safe": [...], "risky": [...]}
    deps: list[str]                      # runtime deps (strings)
```

Also add `py.typed` at the package roots you ship (at minimum):
`touch src/kgfoundry_common/py.typed src/kgfoundry/py.typed`

---

# 2) Instrument one module (template you can copy)

Example for `src/embeddings_sparse/bm25.py` (apply similarly elsewhere):

```python
# imports…

__all__ = ["SparseBM25", "build_bm25", "BM25Error"]

# --- Machine-parseable navmap (typed) ---------------------------------------
from typing import Final
from kgfoundry_common.navmap_types import NavMap

__navmap__: Final[NavMap] = {
    "title": "kgfoundry.embeddings_sparse.bm25",
    "synopsis": "Sparse BM25 helpers (build/search).",
    "exports": __all__,
    "sections": [
        {"id": "public-api", "title": "Public API",
         "symbols": ["SparseBM25", "build_bm25", "BM25Error"]},
        {"id": "internals", "title": "Internals",
         "symbols": ["_tokenize", "_bm25_matrix"]},
    ],
    "symbols": {
        "build_bm25": {
            "stability": "stable", "since": "2025.10",
            "side_effects": ["fs"], "async_ok": False,
            "tests": ["tests/smoke/test_search_api_smoke.py::test_bm25_index_smoke"],
            "perf_budget_ms": 200.0,
        },
    },
    "edit_scopes": {"safe": ["build_bm25"], "risky": ["_bm25_matrix"]},
    "tags": ["bm25", "sparse", "pyserini"],
    "since": "2025.10",
}
# ---------------------------------------------------------------------------

# [nav:anchor SparseBM25]
class SparseBM25:  # docstring (Google/NumPy style)
    ...

# [nav:anchor build_bm25]
def build_bm25(...):
    ...

class BM25Error(Exception):
    """Raised when BM25 construction fails."""

# [nav:section internals]
# region internals
def _tokenize(...): ...
def _bm25_matrix(...): ...
# endregion
```

**Rules to follow per file**

* `__all__` + `__navmap__` live right after imports.
* First section is always `"public-api"`.
* Add `# [nav:anchor <Symbol>]` above each exported class/func; `# [nav:section <slug>]` marks foldable blocks.

---

# 3) Generator: build a single JSON index (+ links)

**Create** `tools/navmap/build_navmap.py`:

```python
#!/usr/bin/env python
from __future__ import annotations
import ast, json, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
OUT = REPO / "site" / "_build" / "navmap"
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]")
SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]")

@dataclass
class ModuleInfo:
    path: Path
    mod: str
    all_: list[str]
    navmap: dict[str, Any]
    anchors: dict[str, int]      # symbol -> line (1-based)
    sections: dict[str, int]     # slug -> line

def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()

def parse_module(py: Path) -> ModuleInfo | None:
    text = py.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py))
    mod = ".".join(rel(py)[:-3].split("/")[1:]).replace("/",".")

    all_vals: list[str] = []
    navmap_dict: dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        all_vals = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__navmap__":
                    # Allow "exports: __all__" and literal dict everywhere else
                    src = ast.get_source_segment(text, node.value) or "{}"
                    # crude but effective: replace __all__ with literal list before eval
                    src = src.replace("__all__", json.dumps(all_vals))
                    navmap_dict = eval(src, {"__builtins__": {}})  # only literals post-replacement

    # discover anchors & sections
    anchors, sections = {}, {}
    for i, line in enumerate(text.splitlines(), start=1):
        if m := ANCHOR_RE.match(line):
            anchors[m.group(1)] = i
        if m := SECTION_RE.match(line):
            sections[m.group(1)] = i

    return ModuleInfo(path=py, mod=mod, all_=all_vals, navmap=navmap_dict,
                      anchors=anchors, sections=sections)

def collect() -> dict[str, Any]:
    modules: list[ModuleInfo] = []
    for py in SRC.rglob("*.py"):
        if py.name in {"__init__.py"}:  # still allowed; keep scanning if you want
            pass
        info = parse_module(py)
        if info:
            modules.append(info)

    sha = git_sha()
    out: dict[str, Any] = {"commit": sha, "modules": {}}

    for m in modules:
        nm = m.navmap or {}
        nm_exports = nm.get("exports", m.all_)
        out["modules"][m.mod] = {
            "path": rel(m.path),
            "exports": nm_exports,
            "sections": nm.get("sections", []),
            "anchors": m.anchors,   # {symbol -> line}
            "links": {
                "github": {
                    s: f"https://github.com/paul-heyse/kgfoundry/blob/{sha}/{rel(m.path)}#L{ln}"
                    for s, ln in m.anchors.items()
                },
                # VS Code / Cursor relative links (portable):
                "vscode": {s: f"{rel(m.path)}:{ln}" for s, ln in m.anchors.items()},
            },
            "meta": nm.get("symbols", {}),
            "tags": nm.get("tags", []),
            "synopsis": nm.get("synopsis", ""),
        }
    return out

def main() -> None:
    data = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "navmap.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {rel(OUT / 'navmap.json')} @ {data['commit']}")

if __name__ == "__main__":
    main()
```

---

# 4) Validator: enforce invariants (CI + pre-commit)

**Create** `tools/navmap/check_navmap.py`:

```python
#!/usr/bin/env python
from __future__ import annotations
import ast, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

def inspect_file(py: Path):
    text = py.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py))

    all_vals, navmap = set(), None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        all_vals = {elt.s for elt in node.value.elts if isinstance(elt, ast.Str)}
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__navmap__":
                    src = ast.get_source_segment(text, node.value) or "{}"
                    src = src.replace("__all__", str(list(all_vals)))
                    navmap = eval(src, {"__builtins__": {}})

    errs = []
    if navmap:
        exports = set(navmap.get("exports", []))
        if exports != all_vals:
            errs.append(f"{py}: exports mismatch (__all__ vs __navmap__): {sorted(all_vals)} != {sorted(exports)}")

        sections = navmap.get("sections", [])
        if sections:
            if sections[0].get("id") != "public-api":
                errs.append(f"{py}: first section must be 'public-api'")
            for sec in sections:
                sid = sec.get("id", "")
                if not SLUG.match(sid):
                    errs.append(f"{py}: bad section slug '{sid}'")
                for sym in sec.get("symbols", []):
                    # presence will be checked by anchor pass below
                    if not re.match(r"^[A-Za-z_]\w*$", sym):
                        errs.append(f"{py}: invalid symbol name '{sym}'")

    # anchors present?
    anchors = set()
    for i, line in enumerate(text.splitlines(), start=1):
        m = re.match(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]", line)
        if m:
            anchors.add(m.group(1))

    missing = all_vals.difference(anchors)
    if missing:
        errs.append(f"{py}: missing [nav:anchor] for exported symbols: {sorted(missing)}")

    return errs

def main() -> int:
    errors = []
    for py in SRC.rglob("*.py"):
        errors.extend(inspect_file(py))
    if errors:
        print("\n".join(errors))
        return 1
    print("navmap check: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

# 5) Sphinx: add stable source links (GitHub + line ranges)

In `site/conf.py`, enable **linkcode** and point to GitHub using the current commit:

```python
extensions = [
    # ...
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",       # Google/NumPy docstrings
]

import inspect
from pathlib import Path
import importlib
import subprocess
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        src, start = inspect.getsourcelines(obj)
        end = start + len(src) - 1
        rel = Path(fn).resolve().relative_to(REPO_ROOT).as_posix()
        return f"https://github.com/paul-heyse/kgfoundry/blob/{COMMIT}/{rel}#L{start}-L{end}"
    except Exception:
        return None
```

(If you prefer to drive links from `site/_build/navmap/navmap.json`, load that JSON and look up the symbol → line number instead of using `inspect`.)

---

# 6) Project config (Ruff, doctest, pytest)

**pyproject.toml** additions (Google docstrings; tweak if you prefer NumPy):

```toml
[tool.ruff.lint]
select = ["E","F","I","B","UP","D","RUF"]
ignore = ["D203","D213"]                        # common preferences

[tool.ruff.lint.pydocstyle]
convention = "google"
```

**pytest.ini** (enable doctests so examples double as smoke tests):

```ini
[pytest]
addopts = --doctest-modules
```

---

# 7) Pre-commit & CI wiring

**.pre-commit-config.yaml** – add local hooks:

```yaml
- repo: local
  hooks:
  - id: navmap-check
    name: navmap-check
    entry: python tools/navmap/check_navmap.py
    language: system
    types: [python]
  - id: navmap-build
    name: navmap-build
    entry: python tools/navmap/build_navmap.py
    language: system
    pass_filenames: false
```

**.github/workflows/ci.yml** – add steps before tests:

```yaml
- name: Build navmap
  run: python tools/navmap/build_navmap.py

- name: Check navmap
  run: python tools/navmap/check_navmap.py
```

---

# 8) Makefile / tasks (optional but handy)

```make
navmap-build:
\tpython tools/navmap/build_navmap.py

navmap-check:
\tpython tools/navmap/check_navmap.py
```

(Your workspace already has tasks—add two entries mirroring these if you want one-click runs.)

---

# 9) README generator (keep links rock-solid)

If your `tools/gen_readmes.py` emits “open”/“view” links, use the navmap JSON to build:

* local **open** links: `path:line` (VS Code/Cursor colon syntax)
* GitHub **view** links: `.../blob/<sha>/<path>#L<line>`

This avoids hard-coding line numbers in READMEs and stays stable after refactors.

---

## How to roll this out (order of ops)

1. Add `navmap_types.py` + `py.typed` files.
2. Pick 1–2 modules (e.g., `embeddings_sparse/bm25.py`, `search_api/service.py`) and add `__navmap__` + `# [nav:anchor]` comments.
3. Run `python tools/navmap/build_navmap.py` → confirm `site/_build/navmap/navmap.json`.
4. Run `python tools/navmap/check_navmap.py` → fix any mismatches.
5. Enable Sphinx `linkcode` and build docs.
6. Wire pre-commit and CI steps.

If you want, I can open a PR that adds the new `tools/navmap/*` scripts, Sphinx snippet, and a first pass of `__navmap__`+anchors for a couple of modules so you’ve got a working end-to-end example inside `kgfoundry`.
