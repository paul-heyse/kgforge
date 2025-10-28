Absolutely—here’s a **fully-specified, repo-grounded implementation plan** to take your **Symbol Index** from “flat list” to a **best-in-class, agent-ready index** with ownership, stability, sections, test pointers, reverse maps, and CI deltas. I’m basing this on your current stage documentation and the existing script at `docs/_scripts/build_symbol_index.py`.  

---

# What you have today (baseline)

* A single script `docs/_scripts/build_symbol_index.py` that:

  * Loads packages with **Griffe** (no imports).
  * Walks the object tree and emits a **flat array** of rows to `docs/_build/symbols.json`:

    * `path` (FQN), `kind`, `file`, `lineno`, `endlineno`, `doc` (first paragraph).
  * Appends `tested_by` by reading `docs/_build/test_map.json` if present. 

That’s a good spine. The plan below **augments** the index, keeps it deterministic & idempotent, and wires in **reverse maps** and a **delta file** for PRs.

---

# Objectives (definition of done)

1. `symbols.json` entries include **owner, stability, since, deprecated_in, section** (from the NavMap), **tested_by** (from test map), and two deep links (**commit-stable GitHub** + **editor**) for each symbol.
2. Sidecar reverse maps: **`by_file.json`** and **`by_module.json`** for fast “what lives here?” queries.
3. A **delta** report `symbols.delta.json` that highlights **added / removed / changed** symbols (including signature, location, or summary changes) for PR commentary.
4. The builder is **deterministic** (stable sort; write-if-changed).
5. CI **fails** on drift (index out of date).

---

# Data model (authoritative)

## 1) `docs/_build/symbols.json` (flat list)

One JSON object per symbol:

```json
{
  "path": "kgfoundry_common.config.load_config",     // FQN (Griffe Object.path)
  "canonical_path": "kgfoundry_common.config.load_config", // or where alias points (Griffe canonical_path)
  "kind": "function",                                 // module|package|class|function|method|property|attribute|exception
  "package": "kgfoundry_common",                      // top-level package name
  "module": "kgfoundry_common.config",                // module FQN
  "file": "src/kgfoundry_common/config.py",           // repo-relative path (Griffe relative_package_filepath or relative_filepath)
  "lineno": 15,                                       // start line (1-based)
  "endlineno": 25,                                    // end line (1-based)
  "doc": "Load configuration from YAML file.",        // first paragraph only, trimmed
  "signature": "(path: str) -> dict[str, Any]",       // where available (functions/methods); omit if not applicable
  "is_async": false,                                  // optional flags for callables
  "is_property": false,
  "owner": "@core-search",                            // from navmap (fall back to module_meta if defined)
  "stability": "stable",                              // stable|beta|experimental|deprecated
  "since": "0.6.0",                                   // PEP 440 string
  "deprecated_in": null,                              // PEP 440 or null
  "section": "public-api",                            // nav section ID (kebab-case)
  "tested_by": [{"file":"tests/unit/test_config.py","lines":[42,73]}], // from test map (top 1–3)
  "source_link": {
    "github": "https://github.com/<org>/<repo>/blob/<SHA>/src/kgfoundry_common/config.py#L15-L25",
    "editor": "vscode://file/ABS/PATH/src/kgfoundry_common/config.py:15:1"
  }
}
```

**Notes & sources**

* Griffe object location fields `path`, `relative_package_filepath`, `lineno`, `endlineno` are standard and reliable. ([Mkdocstrings][1])
* GitHub line-anchored permalinks use `blob/<SHA>#Lnn` and `#Lnn-Lmm` fragments; this is the documented pattern and ensures links **never rot**. ([GitHub Docs][2])

## 2) `docs/_build/by_file.json` (reverse map)

```json
{
  "src/kgfoundry_common/config.py": [
    "kgfoundry_common.config",
    "kgfoundry_common.config.load_config",
    "kgfoundry_common.config.ConfigSchema"
  ],
  "...": ["..."]
}
```

## 3) `docs/_build/by_module.json` (reverse map)

```json
{
  "kgfoundry_common.config": [
    "kgfoundry_common.config.load_config",
    "kgfoundry_common.config.ConfigSchema"
  ],
  "...": ["..."]
}
```

## 4) `docs/_build/symbols.delta.json` (per PR, optional locally)

```json
{
  "base_sha": "<BASE_SHA>",
  "head_sha": "<HEAD_SHA>",
  "added":    ["kgfoundry_common.config.save_config", "..."],
  "removed":  ["kgfoundry_common.config.LegacyLoader", "..."],
  "changed": [
    {
      "path": "kgfoundry_common.config.load_config",
      "before": {"signature":"(path: str) -> dict", "lineno":15, "doc":"Load..."},
      "after":  {"signature":"(path: str, schema: BaseModel) -> dict", "lineno":18, "doc":"Load and validate..."},
      "reasons": ["signature", "lineno", "doc"]
    }
  ]
}
```

---

# Builder changes (surgical upgrades to your script)

You already do the heavy lifting (load packages, walk tree, write JSON). We’ll **add**:

1. **NavMap fusion** (owner, stability, since, deprecated_in, section): read `site/_build/navmap/navmap.json` and attach per-symbol metadata, inheriting module defaults when present.
2. **Signature capture** using Griffe (`node.signature` when available).
3. **Deep links**: add a GH permalink (`blob/<SHA>#Lstart-Lend`) and an editor link (VS Code URL by default). ([GitHub Docs][2])
4. **Reverse maps**: aggregate while walking; write **by_file.json** and **by_module.json**.
5. **Determinism**: stable sort rows by `path`; write-if-changed to reduce churn.

Below are **drop-in patches** that build on your current structure. (Function names are chosen to co-exist with what you already have.)

---

## A) Read NavMap & GitHub context

```python
# top of docs/_scripts/build_symbol_index.py (after imports)
from typing import Any, Iterable, Dict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"
OUT.mkdir(parents=True, exist_ok=True)

NAVMAP = ROOT / "site" / "_build" / "navmap" / "navmap.json"
TESTMAP = OUT / "test_map.json"

G_ORG  = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA  = os.getenv("DOCS_GITHUB_SHA") or os.popen("git rev-parse HEAD").read().strip()
LINK_MODE = os.getenv("DOCS_LINK_MODE", "both").lower()  # 'github'|'editor'|'both'

def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

NAV = _load_json(NAVMAP)      # navmap.json
TEST = _load_json(TESTMAP)    # test_map.json
```

---

## B) Helpers for GH/editor links & section/meta lookup

```python
def gh_permalink(file_rel_repo: str, start: int | None, end: int | None) -> str | None:
    if not (G_ORG and G_REPO and G_SHA and file_rel_repo):
        return None
    frag = f"#L{start}" if start and not end else (f"#L{start}-L{end}" if start and end else "")
    # Documented pattern for commit-stable permalinks with line anchors:
    # https://docs.github.com/en/.../getting-permanent-links-to-files
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{G_SHA}/{file_rel_repo}{frag}"

def editor_link(abs_path: Path, line: int) -> str:
    # VS Code URL scheme; officially supported:
    # https://code.visualstudio.com/docs/editor/command-line#_opening-vs-code-with-urls
    return f"vscode://file/{abs_path}:{max(1, int(line or 1))}:1"

# Build a module → {meta, sections} index for quick lookups
def _nav_module_index(nav: dict) -> dict[str, dict]:
    mods = {}
    for mod_name, entry in (nav.get("modules") or {}).items():
        mods[mod_name] = {
            "meta": entry.get("meta", {}),
            "sections": entry.get("sections", []),
            "module_meta": entry.get("module_meta", {}),  # if you added defaults in your navmap
            "path": entry.get("path"),
        }
    return mods

NAV_MODS = _nav_module_index(NAV)

def symbol_navmeta(module_fqn: str, short_name: str) -> dict[str, Any]:
    entry = NAV_MODS.get(module_fqn, {})
    meta = (entry.get("meta") or {}).get(short_name, {})
    # inherit from module_meta if missing
    mod_defaults = entry.get("module_meta") or {}
    for k in ("owner","stability","since","deprecated_in"):
        if k not in meta and k in mod_defaults:
            meta[k] = mod_defaults[k]
    # find section containing this symbol
    section_id = None
    for sec in entry.get("sections", []):
        if short_name in (sec.get("symbols") or []):
            section_id = sec.get("id")
            break
    if section_id:
        meta["section"] = section_id
    return meta
```

*(GitHub permalink pattern is cited in docs; line fragments use `#Lnn` and `#Lnn-Lmm`.)* ([GitHub Docs][2])

---

## C) Walking & row assembly (augment your `walk()`)

Key upgrades:

* Calculate `package` and `module` for each `node.path`.
* Capture **signature** for callables (use `getattr(node, "signature", None)`; Griffe exposes it where available).
* Attach **navmap** meta and **test_map** hits.
* Build **source_link** (`github` + `editor`) when environment allows.

```python
by_file: dict[str, list[str]] = {}
by_module: dict[str, list[str]] = {}
rows: list[dict[str, Any]] = []

def _first_para(doc) -> str:
    if not doc or not getattr(doc, "value", None):
        return ""
    return doc.value.split("\n\n", 1)[0].strip()

def _package_and_module(fqn: str) -> tuple[str, str]:
    # fqn = "pkg.mod.Class.meth" → package="pkg", module="pkg.mod"
    parts = fqn.split(".")
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], ".".join(parts[:2]) if len(parts) > 1 and parts[1] else ".".join(parts[:-1])

def _file_repo_relative(node: Object) -> str | None:
    # Prefer Griffe relative_filepath (repo-root relative) then relative_package_filepath
    rfp = safe_attr(node, "relative_filepath")
    if rfp:
        return str(rfp)
    return str(safe_attr(node, "relative_package_filepath") or "")

def walk(node: Object, repo_root: Path) -> None:
    # Griffe facts (path, lineno, endlineno, etc.)
    fqn = node.path
    kind = node.kind.value
    file_rel = _file_repo_relative(node)
    lineno = safe_attr(node, "lineno")
    endlineno = safe_attr(node, "endlineno")
    doc = safe_attr(node, "docstring")
    signature = None
    try:
        sig = safe_attr(node, "signature")
        signature = str(sig) if sig else None
    except Exception:
        signature = None

    # module & short name
    pkg, module = _package_and_module(fqn)
    short = fqn.rsplit(".", 1)[-1]

    # Deep links
    gh = gh_permalink(file_rel, lineno, endlineno) if (LINK_MODE in {"github","both"} and file_rel) else None
    ed = editor_link((repo_root / file_rel).resolve(), lineno or 1) if (LINK_MODE in {"editor","both"} and file_rel) else None

    # NavMap fusion
    navmeta = symbol_navmeta(module, short) if module in NAV_MODS else {}

    # Tests (already in your script; keep + slice top 3)
    tested_by = (TEST.get(fqn, []) if isinstance(TEST, dict) else [])[:3]

    row = {
        "path": fqn,
        "canonical_path": getattr(node, "canonical_path", fqn) if hasattr(node, "canonical_path") else fqn,
        "kind": kind,
        "package": pkg,
        "module": module if module else fqn,
        "file": file_rel or None,
        "lineno": lineno,
        "endlineno": endlineno,
        "doc": _first_para(doc),
        "signature": signature,
        "owner": navmeta.get("owner"),
        "stability": navmeta.get("stability"),
        "since": navmeta.get("since"),
        "deprecated_in": navmeta.get("deprecated_in"),
        "section": navmeta.get("section"),
        "tested_by": tested_by,
        "source_link": {
            **({"github": gh} if gh else {}),
            **({"editor": ed} if ed else {}),
        },
    }
    rows.append(row)

    # Reverse maps
    if file_rel:
        by_file.setdefault(file_rel, []).append(fqn)
    by_module.setdefault(row["module"], []).append(fqn)

    # Recurse
    try:
        for member in list(node.members.values()):
            walk(member, repo_root)
    except Exception:
        pass
```

**Why store both `path` and `canonical_path`?** If a name is an **alias**, Griffe’s `path` is where it is imported, but `canonical_path` is where it originates—handy to de-duplicate re-exports. ([Mkdocstrings][1])

---

## D) Deterministic write & reverse map emission

```python
def _write_json(path: Path, data: Any) -> bool:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True

def _stable_sort():
    rows.sort(key=lambda r: r["path"])
    for key in (by_file, by_module):
        for k, v in key.items():
            v.sort()

# main entry
repo_root = ROOT
for pkg in iter_packages():  # keep your existing detector
    root = loader.load(pkg)
    walk(root, repo_root)

_stable_sort()

changed = False
changed |= _write_json(OUT / "symbols.json", rows)
changed |= _write_json(OUT / "by_file.json", by_file)
changed |= _write_json(OUT / "by_module.json", by_module)
print(f"Wrote {len(rows)} entries to {OUT/'symbols.json'}; reverse maps updated")
```

---

# Symbol delta (for PRs)

Add a small companion script: **`docs/_scripts/symbol_delta.py`**. It computes **added / removed / changed** by comparing **two symbol snapshots** (base vs head). In CI, you’ll check out a base worktree and run the symbol builder there; locally you can supply a path.

```python
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, subprocess, tempfile, shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_build"

def _load(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text("utf-8"))
    return {r["path"]: r for r in rows}

def _changed(a: dict, b: dict) -> list[str]:
    reasons = []
    for k in ("signature","file","lineno","endlineno","doc"):
        if (a.get(k) or None) != (b.get(k) or None):
            reasons.append(k)
    return reasons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="path to base symbols.json (or git sha)", required=True)
    ap.add_argument("--head", help="path to head symbols.json", default=str(OUT / "symbols.json"))
    args = ap.parse_args()

    base_path = Path(args.base)
    # If '--base' is a SHA, generate symbols.json in a temp worktree
    tmp_dir = None
    if not base_path.exists():
        tmp_dir = Path(tempfile.mkdtemp())
        subprocess.run(["git", "worktree", "add", "--detach", str(tmp_dir), args.base], check=True)
        # run your symbol builder in tmp_dir
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"),
                        str(tmp_dir / "docs" / "_scripts" / "build_symbol_index.py")], check=True)
        base_path = tmp_dir / "docs" / "_build" / "symbols.json"

    base = _load(base_path)
    head = _load(Path(args.head))

    added = sorted(set(head) - set(base))
    removed = sorted(set(base) - set(head))
    changed = []
    for k in sorted(set(head) & set(base)):
        reasons = _changed(base[k], head[k])
        if reasons:
            changed.append({
                "path": k,
                "before": {x: base[k].get(x) for x in ("signature","file","lineno","endlineno","doc")},
                "after":  {x: head[k].get(x) for x in ("signature","file","lineno","endlineno","doc")},
                "reasons": reasons,
            })
    delta = {
        "base_sha": args.base if len(args.base) == 40 else None,
        "head_sha": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        "added": added,
        "removed": removed,
        "changed": changed,
    }
    OUT.write_text(json.dumps(delta, indent=2), "utf-8") if False else None
    (OUT / "symbols.delta.json").write_text(json.dumps(delta, indent=2), "utf-8")
    if tmp_dir:
        subprocess.run(["git", "worktree", "remove", "--force", str(tmp_dir)], check=True)

if __name__ == "__main__":
    main()
```

**Why SHA+worktree?** In a PR, you typically have **base SHA** and **head SHA** available; this lets you compute a precise symbol delta without artifacts. (Same idea we used in the “Agent Brief.”)

---

# CLI & env knobs (precise)

* `DOCS_PKG`: comma-separated list of packages to index (falls back to your package detector).
* `DOCS_LINK_MODE`: `github` | `editor` | `both` (default `both`).
* `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`: populate commit-stable permalinks; if unset, detect SHA via `git rev-parse HEAD`.
* `--base <sha|path>` for `symbol_delta.py` to define the baseline snapshot.

---

# CI / pre-commit wiring

**Pre-commit (fast safety):**

```yaml
- repo: local
  name: symbols: build (fast)
  entry: python docs/_scripts/build_symbol_index.py
  language: system
  pass_filenames: false
  always_run: true
```

**CI (docs job, after NavMap/Test Map):**

```bash
# 1) Build symbol index (head)
python docs/_scripts/build_symbol_index.py

# 2) Ensure no drift in committed artifacts (optional for you)
git diff --exit-code docs/_build/symbols.json docs/_build/by_file.json docs/_build/by_module.json \
  || (echo "::error::Symbol index drift; run builder locally and commit" && exit 1)

# 3) (PRs only) Produce delta against base SHA
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  BASE_SHA="${GITHUB_EVENT_PATH:+$(jq -r '.pull_request.base.sha' "$GITHUB_EVENT_PATH")}"
  python docs/_scripts/symbol_delta.py --base "$BASE_SHA"
  echo "### Symbol delta" >> $GITHUB_STEP_SUMMARY
  cat docs/_build/symbols.delta.json >> $GITHUB_STEP_SUMMARY
fi
```

---

# Tests (quick but real)

1. **Determinism**: run the builder twice → no diff.
2. **NavMap fusion**: when `navmap.json` provides `owner/stability/section`, rows contain them; when `module_meta` has defaults, they’re inherited.
3. **Deep links**: with `DOCS_LINK_MODE=github`, entries include `source_link.github` of the form `…/blob/<SHA>/path#Lx-Ly` (documented GitHub pattern). ([GitHub Docs][2])
4. **Aliases**: if a symbol is an alias, `canonical_path` differs from `path` (matches Griffe semantics). ([Mkdocstrings][1])
5. **Reverse maps**: `by_file.json["src/.../config.py"]` contains all FQNs from that file; `by_module.json["pkg.mod"]` contains module’s symbols.
6. **Delta**: modify a function signature and line span; `symbols.delta.json` lists it under `changed` with reasons `["signature","lineno","endlineno"]`.

---

# Developer UX (copy-paste quick refs)

* **Generate now**: `python docs/_scripts/build_symbol_index.py`
* **Open a symbol**: find FQN in `symbols.json` → jump via `source_link.github` or `source_link.editor`.
* **Explain a row**:

  * `path`: where it’s imported/declared; `canonical_path`: where it originates (for aliases). ([Mkdocstrings][1])
  * `owner/stability/section`: from NavMap (your policy).
  * `tested_by`: top tests (file + line hints) from test map.
  * `signature`: callable signature when Griffe can parse it.

---

# Why this design works well with the rest of your stack

* It **reuses** the **NavMap** as the single source of truth for ownership/stability/sections and merges it into a flat, agent-friendly table.
* It **reuses** the **Test Map** (already built earlier) so “who tests me?” is a single lookup.
* It adds **commit-stable permalinks** that are robust and documented, and **editor deep links** that let humans jump straight to the line. ([GitHub Docs][2])
* It stores **canonical paths** so aliases and re-exports don’t confuse agents (a well-known pattern in Griffe). ([Mkdocstrings][1])

---

# Acceptance checklist

* `docs/_build/symbols.json` contains: `path`, `canonical_path`, `kind`, `package`, `module`, `file`, `lineno/endlineno`, `doc`, `signature`, `owner`, `stability`, `since/deprecated_in`, `section`, `tested_by`, and `source_link.{github,editor}` for all public symbols.
* `docs/_build/by_file.json` and `docs/_build/by_module.json` exist and are sorted.
* Re-running builder is a **no-op** when nothing changed.
* In a PR, `symbols.delta.json` reflects the symbol-surface changes and lists concise reasons.
* CI fails on drift (optional) and publishes the delta in the job summary.

---

## Pointers to your current code and stage doc (for reviewers)

* Current symbol builder (baseline we’re extending): `docs/_scripts/build_symbol_index.py`. 
* Stage description & orchestration in the docs README (“Stage 13: Symbol Index Build”). 

**External references used**

* Griffe object identity & source location (paths, lineno, canonical vs import path). ([Mkdocstrings][1])
* GitHub **commit-stable permalinks** and **line/range anchors** (`#Lnn`, `#Lnn-Lmm`). ([GitHub Docs][2])

If you want, I can turn these patches into a PR against your current `build_symbol_index.py` and add a tiny test fixture to keep the behavior locked in.

[1]: https://mkdocstrings.github.io/griffe/guide/users/navigating/?utm_source=chatgpt.com "Navigating - Griffe"
[2]: https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files?utm_source=chatgpt.com "Getting permanent links to files - GitHub Docs"
