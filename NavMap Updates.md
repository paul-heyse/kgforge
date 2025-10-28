Awesome—here’s a **fully-specified, repo-grounded implementation plan** for **Item 2: the NavMap system**. It is written to drop directly into your current toolchain (`tools/navmap/*`) and to align with your **Automated Documentation System** README and the six scripts you attached (update, strip, build, check, migrate, repair). Where I say “patch,” I show exact code you can paste in. I also call out the external rules I’m relying on (Python identifiers and version strings; GitHub permalinks; VS Code URL scheme) with citations.

> TL;DR
> You’ll (1) declare a **policy-as-code** doc + schema, (2) extend `build_navmap.py` so the JSON is richer and link-mode aware, (3) harden `check_navmap.py` to enforce exports/sections/anchors **and** `owner`/`stability`/`since`/`deprecated_in` with PEP 440 validation, (4) add **round-trip** checks, (5) wire **autofixes** (anchors/sections) in `repair_navmaps.py`, (6) keep docstrings free of legacy `NavMap:` (your current `update_navmaps.py`+`strip_navmap_sections.py`), and (7) gate everything in pre-commit/CI.

---

## 0) What you already have (baseline)

* **Legacy kill-switch:** `tools/update_navmaps.py` fails if any module docstring still contains a `NavMap:` block; `tools/strip_navmap_sections.py` removes such blocks safely.  
* **Index builder:** `tools/navmap/build_navmap.py` parses every `src/**.py`, extracts `__all__`, `__navmap__` (if present), and inline markers `# [nav:section id]`, `# [nav:anchor Symbol]`; produces `site/_build/navmap/navmap.json` with `commit`, per-module `exports`, `sections`, `section_lines`, `anchors`, `links.source` (vscode://), `meta` (per-symbol dict), plus optional tags/synopsis/see_also/deps. 
* **Validator:** `tools/navmap/check_navmap.py` enforces **exports match**, **first section is `public-api`**, **kebab-case section IDs**, **valid Python identifiers**, and **anchor presence** for each section symbol. 
* **Migrate/repair wrappers:** `migrate_navmaps.py` writes the JSON via `build_index()`, and `repair_navmaps.py` currently just proxies to the checker.  
* **Orchestration:** Your README documents “Stage 3–5: NavMap Update/Build/Check” as part of `tools/update_docs.sh`. 

That’s a solid spine. Below are the targeted upgrades.

---

## 1) Policy-as-code (make rules explicit and machine-checked)

Create **`docs/policies/visibility-policy.md`** (human) and **`docs/policies/visibility-policy.json`** (machine) with *exact* rules your tools will enforce:

### 1.1 Required structure

* Each module **defines the public surface** as the union of:

  * `__all__` (authoritative export list), and
  * `__navmap__["exports"]` (may reference `__all__`; must match set-wise).
* `__navmap__` **must** include:

  * `"sections"`: list of `{id: "<kebab-case>", symbols: ["Name", ...]}`

    * First section **must** be `"public-api"`. (You already validate this.)
  * `"symbols"`: mapping `name -> {"stability": "stable|beta|experimental|deprecated", "owner": "@team", "since": "X[.Y[.Z]]", "deprecated_in": "X[.Y[.Z]]" (optional) }`.

    * `since` and `deprecated_in` **must** follow **PEP 440** version scheme; if both exist, `deprecated_in >= since`. ([Python Enhancement Proposals (PEPs)][1])
* For every symbol listed in any section:

  * A **source anchor** comment `# [nav:anchor Symbol]` must appear adjacent to the definition line. (You already enforce presence; we’ll add autofix below.)
  * The symbol name must be a valid **Python identifier** (starts with letter or `_`, then letters/digits/underscore; full rule per Python lexical analysis). ([Python documentation][2])

### 1.2 Link policy

* *Editor links* use **VS Code URL scheme**: `vscode://file/<abs_or_repo_rel_path>:line:column` (official). ([Visual Studio Code][3])
* *GitHub permalinks* must pin **commit SHA** and **line or range** using `#Lnn`/`#Lnn-Lmm`. ([GitHub Docs][4])

> The policy file is read by `check_navmap.py` at runtime so you can tweak stability vocab or section ID rules without code edits.

---

## 2) Extend the JSON schema of `navmap.json` (no breaking changes)

**Today** your builder emits (per module): path, exports, sections, section_lines, anchors, links.source (vscode), meta, tags, synopsis, see_also, deps. 

**Add** (safe additions, backward-compatible):

```json
{
  "policy_version": "1",
  "link_mode": "editor|github|both",
  "links": {
    "source": "vscode://file/...",              // existing
    "github": "https://github.com/...#Lnn-Lmm"  // new when link_mode includes github
  },
  "module_meta": {
    "owner": "@team-id",
    "stability": "stable|beta|experimental|deprecated",
    "since": "0.5.0",
    "deprecated_in": null
  }
}
```

**Why:** Agents and tools get one place to read current policy version, link preferences, and module-level defaults (applies when a symbol omits owner/stability).

---

## 3) Patch `tools/navmap/build_navmap.py` (richer links + defaults)

Keep your AST parsing + data classes. Add:

1. **Link-mode awareness**

   * Read `DOCS_LINK_MODE` = `editor|github|both` and `DOCS_GITHUB_ORG/REPO/SHA`.
   * Populate `links.github` alongside `links.source` when requested; format permalinks with `blob/<SHA>#Lstart-Lend`. ([GitHub Docs][4])

2. **Module defaults**

   * If `__navmap__` has top-level `"owner"`, `"stability"`, `"since"`, `"deprecated_in"`, copy them to `module_meta`.
   * During symbol collection, **inherit** missing per-symbol fields from `module_meta`.

3. **Normalize**

   * Ensure section IDs are stored kebab-case (leave validation to checker).
   * Ensure `exports` is **list-stable** (sort) while keeping original `__all__` ordering for display if you prefer.

**Minimal patch (additions only; keep your structure):**

```python
# build_navmap.py (add near top)
import os
from typing import cast
G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "editor")

def _gh_link(path: Path, start: int | None, end: int | None) -> str | None:
    if not (G_ORG and G_REPO):
        return None
    sha = G_SHA or _git_sha()
    frag = f"#L{start}" if start and not end else (f"#L{start}-L{end}" if start and end else "")
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{sha}/{_rel(path)}{frag}"
```

Inside `build_index()`, after you assemble each module entry:

```python
entry = {
  "path": _rel(info.path),
  "exports": exports,
  "sections": navmap.get("sections", []),
  "section_lines": info.sections,
  "anchors": info.anchors,
  "links": {
     "source": f"vscode://file/{_rel(info.path)}",
  },
  "meta": navmap.get("symbols", {}),
  "tags": navmap.get("tags", []),
  "synopsis": navmap.get("synopsis", ""),
  "see_also": navmap.get("see_also", []),
  "deps": navmap.get("deps", []),
  "module_meta": {k: navmap.get(k) for k in ("owner","stability","since","deprecated_in") if navmap.get(k) is not None},
}
if LINK_MODE in ("github", "both"):
    entry["links"]["github"] = _gh_link(info.path, None, None)

data.setdefault("policy_version", "1")
data.setdefault("link_mode", LINK_MODE)
data["modules"][info.module] = entry
```

(If you prefer line-exact GitHub links, you can map anchors to `def/class` start and compute ends via AST; that’s optional and can come later.)

---

## 4) Harden `tools/navmap/check_navmap.py` (strict + friendly)

You already check **exports match**, **first section public-api**, **kebab-case**, **identifier validity**, **anchor presence**. 
Add:

* **Owner/Stability required** for every **exported** symbol (`__all__` or `__navmap__["exports"]`):

  * `stability ∈ {"stable","beta","experimental","deprecated"}`
  * `owner` is non-empty string (`"@team"` recommended)
* **PEP 440** validation for `since` and `deprecated_in` (use `packaging.version.Version`); if both present, `deprecated_in >= since`. ([Python Enhancement Proposals (PEPs)][1])
* **Round-trip guard**: load JSON via `build_index()` (in-process), then verify for each module that:

  * `exports` sets match (`__all__` vs JSON → already handled).
  * For every section ID in JSON, the recorded `section_lines[section]` exists in text and is adjacent to a `# [nav:section ...]` line with the same slug.
  * For every symbol in any section, `anchors[symbol]` exists and the text line still matches `# [nav:anchor Symbol]`.
  * If any mismatch, emit a single “round-trip mismatch” group with expected and observed values and **one-line fixes** (see next subsection).

**Patch (sketch—drop into your file at the end of `_inspect` or post-processing):**

```python
# check_navmap.py (add utility near top)
from packaging.version import Version, InvalidVersion  # add to docs extras

STABILITY = {"stable","beta","experimental","deprecated"}

def _validate_versions(meta: dict[str, Any], py: Path, errors: list[str]) -> None:
    for name, fields in meta.items():
        since = fields.get("since")
        deprec = fields.get("deprecated_in")
        if since:
            try: Version(str(since))
            except InvalidVersion: errors.append(f"{py}: symbol '{name}' has non-PEP440 since='{since}'")
        if deprec:
            try: Version(str(deprec))
            except InvalidVersion: errors.append(f"{py}: symbol '{name}' has non-PEP440 deprecated_in='{deprec}'")
        if since and deprec:
            if Version(str(deprec)) < Version(str(since)):
                errors.append(f"{py}: symbol '{name}' deprecated_in ({deprec}) < since ({since})")

def _validate_meta_required(meta: dict[str, Any], exports: set[str], py: Path, errors: list[str]) -> None:
    for name in sorted(exports):
        fields = meta.get(name, {})
        stab = fields.get("stability")
        own  = fields.get("owner")
        if stab not in STABILITY:
            errors.append(f"{py}: symbol '{name}' missing/invalid stability (got {stab!r})")
        if not own:
            errors.append(f"{py}: symbol '{name}' missing owner (e.g., '@core-search')")
```

Call these inside `_inspect` once you’ve derived `exports`, `navmap`, `anchors`, `sections`:

```python
meta = (navmap or {}).get("symbols", {})
_validate_meta_required(meta, exports, py, errors)
_validate_versions(meta, py, errors)
```

**Ready-to-paste fixes in messages:** make errors actionable by including **sed-ready** snippets:

* Missing anchor →
  `echo '# [nav:anchor {sym}]' | gsed -i '{lineno}i\# [nav:anchor {sym}]' {path}`
  (or a short Python command that inserts a line after the `def/class` line).

* Missing section header →
  `echo '# [nav:section public-api]' | gsed -i '1i\# [nav:section public-api]' {path}`

(Use `sed -i ''` on macOS/BSD; document both variants.)

**Identifier rule source:** Python identifiers must start with a letter or underscore, then letters/digits/underscore—your REGEX logic matches the Python reference. ([Python documentation][2])

---

## 5) Add **autofix** pathways (`repair_navmaps.py`)

Right now `repair_navmaps.py` just reuses `_inspect`. Enhance it so it can **insert anchors** next to symbol definitions and **seed a `public-api`** section if missing:

* **Strategy**:

  * Parse AST; for each exported symbol with missing anchor, find the `FunctionDef`/`ClassDef` line; insert `# [nav:anchor Name]` on the line above.
  * If no `sections` or first section ≠ `public-api`, insert `# [nav:section public-api]` at top (after module docstring).
  * If `__navmap__` missing entirely, insert a minimal stub that references `__all__`.

*You already have the AST + rewrite skills in `strip_navmap_sections.py`; reuse that pattern to rewrite doc/inline comments safely.* 

---

## 6) Round-trip guard (detect drift early)

Add one more checker **after** building navmap:

* **`tools/navmap/roundtrip_check.py`**:

  * Runs `build_index()` to get a deterministic JSON snapshot.
  * Reloads each source module, verifies that every section/anchor referenced by JSON is still present at the recorded line, and that every section listed in the module re-appears in JSON at the same slug.
  * Exit 1 with a compact diff (expected vs observed line numbers, slugs, and symbol lists).

Wire this as a final step inside `check_navmap.py` (so you keep a single entrypoint).

---

## 7) CLI & environment contract (simple and explicit)

All tools in `tools/navmap` accept:

* `--root PATH` (default `src/`): subtree to scan. (You already accept `--root` in `repair_navmaps.py`.) 
* `--json PATH` (default `site/_build/navmap/navmap.json`): where to read/write index. (You already write here.) 
* `--mode warn|error` (checker): downgrade errors to warnings for local dev if desired.
* `DOCS_LINK_MODE=editor|github|both` and `DOCS_GITHUB_ORG/REPO/SHA` to control link emission (build step). (Cited above.) ([Visual Studio Code][3])

---

## 8) Pre-commit & CI gates (make it impossible to drift)

* **Pre-commit**: always run, whole tree (order matters):

  1. `python tools/navmap/build_navmap.py` → refresh JSON
  2. `python tools/navmap/check_navmap.py` → **must pass**
  3. *(optional)* `python tools/navmap/repair_navmaps.py` in **dry-run** to print autofix hints.
* **CI** (docs job):

  * Run build → check → roundtrip.
  * Fail if any **exported symbol** is missing `owner` or `stability`, or if `public-api` isn’t first, or if anchors/sections are missing, or if round-trip fails.
  * Fail if `navmap.json` differs after running build (drift).

---

## 9) Developer UX (tiny, copy-pasteable fixes)

**Missing anchor** (Linux/macOS GNU sed):

```bash
sym="foo"; file="src/pkg/mod.py"; line=$(nl -ba "$file" | grep -nE "^(.*def $sym\(|.*class $sym\b)" | head -1 | cut -f1)
[ -n "$line" ] && gsed -i "${line}i\# [nav:anchor ${sym}]" "$file"
```

**Seed minimum `__navmap__`** (module has `__all__`):

```python
# add below existing __all__ in src/pkg/mod.py
__navmap__ = {
    "exports": __all__,
    "sections": [{"id": "public-api", "symbols": list(__all__)}],
    "symbols": {name: {"stability": "experimental", "owner": "@core"} for name in __all__},
}
```

**Fix versions** (PEP 440; e.g. when deprecating):

```python
__navmap__["symbols"]["Foo"]["since"] = "0.6.0"
__navmap__["symbols"]["Foo"]["deprecated_in"] = "0.9.0"
```

> PEP 440 governs valid version strings—stick to `N(.N)*[.postN][.devN][pre/rc]`. ([Python Enhancement Proposals (PEPs)][1])

---

## 10) Tests (quick but real)

* **Unit**:

  * “happy path” module with `__all__`, `__navmap__`, sections, anchors → `check_navmap.py` returns OK.
  * Missing anchor/owner/stability/first section/exports mismatch → each yields one precise error with file/line.
  * Invalid identifiers (e.g., `123abc`) rejected. (Matches Python identifiers spec). ([Python documentation][2])
  * PEP 440 versions validated; `deprecated_in < since` rejected. ([Python Enhancement Proposals (PEPs)][1])
* **Round-trip**: alter an anchor line number and ensure `roundtrip_check` detects the mismatch.
* **Integration**: run full sequence (build → check) on a small fixture tree.

---

## 11) Acceptance (what “done” looks like)

* `site/_build/navmap/navmap.json` contains **policy_version**, **link_mode**, **links.github** when requested, and **module_meta** defaults.
* `check_navmap.py` produces **zero** errors on a clean tree; emits **actionable** messages with one-liners when something is missing.
* CI **fails** if:

  * any exported symbol lacks `owner`/`stability`,
  * first section isn’t `public-api`,
  * anchors missing, invalid IDs, identifier mismatch,
  * PEP 440 version violations, or
  * round-trip mismatch or JSON drift.
* Pre-commit can run on an arbitrary diff and still green-light only if the tree is consistent.

---

## 12) Why these external rules?

* **Python identifiers**: must begin with a letter or underscore; the rest can include digits. That’s exactly what your `SYMBOL_RE = r"^[A-Za-z_]\w*$"` encodes. ([Python documentation][2])
* **`__all__` semantics**: it’s the standard way to define a module/package’s public export surface for `from X import *`, so syncing `__navmap__["exports"]` with `__all__` is correct. ([Python documentation][5])
* **PEP 440**: using canonical version strings for `since`/`deprecated_in` makes tooling predictable and lets your validator compare versions robustly. ([Python Enhancement Proposals (PEPs)][1])
* **GitHub permalinks with `#L…`**: pin to the commit hash so links never rot. ([GitHub Docs][4])
* **VS Code `vscode://file`**: first-class, documented URL scheme for opening files to a specific line/column. ([Visual Studio Code][3])

---

## Ready-to-run sequence

1. **Land policy files** in `docs/policies/` and add `packaging` to your `[docs]` extra (for PEP 440 checks).
2. **Patch `build_navmap.py`** with link-mode + module defaults + schema fields. 
3. **Patch `check_navmap.py`** with meta-required + PEP 440 + round-trip call. 
4. **Enhance `repair_navmaps.py`** with anchor/section inserters. 
5. **Keep `update_navmaps.py`** + **`strip_navmap_sections.py`** early in the pipeline to prevent legacy docstring blocks from creeping back.  
6. **Pre-commit & CI**: build → check → round-trip; fail on drift or missing required metadata.
7. (**Optional**) add a nightly job that prints a navmap “health” summary (missing owners, deprecated symbols remaining after their target version, modules with no `public-api`).

If you’d like, I can now turn those patches into a PR (with tests) against the six scripts you shared.

---

**Web references used:**

* Python identifiers (valid name syntax), Python Language Reference — *Lexical analysis*. ([Python documentation][2])
* `__all__` behavior in modules and packages — *Python Tutorial: Modules*. ([Python documentation][5])
* PEP 440 version scheme — *peps.python.org*. ([Python Enhancement Proposals (PEPs)][1])
* GitHub permalinks with `#L`/`#L-L` — *GitHub Docs*. ([GitHub Docs][4])
* VS Code URL scheme `vscode://file/...:line:col` — *VS Code docs (CLI/URLs)*. ([Visual Studio Code][3])

[1]: https://peps.python.org/pep-0440/?utm_source=chatgpt.com "PEP 440 – Version Identification and Dependency Specification | peps.python.org"
[2]: https://docs.python.org/3.12/reference/lexical_analysis.html?utm_source=chatgpt.com "2. Lexical analysis — Python 3.12.11 documentation"
[3]: https://code.visualstudio.com/docs/editor/command-line?utm_source=chatgpt.com "Command Line Interface (CLI)"
[4]: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-a-permanent-link-to-a-code-snippet?utm_source=chatgpt.com "Creating a permanent link to a code snippet - GitHub Docs"
[5]: https://docs.python.org/3/tutorial/modules.html?utm_source=chatgpt.com "6. Modules — Python 3.14.0 documentation"
