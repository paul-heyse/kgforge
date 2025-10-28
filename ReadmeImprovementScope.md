Got it — let’s make this painfully clear and fully specified, with exact knobs, sensible defaults, examples, and drop-in code you can paste into `tools/gen_readmes.py`. I’ll also call out what *changes* in your current script (I read the copy you attached) and why.

---

# What the README generator does — in one sentence

For each public package under `src/`, generate a **deterministic** `README.md` that lists Modules / Classes / Functions / Exceptions with a one-line summary, **badges** (stability, owner, section), **tested-by** pointers, and **links**:

* **view** → commit-pinned GitHub permalink to the exact lines
* **open** → a local editor deep-link when possible (VS Code URL scheme) or your existing relative `./path:line:col`.
  GitHub permalinks use `…/blob/<SHA>/file#Lstart-Lend`, which is the documented way to pin to specific lines. ([GitHub Docs][1])

---

# Configuration interface (precise, English, and strict)

## 1) CLI flags (highest precedence)

Run:
`python tools/gen_readmes.py [options]`

| Flag                      | Type             |         Default | Allowed                    | Meaning                                                                                                                                                                                                                                                                                             |
| ------------------------- | ---------------- | --------------: | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--packages`              | list (comma-sep) | *(auto-detect)* | any                        | Limit generation to these top-level packages (e.g. `kgfoundry,kg_builder`). If omitted, we use your current `iter_packages()` logic (env or detected).                                                                                                                                              |
| `--link-mode`             | enum             |          `both` | `github`, `editor`, `both` | Which links to render after each item. `github` = only permalinks; `editor` = only local editor deep-links; `both` = show both.                                                                                                                                                                     |
| `--editor`                | enum             |        `vscode` | `vscode`, `relative`       | How to build “open” links. `vscode` emits `vscode://file/<abs>:line:col` (clickable in many browsers/OS). `relative` keeps your existing `./path:line:col`. The VS Code CLI also supports `code -g file:line[:col]`, which we’ll mention in docs but not embed as a link. ([Visual Studio Code][2]) |
| `--fail-on-metadata-miss` | bool             |         `false` | —                          | If set, exit non-zero when a public symbol lacks `stability` or `owner` in navmap.                                                                                                                                                                                                                  |
| `--dry-run`               | bool             |         `false` | —                          | Print planned writes, don’t touch files.                                                                                                                                                                                                                                                            |
| `--verbose`               | bool             |         `false` | —                          | Extra logging (counts, timing).                                                                                                                                                                                                                                                                     |

**Precedence**: CLI flag → environment variable → default.

## 2) Environment variables (fallbacks for flags)

| Variable           | Maps to       | Example                | Notes                                                                                                                                                                                                                                              |
| ------------------ | ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DOCS_PKG`         | `--packages`  | `kgfoundry,kg_builder` | Comma-separated packages to process.                                                                                                                                                                                                               |
| `DOCS_LINK_MODE`   | `--link-mode` | `both`                 | `github` | `editor` | `both`.                                                                                                                                                                                                                      |
| `DOCS_EDITOR`      | `--editor`    | `vscode`               | `vscode` emits `vscode://file/...`; `relative` keeps `./path:line:col`. (PyCharm doesn’t have an officially documented universal URL scheme; use your relative link and let devs use `pycharm --line N path` from the shell.) ([jetbrains.com][3]) |
| `DOCS_GITHUB_ORG`  | owner         | `paul-heyse`           | You already read these today.                                                                                                                                                                                                                      |
| `DOCS_GITHUB_REPO` | repo          | `kgfoundry`            | —                                                                                                                                                                                                                                                  |
| `DOCS_GITHUB_SHA`  | sha           | `5b103c1...`           | If unset, we fall back to `git rev-parse HEAD`.                                                                                                                                                                                                    |

That “open in editor” note, in English:

* If your team uses **VS Code**, we generate **clickable** `vscode://file/<abs>:<line>:<col>` links. This is an official scheme and pairs with `code -g file:line[:col]` on the CLI. ([Visual Studio Code][2])
* If your team uses **PyCharm**, there’s no official, stable `pycharm://` URL documented for general use. The *documented* way is the CLI `pycharm --line 42 <path>`; so we keep the **relative** link and provide GitHub “view” permalinks for everyone. ([jetbrains.com][3])

---

# Inputs consumed (where data comes from)

* **Object tree**: via Griffe (you already use it in `gen_readmes.py`). ([mkdocstrings.github.io][4])
* **NavMap JSON**: `site/_build/navmap/navmap.json` — provides `stability`, `owner`, `section`, optional `since`, `deprecated_in` per symbol.
* **Test map JSON**: `docs/_build/test_map.json` — provides `symbol → [{file, lines[]}, …]` (we show top 1–3).
* **Repo metadata**: owner/repo/SHA from env or `git rev-parse HEAD` (your current functions).
* **Package list**: from `--packages`/`DOCS_PKG`; else `iter_packages()` (your current detection).

---

# Output contract (what exactly gets written)

For each package `src/<pkg>/README.md`:

1. H1: ``# `<pkg>` `` + one-line synopsis (first line of the package `__init__` docstring if present; else a deterministic fallback sentence).
2. Doctoc markers (your current comments) — if you run DocToc later, it fills the TOC. (DocToc is a well-known CLI that updates the region between its HTML comment markers; we keep your markers to stay compatible.) ([npm][5])
3. **Sections in fixed order**: `## Modules`, `## Classes`, `## Functions`, `## Exceptions` (only emitted if non-empty).
4. **Entry format (bullet):**

   ```
   - **`fully.qualified.name`** — First sentence of docstring
     `stability:<stable/beta/experimental/deprecated>` `owner:@team` `section:<nav-section>`
     `since:0.x` `deprecated:0.y` `tested-by: tests/unit/test_foo.py:42, tests/e2e/test_bar.py:15`
     → [open](vscode://file/ABS:LINE:1) | [view](https://github.com/<org>/<repo>/blob/<SHA>/path#Lstart-Lend)
   ```

   “view” uses the GitHub line anchor patterns (`#L10` or `#L10-L15`) so it’s precise at that commit. ([GitHub Docs][1])
5. **Provenance footer** (invisible to readers):
   `<!-- agent:readme v1 sha:<SHA> content:<hash12> -->`
   We compute the hash from the rendered content and only write when it changed (idempotent).

---

# Filtering rules (exact)

* Skip names starting with `_` (you already do this in `is_public()`).
* Only include kinds in `{module, package, class, function}`; you already gate these.
* Treat exception classes as “Exceptions” **iff** their MRO includes a subclass of `Exception` (we check base names during render).
* Skip Pydantic/private/generated artifacts if they leak into Griffe (you’re already filtering by kind; that’s usually enough).

---

# What changes in your current file (surgical diffs)

Your attached `tools/gen_readmes.py` already has:

* `detect_repo()`, `git_sha()`, `gh_url()`, `get_open_link()`, `get_view_link()`, `render_member()`, `write_readme()`, and the Griffe loader.

**We keep** your loader, repo detection, GitHub URL builder, and traversal.
**We add / change**:

1. **Config parsing** (CLI + env) with clean precedence.
2. **Metadata loader** for navmap + test map.
3. **Badges** and **tested-by** rendering.
4. **Editor link** builder that can emit `vscode://file/...` when asked; else fall back to your `./path:line:col`.
5. **Deterministic grouping** (Modules / Classes / Functions / Exceptions) rather than a single tree; still strictly sorted by FQN.
6. **Write-if-changed** with content hash in a footer.

---

# Drop-in code (clear, cohesive snippets)

> Paste these into your existing `gen_readmes.py`. They’re self-contained and reuse your functions where possible. Replace or extend your helpers where names collide.

### 1) Configuration & inputs (top of file)

```python
# --- Configuration & Inputs ---------------------------------------------------
import argparse, json, hashlib, time
from dataclasses import dataclass
from typing import Iterable

NAVMAP_PATH = Path("site/_build/navmap/navmap.json")
TESTMAP_PATH = Path("docs/_build/test_map.json")

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

@dataclass(frozen=True)
class Config:
    packages: list[str]
    link_mode: str       # 'github' | 'editor' | 'both'
    editor: str          # 'vscode' | 'relative'
    fail_on_metadata_miss: bool
    dry_run: bool
    verbose: bool

def parse_config() -> Config:
    p = argparse.ArgumentParser(description="Generate per-package READMEs.")
    p.add_argument("--packages", default=os.getenv("DOCS_PKG", ""), help="Comma-separated packages")
    p.add_argument("--link-mode", default=os.getenv("DOCS_LINK_MODE", "both"), choices=["github","editor","both"])
    p.add_argument("--editor", default=os.getenv("DOCS_EDITOR", "vscode"), choices=["vscode","relative"])
    p.add_argument("--fail-on-metadata-miss", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    pkgs = [s.strip() for s in args.packages.split(",") if s.strip()] if args.packages else iter_packages()
    return Config(
        packages=pkgs,
        link_mode=args.link_mode,
        editor=args.editor,
        fail_on_metadata_miss=args.fail_on_metadata_miss,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
```

### 2) Metadata (badges + tests)

```python
@dataclass(frozen=True)
class Badges:
    stability: str | None = None
    owner: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    tested_by: list[dict] = None

_nav = _load_json(NAVMAP_PATH)
_tests = _load_json(TESTMAP_PATH)

def _lookup_nav(qname: str) -> dict:
    # Expect a dict keyed by module/package with per-symbol meta; be liberal in reading.
    # If your navmap differs, adapt this lookup in one place.
    if not isinstance(_nav, dict):
        return {}
    for mod in _nav.get("modules", {}).values():
        meta = mod.get("meta", {}).get(qname)
        if meta:
            # Try to derive the section (if your schema has sections with symbol lists)
            for sec in mod.get("sections", []):
                if qname.split(".")[-1] in sec.get("symbols", []):
                    meta = {**meta, "section": sec.get("id")}
                    break
            return meta
    return {}

def badges_for(qname: str) -> Badges:
    m = _lookup_nav(qname)
    tb = _tests.get(qname, []) if isinstance(_tests, dict) else []
    return Badges(
        stability=m.get("stability"),
        owner=m.get("owner"),
        section=m.get("section"),
        since=m.get("since"),
        deprecated_in=m.get("deprecated_in"),
        tested_by=tb[:3] if isinstance(tb, list) else [],
    )

def format_badges(qname: str) -> str:
    b = badges_for(qname)
    tags = []
    if b.stability:      tags.append(f"`stability:{b.stability}`")
    if b.owner:          tags.append(f"`owner:{b.owner}`")
    if b.section:        tags.append(f"`section:{b.section}`")
    if b.since:          tags.append(f"`since:{b.since}`")
    if b.deprecated_in:  tags.append(f"`deprecated:{b.deprecated_in}`")
    if b.tested_by:
        show = [f"{t['file']}:{(t.get('lines') or [1])[0]}" for t in b.tested_by]
        tags.append("`tested-by:" + ", ".join(show) + "`")
    return (" " + " ".join(tags)) if tags else ""
```

### 3) Editor link (VS Code URL or your current relative)

```python
def editor_link(abs_path: Path, lineno: int, editor_mode: str) -> str | None:
    if editor_mode == "vscode":
        # Official scheme: vscode://file/<abs>:line:col
        # CLI alternative (for docs only): `code -g file:line[:character]`
        return f"vscode://file/{abs_path}:{lineno}:1"  # clickable link
    # 'relative' → keep your existing behavior (./path:line:col)
    return None  # caller will fall back to get_open_link()
```

*(Why we don’t emit a PyCharm URL: JetBrains officially documents the **CLI** (`pycharm --line 42 <file>`), not a universal URL scheme we can rely on in Markdown. Keep relative links + GitHub permalinks for PyCharm users.)* ([jetbrains.com][3])

### 4) Grouping & rendering (replace your `render_member` body)

```python
KINDS = {"module","package","class","function"}

def bucket_for(node: Object) -> str:
    k = getattr(getattr(node, "kind", None), "value", "")
    if k in {"module","package"}: return "Modules"
    if k == "class":
        # Heuristic: exception bucket if MRO contains 'Exception'
        bases = [b.target.path if hasattr(b, "target") else str(b) for b in getattr(node, "bases", [])]
        return "Exceptions" if any("Exception" in str(x) for x in bases) else "Classes"
    if k == "function": return "Functions"
    return "Other"

def render_line(node: Object, readme_dir: Path, cfg: Config) -> str | None:
    qname = getattr(node, "path", "")
    summary = summarize(node)
    open_link = get_open_link(node, readme_dir) if cfg.link_mode in {"editor","both"} else None
    view_link = get_view_link(node, readme_dir) if cfg.link_mode in {"github","both"} else None

    # Prefer VS Code URI when chosen
    if cfg.link_mode in {"editor","both"}:
        rel_path = getattr(node, "relative_package_filepath", None)
        if rel_path:
            base = SRC if SRC.exists() else ROOT
            ed = editor_link((base / rel_path).resolve(), int(getattr(node, "lineno", 1) or 1), cfg.editor)
            if ed: open_link = ed

    parts = [f"- **`{qname}`**"]
    if summary: parts.append(f" — {summary}")
    parts.append(format_badges(qname))

    links = []
    if open_link: links.append(f"[open]({open_link})")
    if view_link: links.append(f"[view]({view_link})")
    tail = (" → " + " | ".join(links)) if links else ""
    return " ".join(p for p in parts if p) + tail + "\n"
```

### 5) Write-if-changed with provenance (replace your final write)

```python
def write_if_changed(path: Path, content: str) -> bool:
    stamp = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    footer = f"\n<!-- agent:readme v1 sha:{SHA} content:{stamp} -->\n"
    new_text = content.rstrip() + footer
    old_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if old_text == new_text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True
```

### 6) Build one README (replace your `write_readme` body)

```python
def write_readme(node: Object, cfg: Config) -> bool:
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"

    buckets = {"Modules": [], "Classes": [], "Functions": [], "Exceptions": [], "Other": []}
    children = [c for c in iter_public_members(node) if getattr(getattr(c, "kind", None), "value", "") in KINDS]
    for child in sorted(children, key=lambda c: getattr(c, "path", "")):
        line = render_line(child, readme_dir=pkg_dir, cfg=cfg)
        if line:
            buckets[bucket_for(child)].append(line)

    lines: list[str] = []
    lines += [f"# `{node.path}`\n\n"]
    lines += ["<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n",
              "<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n\n"]

    for section in ("Modules","Classes","Functions","Exceptions","Other"):
        items = buckets.get(section, [])
        if items:
            lines += [f"## {section}\n\n"] + items + ["\n"]

    content = "".join(lines)
    if cfg.dry_run:
        print(f"[dry-run] would write {readme}")
        return False
    changed = write_if_changed(readme, content)
    if changed:
        print(f"Wrote {readme}")
    return changed
```

### 7) Main (tie it together; enforce metadata if asked)

```python
def main():
    t0 = time.time()
    cfg = parse_config()
    if cfg.verbose:
        print(f"packages={cfg.packages} link_mode={cfg.link_mode} editor={cfg.editor}")

    missing_meta: list[str] = []
    changed_any = False

    # Load the root module objects with your existing Griffe loader:
    loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])
    for pkg in cfg.packages:
        root = loader.load(pkg)
        # pre-flight: check metadata if required
        if cfg.fail_on_metadata_miss:
            for obj in iter_public_members(root):
                qn = getattr(obj, "path", "")
                b = badges_for(qn)
                if getattr(getattr(obj, "kind", None), "value", "") in KINDS and (not b.stability or not b.owner):
                    missing_meta.append(qn)
        changed_any |= write_readme(root, cfg)

    if cfg.fail_on_metadata_miss and missing_meta:
        print("ERROR: Missing owner/stability for public symbols:\n  - " + "\n  - ".join(missing_meta))
        raise SystemExit(2)

    if cfg.verbose:
        print(f"done in {time.time()-t0:.2f}s; changed={changed_any}")

if __name__ == "__main__":
    main()
```

---

# Usage examples (copy/paste)

* Generate all READMEs with both links and VS Code deep-links:

  ```bash
  python tools/gen_readmes.py --link-mode both --editor vscode
  ```
* Only GitHub permalinks (no editor link):

  ```bash
  python tools/gen_readmes.py --link-mode github
  ```
* Limit to specific packages:

  ```bash
  DOCS_PKG="kgfoundry,kg_builder" python tools/gen_readmes.py
  ```
* Fail if any public symbol is missing owner/stability:

  ```bash
  python tools/gen_readmes.py --fail-on-metadata-miss
  ```

---

# CI & pre-commit wiring (exact)

**CI step (in your docs workflow, before Sphinx):**

```bash
python tools/gen_readmes.py --link-mode github --editor relative
git diff --exit-code -- 'src/**/README.md' \
  || (echo "::error::README drift; run: python tools/gen_readmes.py"; exit 1)
```

**Optional pre-commit (changed packages only):**

```yaml
- repo: local
  name: readme generator (fast)
  entry: bash -lc 'pkgs=$(git diff --name-only --cached | grep -E "^src/[^/]+/" | cut -d/ -f2 | sort -u | paste -sd, -); DOCS_PKG="$pkgs" python tools/gen_readmes.py --link-mode github --editor relative'
  language: system
  pass_filenames: false
```

**TOC refresher** (if you want DocToc to fill the placeholders automatically in CI):
`doctoc` is a simple CLI (`npm i -g doctoc`), and it will update the region between the HTML comments we already include. It’s idempotent and widely used for README TOCs. ([npm][5])

---

# Tests (minimal but real)

* **Determinism**: run generator twice → no diff; verify provenance footer hash unchanged.
* **Badges**: when navmap provides stability/owner, markdown contains backtick badges; when missing and `--fail-on-metadata-miss`, tool exits with code 2.
* **Links**: with `--link-mode both --editor vscode`, bullets contain a `vscode://file/...` link and a `…/blob/<SHA>/…#Lx-Ly` link; with `--link-mode github`, only the latter. (VS Code URL and CLI semantics are documented here. ([Visual Studio Code][2]))
* **Buckets**: exception classes end up under **Exceptions** (MRO includes `Exception`).
* **Scope**: names beginning with `_` do not appear.

---

# Why these link types (justification)

* **GitHub “view”**: use permalinks (`blob/<SHA>#Lx-Ly`) so links never rot as the default branch moves. This is the officially documented pattern. ([GitHub Docs][1])
* **VS Code “open”**: `vscode://file/<abs>:line:col` is a documented URL scheme and works across OSs with VS Code installed; CLI alternative is `code -g file:line[:character]`. ([Visual Studio Code][2])
* **PyCharm “open”**: JetBrains documents **CLI** (`pycharm --line N <file>`), not a universal URL that’s guaranteed to work in Markdown. We therefore keep your relative `./path:line:col` (clicking gives context, copying gives a usable path) and always include the GitHub “view” link. ([jetbrains.com][3])

---

# Mapping to your current script (so you know what to keep)

Keep: `detect_repo()`, `git_sha()`, `gh_url()`, `summarize()`, `is_public()`, `get_open_link()`, `get_view_link()`, Griffe loader.
Replace/extend: add **parse_config**, **badges_for/format_badges**, **editor_link**, **bucket_for** and **render_line**, **write_if_changed**, and update **write_readme** to group items and call `render_line(..., cfg)`.

---

# Acceptance checklist (what “done” looks like)

* Re-running `python tools/gen_readmes.py` on a clean tree produces **no diff**.
* Each README shows stable sections, one-line summaries, badges, tested-by, and links in the chosen `--link-mode`.
* CI fails on drift and (optionally) on missing owner/stability.
* Clicking **view** opens the *exact* lines at that commit on GitHub; clicking **open** (VS Code mode) opens the file at that line locally.

---

If you want, I can turn this into a small PR: commit the code edits, add the pre-commit entry, and add a one-page “How to read package READMEs” doc for contributors.

[1]: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-a-permanent-link-to-a-code-snippet?utm_source=chatgpt.com "Creating a permanent link to a code snippet - GitHub Docs"
[2]: https://code.visualstudio.com/docs/configure/command-line?utm_source=chatgpt.com "Command Line Interface (CLI)"
[3]: https://www.jetbrains.com/help/pycharm/opening-files-from-command-line.html?utm_source=chatgpt.com "Open files from the command line | PyCharm Documentation"
[4]: https://mkdocstrings.github.io/griffe/?utm_source=chatgpt.com "Overview - Griffe"
[5]: https://www.npmjs.com/package/doctoc?utm_source=chatgpt.com "doctoc - npm"
