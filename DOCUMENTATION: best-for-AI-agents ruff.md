# DOCUMENTATION: best-for-AI-agents ruff formatting 

Absolutely—here’s an **updated, repo‑aware guide** that folds in your repo map, current `pyproject.toml`, and automated docs system. It’s written **to** an AI programming agent operating inside your environment, so it’s explicit about *what to change* and *how to verify it*.

---

# Best‑for‑AI Ruff Implementation (Repo‑Aware Guide)

**Goal:** give AI agents a deterministic, low‑noise, modern Python surface: stable imports, consistent formatting, parseable docstrings, and predictable complexity—*so edits are safer and more accurate*. This update merges your original guide with repo‑specific details (layout, tooling, current config). 

---

## TL;DR — What to implement

1. **Ruff config upgrade** in `pyproject.toml`:

   * Pin Ruff version, set grouped CI output, declare `src = ["src"]`, exclude generated trees (`docs/_build`, `site/`, egg‑info), define **first‑party packages** from this repo, and ban relative imports.
   * Expand `select` to include naming, annotations, docstrings, returns/raises/try, tidy‑imports, logging, etc.; add `RUF100` to auto‑clean unused `noqa`.
   * Cap complexity/branches with McCabe + Pylint (gentle defaults).
   * Keep quotes/wrapping in the **formatter**, not linters.
     These changes extend the curated profile from your original “best‑for‑AI” doc. 

2. **Preserve the two‑stage flow**: `ruff check --select I --fix && ruff check --fix && ruff format`. Ruff’s formatter does **not** sort imports, so run `I` before formatting. 

3. **Repo‑aware import semantics**:

   * Mark first‑party modules (derived from your `src/` layout).
   * Ban relative imports to make navigation and patch targets obvious.
     Module families include: `docling`, `download`, `embeddings_dense`, `embeddings_sparse`, `kg_builder`, `kgfoundry_common`, `linking`, `observability`, `ontology`, `orchestration`, `registry`, `search_api`, `search_client`, `vectorstore_faiss`. 

4. **Docs pipeline alignment**:

   * Keep docstring *text* reflow in your doc tools, but let Ruff handle **code blocks inside docstrings** (`format.docstring-code-format = true`).
   * Ensure Sphinx/Napoleon style (Google/NumPy) matches Ruff’s pydocstyle convention; leave your existing automated updates (`tools/update_docs.sh`) intact. 

5. **If you keep Black:** retain `ignore = ["E203"]` for Black compatibility. If you migrate to **Ruff as sole formatter**, you can drop that ignore and remove Black’s hook later. Your current `pyproject.toml` has both Ruff and Black configured. 

---

## 1) Replace the Ruff blocks in `pyproject.toml`

> **Action:** Replace **only** the `[tool.ruff]`, `[tool.ruff.format]`, and `[tool.ruff.lint*]` sections with the blocks below. Keep your existing project metadata and tools. Current values (e.g., `line-length = 100`, `target-version = "py313"`) are preserved. 

```toml
# --- Ruff: top-level settings ---
[tool.ruff]
# Keep your line-length/target-version; add determinism and repo awareness.
line-length = 100
target-version = "py313"
required-version = ">=0.14.2"        # pin to the version you run in CI
output-format = "grouped"           # easier to parse in CI logs
src = ["src"]                       # repo uses a src/ layout

# Ignore generated or non-source trees to cut noise and speed up runs.
extend-exclude = [
  "docs/_build/**",
  "site/**",
  "src/kgfoundry.egg-info/**",
  "kgfoundry.egg-info/**",
]

# --- Ruff Formatter ---
[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
# optional (keeps code in docstrings readable):
# docstring-code-line-length = "dynamic"

# --- Ruff Linter: rule selection & ergonomics ---
[tool.ruff.lint]
# Curated, explicit set for "best-for-agents".
select = [
  "F","E4","E7","E9",         # core correctness
  "I",                        # import sorting (linter side)
  "N",                        # pep8-naming
  "UP","SIM","C4","B",        # modernization + simplifications + bugbear
  "RUF",                      # ruff-specific improvements
  "ANN","D",                  # annotations + docstrings
  # "DOC",                    # (enable later if you want stricter doc structure)
  "RET","RSE","TRY",          # clear returns/raises/exception handling
  "EM","G","LOG","ISC",       # error messages, logging, ban implicit str concat
  "TID","ICN",                # tidy imports + import conventions
  "TD","ERA","PGH",           # TODOs, eradicate commented-out code, grep hooks
  "C90","PLR"                 # mild complexity/branches caps (McCabe/Pylint)
]
# Keep Black-compatible ignore; remove "E203" if you later switch to Ruff-only formatting.
ignore = ["Q", "E203"]

# Be conservative with bleeding-edge rule surfaces.
preview = false
# Auto-remove unused "noqa" comments: invaluable for keeping suppressions honest.
extend-select = ["RUF100"]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pylint]
max-branches = 12
max-returns  = 6

# Align docstring convention with your docs stack (Napoleon); choose ONE.
[tool.ruff.lint.pydocstyle]
convention = "numpy"    # or "google" — must match Sphinx Napoleon config

# Make import order deterministic across the monorepo
[tool.ruff.lint.isort]
known-first-party = [
  "docling", "download", "embeddings_dense", "embeddings_sparse",
  "kg_builder", "kgfoundry_common", "linking", "observability",
  "ontology", "orchestration", "registry",
  "search_api", "search_client", "vectorstore_faiss",
]

# Disallow relative imports; agents navigate absolute imports better.
[tool.ruff.lint.flake8-tidy-imports]
ban-relative-imports = "all"

# Optional: standardize common scientific aliases if these libs are used.
[tool.ruff.lint.flake8-import-conventions.aliases]
numpy = "np"
pandas = "pd"
"matplotlib.pyplot" = "plt"
```

**Why this matches your repo:**

* Your project already uses `line-length = 100`, Python 3.13, and both Black and Ruff; the new blocks preserve those while adding determinism and agent‑friendly lint gates. 
* The first‑party list mirrors your `src/` layout so import grouping/sorting is stable across contributors and CI. 
* Exclusions skip generated sites and egg metadata that appear in your tree (e.g., `docs/_build`, `site/`, `kgfoundry.egg-info`). 

---

## 2) Keep (and enforce) the **run order**

> **Action:** Use this exact sequence in pre‑commit, CI, and local scripts:

```bash
ruff check --select I --fix \
&& ruff check --fix \
&& ruff format
```

Ruff’s formatter is Black‑compatible but **doesn’t sort imports**; imports must be handled by the linter (`I`) before formatting. Your original guide already called this out—keep it. 

---

## 3) Pre‑commit wiring (update, don’t replace)

Your docs system explicitly mentions a pre‑commit stack that runs Ruff, Black, Mypy, docformatter, pydocstyle, and interrogate. Keep that model; just ensure **Ruff import sorting → Ruff fixes → Ruff format** happen in that order. 

> **Action:** In `.pre-commit-config.yaml`, add/adjust hooks roughly like:

```yaml
# Ruff: imports first, then fixes, then formatter
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: <pin-to-your-ruff-version>     # match [tool.ruff].required-version
  hooks:
    - id: ruff
      name: ruff (imports)
      args: ["--select", "I", "--fix"]
    - id: ruff
      name: ruff (lint+fix)
      args: ["--fix"]
    - id: ruff-format
      name: ruff (format)
```

If you **keep Black**, leave the Black hook as-is and retain `ignore = ["E203"]`. If you later migrate to **Ruff-only formatting**, remove Black’s hook and drop `E203` from Ruff’s ignores. Your current `pyproject.toml` carries both Black and Ruff sections, so either path is supported. 

---

## 4) CI: run order and gating

Your CI already runs Ruff, Black, pytest, and the docs pipeline (`tools/update_docs.sh`). Keep that, just enforce the same Ruff order and pin the same version used locally for reproducibility. After formatting/linting, run tests and the docs pipeline; fail if the tree is dirty. 

**Suggested job steps (shell-level):**

```bash
uv run ruff --version    # ensure pinned version
uv run ruff check --select I --fix
uv run ruff check --fix
uv run ruff format
uv run pytest -q
uv run tools/update_docs.sh
git diff --quiet || { echo "Working tree changed"; git --no-pager diff; exit 1; }
```

---

## 5) Codebase adjustments the agent should perform

> These are one-time or low-frequency codemods that get you to the new policy.

1. **Convert relative imports → absolute imports**

   * Search for `from .` or `from ..` patterns across `src/**`. Replace with absolute package names from the **first‑party list** above. Re-run Ruff to verify `TID` rules are satisfied. 

2. **Standardize scientific aliases (optional)**

   * If the code uses NumPy/Pandas/Matplotlib, normalize to `np`, `pd`, `plt` so agents can reuse idioms consistently; the linter will enforce it once aliases are declared. (Block is included but harmless if unused.)

3. **Complexity and branching caps**

   * If any functions exceed `max-complexity=10`, `max-branches=12`, or `max-returns=6`, nudge them down with small refactors (early returns, helper functions). These thresholds keep agent patches local and readable.

4. **Suppressions hygiene**

   * Run Ruff with `--fix` to auto‑remove unused `noqa` (`RUF100`), then audit remaining suppressions. The goal is fewer, intentional ignores over time.

---

## 6) Documentation pipeline: align with Ruff

Your automated docs system seeds/rewrites docstrings, formats them, enforces coverage, and builds Sphinx (AutoAPI + Napoleon), all via `tools/update_docs.sh`. Keep that flow, but ensure conventions match Ruff. 

> **Action checklist:**

* **Pick one docstring style**—`numpy` or `google`—and set it in Ruff (`[tool.ruff.lint.pydocstyle]`) **and** ensure Sphinx Napoleon is configured for the same style (Napoleon is already enabled in your stack). 
* Keep your docstring tools (doq, docformatter, pydocstyle, interrogate). Ruff will only format **code blocks in docstrings**; your doc tools continue to reflow docstring text and enforce coverage. 
* Continue to generate/regenerate docs with `tools/update_docs.sh` and Makefile tasks (`make docstrings`, `make readmes`, `make html/json/symbols`). 

---

## 7) Repo-specific toggles

* **First‑party packages** (used by isort inside Ruff) are derived from `src/`:
  `docling`, `download`, `embeddings_dense`, `embeddings_sparse`, `kg_builder`, `kgfoundry_common`, `linking`, `observability`, `ontology`, `orchestration`, `registry`, `search_api`, `search_client`, `vectorstore_faiss`. This makes import grouping/ordering deterministic across machines. 

* **Exclusions**: Skip `docs/_build`, `site/`, and egg‑info dirs; they exist in your tree and are generated. 

* **Python / toolchain alignment**:
  Your `pyproject.toml` targets Python 3.13, with Black pinned to py312 syntax support. The Ruff formatter is Black‑compatible; if/when you migrate to Ruff‑only formatting, you can remove Black to simplify tooling, otherwise keep `E203` ignored for Black compatibility. 

---

## 8) Acceptance criteria (what “done” looks like)

* `ruff check --select I --fix && ruff check --fix && ruff format` produces **no diffs** on a clean checkout. 
* No relative imports remain in `src/**`. Imports are grouped with first‑party blocks stable across contributors. 
* CI shows **grouped** Ruff output and uses the same Ruff version as local (`required-version` satisfied). 
* Sphinx builds cleanly via `tools/update_docs.sh`, and docstring style in code matches Napoleon’s mode (Google/NumPy). 

---

## 9) (Optional) If you want a slimmer toolchain

* Move `black` and `ruff` from `[project.dependencies]` into a developer extra or your pinned `requirements/` set, since they are dev‑time tools. Your current `pyproject.toml` lists them as runtime deps; not harmful, but unnecessary. 
* If you drop Black in favor of Ruff’s formatter, remove Black’s config/hook and delete `E203` from Ruff ignores. 

---

## Appendix A — Rationale (from your prior guide)

* Favor **explicit `select`** over `ALL` to avoid churn when Ruff adds rules.
* Keep stylistic checks that duplicate the formatter (e.g., quotes/wrapping) **disabled**; let the formatter own them.
* Run **import sorting** via `I` before `ruff format`, because the formatter doesn’t sort imports.
  These principles all come from your original “best‑for‑AI” profile and are retained here.  

---

### One-line command for the agent to validate the whole stack locally

```bash
# Lint + format (import sort → fix → format), then docs and tests
ruff check --select I --fix && ruff check --fix && ruff format \
&& tools/update_docs.sh \
&& pytest -q
```

This respects your docs automation and existing test layout.  

---
