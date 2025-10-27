# DOCUMENTATION: best-for-AI-agents ruff formatting 

Here’s a pragmatic “best-for-AI-agents” set of Ruff **prefix groups** to enable. It maximizes structural regularity, rich docstrings, explicit typing, stable imports, and modern syntax—exactly the stuff that helps agents read, refactor, and extend code safely.

### Core (turn these on)

* **F, E4, E7, E9** – Pyflakes + the non-stylistic pycodestyle subsets that Ruff enables by default; good signal, low noise. ([Astral Docs][1])
* **I** – isort (import ordering). Note: Ruff’s *formatter* does **not** sort imports; you must enable `I` and run `ruff check --select I --fix` (then format). ([Astral Docs][2])
* **ANN** – flake8-annotations (enforces complete type hints; complements a real type checker). ([Astral Docs][3])
* **D** (+ **DOC** if you want stricter structure) – pydocstyle (+ pydoclint) for consistent, parseable docstrings; set an explicit convention (Google or NumPy). ([Astral Docs][3])
* **N** – pep8-naming; stable, predictable symbol names help tools. ([Astral Docs][3])
* **UP** – pyupgrade; auto-modernizes syntax (e.g., PEP 604 unions). ([Astral Docs][3])
* **SIM, C4** – simplify and comprehension rules; reduces cyclomatic “texture” agents must parse. ([Astral Docs][3])
* **B** – flake8-bugbear; common foot-guns & design smells. ([Astral Docs][3])
* **RUF** – Ruff-specific improvements and fixes. ([Astral Docs][3])

### Quality & hygiene (strong add-ons)

* **RET, RSE, TRY** – return/raise/exception best practices; clearer control-flow for agents. ([Astral Docs][3])
* **EM, G, LOG** – better error messages and logging patterns. ([Astral Docs][3])
* **ISC** – ban implicit string concat (easy to misread). ([Astral Docs][3])
* **TID, ICN** – tidy imports & import conventions. ([Astral Docs][3])
* **ERA, TD, PGH** – remove commented-out code; surface TODOs. ([Astral Docs][3])

### Optional, depending on stack

* **PERF** (perflint), **S** (bandit), **NPY** (NumPy), **PD** (pandas), **PTH** (pathlib), **TC** (type-checking import placement), **PYI** (stubs). ([Astral Docs][3])

---

### Drop-in config you can use

```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.format]
# Let Ruff handle style; don't duplicate with quote/commas linters.
quote-style = "double"
indent-style = "space"
docstring-code-format = true

[tool.ruff.lint]
select = [
  "F","E4","E7","E9",
  "I","N","UP","SIM","C4","RUF","B",
  "ANN","D","DOC",
  "RET","RSE","TRY","EM","G","LOG","ISC","TID","ICN",
  "TD","ERA","PGH"
]
# Avoid stylistic overlap with the formatter (e.g., flake8-quotes).
ignore = ["Q"]

[tool.ruff.lint.pydocstyle]
# Pick ONE and stick to it project-wide for machine-readability:
convention = "numpy"  # or "google"
```

**How to run (imports → format):**

```bash
ruff check --select I --fix && ruff check --fix && ruff format
```

(Ruff’s formatter aims for Black-compatible output, but it doesn’t sort imports—hence the two-step “check then format”.) ([Astral Docs][2])

---

### Notes & caveats

* Prefer **explicit `select`** (like above) over `ALL`; `ALL` auto-enables new rules on upgrade and can create churn. ([Astral Docs][4])
* Ruff is **not** a type checker—run Mypy/Pyright (or Astral’s emerging *ty* / Red-Knot) alongside it; `ANN` just enforces presence/consistency of hints. ([Astral Docs][5])
* If you adopt Ruff’s formatter, skip linters that duplicate formatting (quotes/commas/line-wrap). Ruff’s formatter supports configurable quotes/indent/line-endings and formats code blocks in docstrings, which is great for AI agents reading examples. ([Astral Docs][2])

This profile gives you consistent structure (imports, names, types, docstrings), modern syntax, and reduced noise—conditions under which AI programming agents make the fewest mistakes and deliver the most reliable edits.

[1]: https://docs.astral.sh/ruff/settings/?utm_source=chatgpt.com "Settings | Ruff - Astral Docs"
[2]: https://docs.astral.sh/ruff/formatter/ "The Ruff Formatter | Ruff"
[3]: https://docs.astral.sh/ruff/rules/ "Rules | Ruff"
[4]: https://docs.astral.sh/ruff/linter/?utm_source=chatgpt.com "The Ruff Linter - Astral Docs"
[5]: https://docs.astral.sh/ruff/faq/?utm_source=chatgpt.com "FAQ | Ruff - Astral Docs"


Here’s the full set of **prefix letters (and letter groups) you can put in `select`** and what each enables:

* **AIR** – Airflow rules
* **ANN** – flake8-annotations
* **ARG** – flake8-unused-arguments
* **A** – flake8-builtins
* **ASYNC** – flake8-async
* **B** – flake8-bugbear
* **BLE** – flake8-blind-except
* **C4** – flake8-comprehensions
* **C90** – mccabe complexity
* **COM** – flake8-commas
* **CPY** – flake8-copyright
* **D** – **pydocstyle** (docstrings)
* **DJ** – flake8-django
* **DOC** – pydoclint
* **DTZ** – flake8-datetimez
* **E**, **W** – pycodestyle (errors/warnings)
* **EM** – flake8-errmsg
* **ERA** – eradicate (commented-out code)
* **EXE** – flake8-executable
* **F** – Pyflakes
* **FA** – flake8-future-annotations
* **FAST** – FastAPI rules
* **FIX** – flake8-fixme (TODO/FIXME etc.)
* **FLY** – flynt (f-string conversions)
* **FURB** – refurb
* **G** – flake8-logging-format
* **I** – isort (import order)
* **ICN** – flake8-import-conventions
* **INP** – flake8-no-pep420
* **INT** – flake8-gettext
* **ISC** – flake8-implicit-str-concat
* **LOG** – flake8-logging
* **N** – pep8-naming
* **NPY** – NumPy-specific rules
* **PD** – pandas-vet
* **PERF** – perflint
* **PGH** – pygrep-hooks
* **PL** – Pylint (with sub-groups **PLC**, **PLE**, **PLR**, **PLW**)
* **PT** – flake8-pytest-style
* **PTH** – flake8-use-pathlib
* **PYI** – flake8-pyi (stub files)
* **Q** – flake8-quotes
* **RET** – flake8-return
* **RSE** – flake8-raise
* **RUF** – Ruff-specific rules
* **S** – flake8-bandit (security)
* **SIM** – flake8-simplify
* **SLOT** – flake8-slots
* **SLF** – flake8-self (private member access)
* **TC** – flake8-type-checking
* **TD** – flake8-todos
* **T10** – flake8-debugger
* **T20** – flake8-print
* **TID** – flake8-tidy-imports
* **TRY** – tryceratops (exception handling)
* **UP** – pyupgrade
* **YTT** – flake8-2020 (version checks) ([Astral Docs][1])

A few useful notes:

* You can also use **`ALL`** to select every rule, then `ignore` specific ones; by default Ruff selects `["E4","E7","E9","F"]`. ([Astral Docs][2])
* Prefixes are hierarchical: `select = ["D"]` enables all docstring rules; `["D2"]` would enable just the `D2xx` subset, etc. ([Astral Docs][3])
* If you’re enforcing a specific docstring style (Google/NumPy/PEP 257), you can set it under `[tool.ruff.lint.pydocstyle]` with `convention = "google" | "numpy" | "pep257"`. ([Astral Docs][4])

All of the above mappings come from Ruff’s official **Rules** index. If you want to double-check or browse individual rules, that page lists every code under each prefix. ([Astral Docs][1])

[1]: https://docs.astral.sh/ruff/rules/ "Rules | Ruff"
[2]: https://docs.astral.sh/ruff/settings/?utm_source=chatgpt.com "Settings | Ruff - Astral Docs"
[3]: https://docs.astral.sh/uv/reference/cli/?utm_source=chatgpt.com "Commands | uv - Astral Docs"
[4]: https://docs.astral.sh/ruff/faq/?utm_source=chatgpt.com "FAQ | Ruff - Astral Docs"
