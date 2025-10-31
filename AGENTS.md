# AGENTS.md — Agent Operating Protocol (AOP)

> **Purpose**: This file is the agent-first playbook for building and maintaining this codebase at **best‑in‑class** quality. It specifies *exact commands*, *acceptance gates*, and *fallbacks* so both humans and AI agents can ship production‑grade Python safely and quickly.

---

## Table of contents
1. [Agent Operating Protocol (TL;DR)](#agent-operating-protocol-tldr)
2. [Environment Setup (Agent‑grade, deterministic)](#environment-setup-agentgrade-deterministic)
3. [Source‑of‑Truth Index](#source-of-truth-index)
4. [Code Formatting & Style (Ruff is canonical)](#code-formatting--style-ruff-is-canonical)
5. [Type Checking (pyrefly sharp, mypy strict)](#type-checking-pyrefly-sharp-mypy-strict)
6. [Docstrings (NumPy style; enforced; runnable)](#docstrings-numpy-style-enforced-runnable)
7. [Testing Standards (markers, coverage, GPU hygiene)](#testing-standards-markers-coverage-gpu-hygiene)
8. [Data Contracts (JSON Schema 2020‑12 / OpenAPI 3.1)](#data-contracts-json-schema-2020-12--openapi-31)
9. [Link Policy for Remote Editors](#link-policy-for-remote-editors)
10. [Agent Catalog & STDIO API (session‑scoped, no daemon)](#agent-catalog--stdio-api-sessionscoped-no-daemon)
11. [Task Playbooks (feature / refactor / bugfix)](#task-playbooks-feature--refactor--bugfix)
12. [PR Template & Checklist](#pr-template--checklist)
13. [Quick Commands (copy/paste)](#quick-commands-copypaste)
14. [Troubleshooting (for Agents)](#troubleshooting-for-agents)
15. [Security & Compliance Hygiene](#security--compliance-hygiene)
16. [Repo Layout & Do‑not‑edit Zones](#repo-layout--do-not-edit-zones)
17. [CI / Pre‑commit parity](#ci--precommit-parity)
18. [Glossary](#glossary)

---

## Agent Operating Protocol (TL;DR)

**You are an autonomous Python engineer. Follow this sequence for *every* change:**

1) **Load context fast**
   - Read this file end‑to‑end.
   - Open the relevant spec/proposal under `openspec/` (when applicable).
   - Bootstrap environment and run local checks:
     ```bash
     uv sync 
     ```

2) **Plan (write before code)**
   - Draft a **4‑item design note** for the PR:
     1. Summary (one paragraph)
     2. Public API sketch (typed signatures)
     3. Data/Schema contracts affected (JSON Schema, OpenAPI)
     4. Test plan (happy paths, edge cases, negative cases)
   - If data crosses boundaries, confirm schemas (see [Data Contracts](#data-contracts-json-schema-2020-12--openapi-31)).

3) **Implement**
   - Code to the typed API sketch.
   - Keep functions small; separate I/O from pure logic; prefer composition over inheritance.

4) **Validate quality gates locally**
   ```bash
   uv run ruff format && uv run ruff check --fix
   uv run pyrefly check
   uv run mypy --config-file mypy.ini
   uv run pytest -q
   make artifacts && git diff --exit-code    # docs/nav/catalog/schemas in sync
   python tools/check_new_suppressions.py src    # verify no untracked suppressions
   uv run python -m importlinter --config importlinter.cfg      # verify architectural boundaries
   ```

5) **Ship the PR**
   - PR description = the **4‑item design note** + **checklist outputs** (commands & exit codes).
   - If behavior changes, include **migration notes** and bump **SemVer** accordingly.

> If any step fails, **stop and fix** before continuing.

---

## Environment Setup (Agent‑grade, deterministic)

- **Canonical manager:** `uv`
- **Python:** pinned to **3.13.9**
- **Virtual env:** project‑local `.venv/` only (never system Python)
- **One‑shot bootstrap (preferred):**
  ```bash
  # If scripts/bootstrap.sh exists, use it. Otherwise run the commands below manually.
  uv python install 3.13.9
  uv python pin 3.13.9
  uv sync --locked
  uv tool install pre-commit
  pre-commit install
  ```
- **Remote container / devcontainer:** follow [Link Policy for Remote Editors](#link-policy-for-remote-editors) so deep links open correctly from generated artifacts.
- **Do not** duplicate tool configs across files. `mypy.ini` and `pyrefly.toml` are canonical; `pyproject.toml` is canonical for Ruff and packaging.

---

## Source‑of‑Truth Index

Read these first when editing configs or debugging local vs CI drift:

- **Formatting & lint:** `pyproject.toml` → `[tool.ruff]`, `[tool.ruff.lint]`
- **Types:** `mypy.ini` (single source), `pyrefly.toml` (single source)
- **Tests:** `pytest.ini` (markers, doctest/xdoctest config)
- **Docs / Artifacts:** `tools/docs/build_artifacts.py`, `make artifacts`, outputs under `docs/_build/**`, `site/_build/**`
- **Nav & Catalog:** `tools/navmap/*`, `docs/_build/agent_catalog.json`, `site/_build/agent/` (Agent Portal)
- **CI:** `.github/workflows/ci.yaml` (job order: precommit → lint → types → tests → docs; OS matrix; caches; artifacts)
- **Pre‑commit:** `.pre-commit-config.yaml` (runs the same gates locally)

---

## Code Formatting & Style (Ruff is canonical)

- **Run order:** `uv run ruff format` → `uv run ruff check --fix` (imports auto‑sorted).
- **Imports:** stdlib → third‑party → first‑party; absolute imports only.
- **Style guardrails:** 100‑col width, 4‑space indent, double quotes, trailing commas on multiline.
- **Complexity:** cyclomatic ≤ 10, returns ≤ 6, branches ≤ 12; refactor if exceeded.
- **Rule families emphasized (non‑exhaustive):**
  - Baseline: `F,E4,E7,E9,I,N,UP,SIM,B,RUF,ANN,D,RET,RSE,TRY,EM,G,LOG,ISC,TID,TD,ERA,PGH,C90,PLR`
  - Extra quality & safety: `W,A,ARG,BLE,DTZ,PTH,PIE,S,PT,T20`
    (builtins shadowing, unused args, bare except, timezone‑aware datetimes, pathlib, security, pytest style, ban prints)

> We standardize on **Ruff** for formatter + linter. Do **not** run Black in parallel (conflicts / duplicate work).

---

## Type Checking (pyrefly sharp, mypy strict)

- **First‑line check (semantics):**
  ```bash
  uv run pyrefly check
  ```
- **Soundness (strict baseline):**
  ```bash
  uv run mypy --config-file mypy.ini
  ```
- **Rules of engagement:**
  - No untyped **public** APIs.
  - Prefer **PEP 695** generics and `Protocol`/`TypedDict` over `Any`.
  - Every `# type: ignore[...]` requires a comment **why** + a ticket reference.
  - Narrow exceptions; HTTP surfaces return **RFC 9457 Problem Details**.

**“Type‑clean” means both pyrefly **and** mypy pass.**

---

## Docstrings (NumPy style; enforced; runnable)

- **Style:** NumPy docstrings; PEP 257 structure (module/class/function docstrings for all public symbols)
- **Enforcement:** `pydoclint` parity checks + `interrogate` coverage (≥90%)
- **Runnability:** Examples in `Examples` must execute (doctest/xdoctest); keep snippets short and copy‑ready
- **Required sections (public APIs):**
  - Summary (one line, imperative)
  - Parameters (name, type, meaning)
  - Returns / None
  - Raises (type + condition)
  - Examples
  - Notes (performance, side‑effects)
- **Loop:** code → `make artifacts` (scaffold/validate) → refine docs → `make artifacts` → commit

---

## Testing Standards (markers, coverage, GPU hygiene)

- **Markers:**
  - `@pytest.mark.integration` — network/services/resources
  - `@pytest.mark.gpu` — requires GPU libraries
  - `@pytest.mark.benchmark` — performance, non‑gating
- **Conventions:**
  - Parametrize edge cases with `@pytest.mark.parametrize`
  - No reliance on test order or realtime; use fixed seeds
  - GPU tests are **skipped by default** in CI unless explicitly enabled
- **Coverage (local whenever core paths change):**
  ```bash
  uv run pytest -q --cov=src --cov-report=xml:coverage.xml --cov-report=html:htmlcov
  ```

---

## Data Contracts (JSON Schema 2020‑12 / OpenAPI 3.1)

- **Boundary rule:** whenever data crosses a boundary (API, file, queue), define a **JSON Schema 2020‑12**. For HTTP, use **OpenAPI 3.1** (embeds 2020‑12).
- **Source of truth:** the schema is canonical; models may be generated from it (or emit it) but do not replace it.
- **Validation:** validate inputs/outputs in tests; version schemas with SemVer and document breaking changes.

---

## Link Policy for Remote Editors

- **Editor mode (preferred for local dev):**
  - `DOCS_LINK_MODE=editor`
  - `EDITOR_URI_TEMPLATE="vscode-remote://dev-container+{container_id}{path}:{line}"`
  - Optional `PATH_MAP` file, lines: `/container/prefix => /editor/workspace/prefix`
- **GitHub mode (fallback):**
  - `DOCS_LINK_MODE=github`
  - `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`
  - Links like: `https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}`
- **Agent rule:** use **editor** mode in remote containers for deep linking; use **github** when editor URIs are unavailable.

Example `PATH_MAP`:
```
/workspace => /workspaces/kgfoundry
/app       => /workspaces/kgfoundry
```

---

## Agent Catalog & STDIO API (session‑scoped, no daemon)

- **Artifacts:**
  - Ground truth: `docs/_build/agent_catalog.json`
  - Portal: `site/_build/agent/index.html`
- **Session API (JSON over stdio):**
  ```json
  {"id":"1","method":"capabilities","params":{}}
  {"id":"2","method":"search","params":{"q":"vector store","k":10}}
  {"id":"3","method":"open_anchor","params":{"symbol_id":"py:kg.index.faiss.build_index","mode":"editor"}}
  {"id":"4","method":"find_callers","params":{"symbol_id":"py:kg.doc.parse.Parser.parse"}}
  {"id":"5","method":"find_callees","params":{"symbol_id":"py:kg.index.faiss.build_index"}}
  ```
- **Lifecycle:**
  - Editor spawns: `python -m tools.docs.stdio_api`
  - First call is `capabilities`; one request at a time (MVP)
  - Process exits when stdin closes

---

## Task Playbooks (feature / refactor / bugfix)

### A) New Feature
1. Read the spec/proposal; write the **4‑item design note**.
2. Define/update **JSON Schema** for any boundary payloads.
3. Implement **typed** public API; write pure logic first, I/O later.
4. Add **parametrized tests** (happy paths, edges, negative cases).
5. Run quality gates (format → lint → pyrefly → mypy → pytest).
6. `make artifacts` and commit regenerated docs/nav/catalog.

**Done when:** all gates green; PR includes design note, schemas, and runnable examples.

### B) Safe Refactor (no behavior change)
1. Prove parity with tests/fixtures **before** changes.
2. Extract pure functions; reduce complexity; improve names & docs.
3. Maintain public signatures; add deprecations if needed (warn + doc).
4. Run full gates; add a “refactor proof” note in the PR (tests proving equivalence).

### C) Bugfix
1. Reproduce with a failing **parametrized** test.
2. Fix with smallest diff; explain root cause in PR.
3. Add a **regression test**.
4. Run full gates; link issue/ticket in commit.

---

## PR Template & Checklist

**Use this PR template:**

- **Title:** `<area>: <short imperative>`
- **Summary:** one paragraph describing the change and why.
- **Public API:** list symbols & **typed** signatures that changed/added.
- **Data Contracts:** link to JSON Schema/OpenAPI diff (if any).
- **Test Plan:** commands + what they prove.
- **Docs:** where examples changed; screenshots/links to artifacts (docs site, portal).
- **Impact:** migration notes / deprecations.

**Checklist (paste outputs):**
```
[ ] uv run ruff format && uv run ruff check --fix
[ ] uv run pyrefly check
[ ] uv run mypy --config-file mypy.ini
[ ] uv run pytest -q
[ ] make artifacts && git diff --exit-code
[ ] python tools/check_new_suppressions.py src
[ ] uv run python -m importlinter --config importlinter.cfg
```

**Problem Details Example:**
- Canonical example: `schema/examples/problem_details/search-missing-index.json`
- All HTTP error responses must conform to RFC 9457 Problem Details format
- See `src/kgfoundry_common/errors/` for exception taxonomy and Problem Details helpers

---

## Quick Commands (copy/paste)

```bash
# Format & lint
uv run ruff format && uv run ruff check --fix

# Types (sharp first, then soundness)
uv run pyrefly check
uv run mypy --config-file mypy.ini

# Tests (incl. doctests/xdoctest via pytest.ini)
uv run pytest -q

# Artifacts (docstrings, navmap, schemas, portal)
make artifacts
git diff --exit-code

# Architectural boundaries & suppression guard
python tools/check_new_suppressions.py src
uv run python -m importlinter --config importlinter.cfg

# All pre-commit hooks
uvx pre-commit run --all-files
```

**Problem Details Reference:**
- Example: `schema/examples/problem_details/search-missing-index.json`
- Schema: RFC 9457 Problem Details format
- Implementation: `src/kgfoundry_common/errors/`

---

## Troubleshooting (for Agents)

- **Ruff vs Black conflicts**: We use **Ruff only**; remove Black changes, re-run Ruff.
- **Third‑party typing gaps**: prefer small typed facades (`Protocol`, `TypedDict`) or `stubs/` + `mypy_path`, not `Any`.
- **Docs drift keeps appearing**: run `make artifacts` from a **clean** tree; ensure `docs/_build/**` isn’t ignored; commit regenerated files.
- **Editor links open wrong path**: verify `PATH_MAP` and `EDITOR_URI_TEMPLATE`; rebuild artifacts.
- **Slow CI**: check cache restore logs for `uv`, `ruff`, `mypy`. If keys miss, verify `uv.lock` & Python version detection step.

---

## Security & Compliance Hygiene

- **Secrets**: never commit `.env` or tokens; redact secrets in logs.
- **Validation**: sanitize all untrusted inputs at boundaries; validate against schemas.
- **Dependencies**: run `uv run pip-audit` locally before large upgrades; nightlies may run in CI.
- **Licenses**: prefer MIT/Apache‑2.0; consult SBOM when adding third‑party libs.

---

## Repo Layout & Do‑not‑edit Zones

- **Source**: `src/**`
- **Tests**: `tests/**` (mirrors `src`)
- **Docs & site**: `docs/**`, `site/**`
- **Generated artifacts**: `docs/_build/**`, `site/_build/**` → **do not hand‑edit**
- **Stubs**: `stubs/**` for local type shims (referenced by `mypy_path`)

---

## CI / Pre‑commit parity

- **Job order**: precommit → lint → types → tests → docs
- **OS matrix**: lint/types on linux + macOS; tests on linux (expand later if needed)
- **Caches**: `~/.cache/uv`, `~/.cache/ruff`, `.mypy_cache` keyed on OS + Python + lock/config
- **Artifacts**: docs site, Agent Portal, coverage, JUnit, (optional) mypy HTML are uploaded for each run
- **Branch protection (recommended)**: require `precommit`, `lint`, `types`, `tests` to merge

---

## Glossary

- **Agent Catalog** — machine‑readable index of packages/modules/symbols with stable anchors and links
- **Agent Portal** — static HTML UI over the catalog with search and deep links
- **Anchor** — a deep link to source (editor/GitHub), optionally including a line number
- **PATH_MAP** — rules for translating container paths to editor workspace paths
- **DocFacts/NavMap** — generated indices that power docs and linking
- **RFC 9457 Problem Details** — standard JSON error envelope for HTTP APIs

---

**This document is the authoritative operating protocol for agents.** If a task conflicts with these rules, prefer the rules — or open a proposal under `openspec/changes/**` to evolve them.
