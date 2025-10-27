## Environment Setup (Preferred Order)

1. **direnv (recommended default)**
   - Install `direnv` for your shell (`sudo apt install direnv` or `brew install direnv`) and hook it via `eval "$(direnv hook bash)"` / `eval "$(direnv hook zsh)"` in your shell rc.
   - Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
   - From the repo root run `direnv allow` once. The committed `.envrc` will:
     - create or reuse `.venv/` via `uv venv`
     - run `uv sync --frozen --extra docs` (no `gpu` extras)
     - activate the environment for that shell and install pre-commit hooks automatically.
   - Future shell entries auto-refresh whenever `pyproject.toml`, `uv.lock`, or `.env` change.

2. **Bootstrap script (when direnv is unavailable)**
   - Execute `bash scripts/bootstrap.sh` from the repository root.
   - The script mirrors the `.envrc` workflow: pins Python if needed, creates `.venv`, runs `uv sync --frozen --extra docs`, activates the environment, and installs pre-commit hooks.
   - Supports overrides such as `OFFLINE=1`, `USE_WHEELHOUSE=1`, or additional extras (GPU installs require `ALLOW_GPU=1`).

3. **Manual uv flow (last resort / CI snippets)**
   - `uv python pin 3.13`
   - `uv venv`
   - `uv sync --frozen --extra docs`
   - `uvx pre-commit install -t pre-commit -t pre-push`
   - Activate via `. .venv/bin/activate` or run tools with `uv run` / `uvx`.

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Environment Setup (Preferred Order)

1. **direnv (recommended default)**
   - Install `direnv` for your shell (`sudo apt install direnv` or `brew install direnv`) and hook it via `eval "$(direnv hook bash)"` / `eval "$(direnv hook zsh)"` in your shell rc.
   - Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
   - From the repo root run `direnv allow` once. The committed `.envrc` will:
     - create or reuse `.venv/` via `uv venv`
     - run `uv sync --frozen --extra docs` (no `gpu` extras)
     - activate the environment for that shell and install pre-commit hooks automatically.
   - Future shell entries auto-refresh whenever `pyproject.toml`, `uv.lock`, or `.env` change.

2. **Bootstrap script (when direnv is unavailable)**
   - Execute `bash scripts/bootstrap.sh` from the repository root.
   - The script mirrors the `.envrc` workflow: pins Python if needed, creates `.venv`, runs `uv sync --frozen --extra docs`, activates the environment, and installs pre-commit hooks.
   - Supports overrides such as `OFFLINE=1`, `USE_WHEELHOUSE=1`, or additional extras (GPU installs require `ALLOW_GPU=1`).

3. **Manual uv flow (last resort / CI snippets)**
   - `uv python pin 3.13`
   - `uv venv`
   - `uv sync --frozen --extra docs`
   - `uvx pre-commit install -t pre-commit -t pre-push`
   - Activate via `. .venv/bin/activate` or run tools with `uv run` / `uvx`.