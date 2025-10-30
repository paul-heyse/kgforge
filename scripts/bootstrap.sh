#!/usr/bin/env bash
# scripts/bootstrap.sh
# Purpose: Provision a **deterministic, agent-grade** Python dev environment using uv.
# Works on Linux and macOS (bash 3.2+). Safe for local machines and remote containers.
#
# What it does:
#   1) Installs uv if missing (user-local)
#   2) Ensures Python 3.13.9 is installed via uv and pinned for this project
#   3) Creates/uses a project-local .venv (no system Python)
#   4) Installs deps via `uv sync` (prefers --locked when uv.lock present)
#   5) Installs and runs pre-commit hooks (optional flags to skip/run)
#   6) Optionally generates PATH_MAP for editor deep links in remote containers
#   7) Prints a ready-to-run command summary
#
# Exit on first error, unset var, or failed pipe; propagate failures out of subshells.
set -Eeuo pipefail

# ------------- Config (change defaults here if needed) -------------
PY_VER_DEFAULT="${PY_VER_DEFAULT:-3.13.9}"
PIN_PYTHON="${PIN_PYTHON:-1}"             # 1=uv python pin <ver>
RUN_PRE_COMMIT="${RUN_PRE_COMMIT:-1}"     # 1=install + run pre-commit on all files
GENERATE_PATH_MAP="${GENERATE_PATH_MAP:-1}" # 1=create docs/_build/path_map.txt if in container
USE_LOCK="${USE_LOCK:-auto}"               # auto|yes|no  -> --locked when uv.lock exists (auto)
EDITOR_URI_TEMPLATE_DEFAULT='vscode-remote://dev-container+{container_id}{path}:{line}'

# ------------- CLI flags -------------
# Support a few handy flags so CI or developers can tailor behavior.
#   --no-pin-python      : do not uv python pin
#   --skip-pre-commit    : don't install or run pre-commit hooks
#   --no-path-map        : don't generate docs/_build/path_map.txt
#   --use-lock[=yes|no]  : force use of uv.lock or ignore it
#   --py 3.13.9          : override Python version
usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap.sh [options]

Options:
  --py <x.y.z>         Pin/install this Python version via uv (default: 3.13.9)
  --no-pin-python      Skip `uv python pin`
  --skip-pre-commit    Do not install/run pre-commit hooks
  --no-path-map        Do not generate docs/_build/path_map.txt
  --use-lock=<auto|yes|no>
  -h, --help           Show this help
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0;;
    --no-pin-python) PIN_PYTHON=0;;
    --skip-pre-commit) RUN_PRE_COMMIT=0;;
    --no-path-map) GENERATE_PATH_MAP=0;;
    --use-lock=*) USE_LOCK="${arg#*=}";;
    --py) shift; PY_VER_DEFAULT="${1:?--py requires a version like 3.13.9}";;
    --py=*) PY_VER_DEFAULT="${arg#*=}";;
  esac
done

# ------------- Logging helpers -------------
# Basic color output (graceful fallback if tput missing/unsupported).
if command -v tput >/dev/null 2>&1; then
  if [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    BOLD="$(tput bold || true)"; DIM="$(tput dim || true)"; RESET="$(tput sgr0 || true)"
    GREEN="$(tput setaf 2 || true)"; YELLOW="$(tput setaf 3 || true)"; RED="$(tput setaf 1 || true)"
  fi
fi
BOLD="${BOLD:-}"; DIM="${DIM:-}"; RESET="${RESET:-}"
GREEN="${GREEN:-}"; YELLOW="${YELLOW:-}"; RED="${RED:-}"

info() { echo "${DIM}>>${RESET} $*"; }
ok()   { echo "${GREEN}✔${RESET} $*"; }
warn() { echo "${YELLOW}⚠${RESET} $*" 1>&2; }
err()  { echo "${RED}✘${RESET} $*" 1>&2; }

have() { command -v "$1" >/dev/null 2>&1; }

# ------------- Sanity: repo root & OS -------------
if [ ! -f "pyproject.toml" ]; then
  err "Run this script from the repository root (pyproject.toml not found)."
  exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"   # linux|darwin|...
REPO_NAME="$(basename "$(pwd)")"

# ------------- Ensure uv is installed -------------
ensure_uv() {
  if have uv; then
    ok "uv present: $(uv --version | head -n1)"
    return 0
  fi
  warn "uv not found; installing user-local uv (no sudo)."
  # Official installer from Astral: https://astral.sh/uv/docs/install
  # -sSf: silent + show errors + fail on errors
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Ensure uv is on PATH for this shell (installer prints its target dir).
  if ! have uv; then
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/share/uv/bin:$PATH"
  fi
  have uv || { err "uv still not on PATH after install. Add it to PATH, then re-run."; exit 1; }
  ok "Installed uv: $(uv --version | head -n1)"
}

# ------------- Ensure Python toolchain via uv -------------
ensure_python() {
  local ver="${1:?python version required}"
  if ! uv python list | grep -q "${ver}"; then
    info "Installing Python ${ver} via uv…"
    uv python install "${ver}"
  fi
  if [ "${PIN_PYTHON}" = "1" ]; then
    uv python pin "${ver}"
    ok "Pinned Python ${ver} for this project"
  else
    warn "Skipping uv python pin (requested)"
  fi
}

# ------------- Sync dependencies -------------
sync_env() {
  # Prefer locked resolution when uv.lock exists (or forced by flag).
  local lock_flag=""
  case "${USE_LOCK}" in
    auto) [ -f uv.lock ] && lock_flag="--locked" ;;
    yes)  lock_flag="--locked" ;;
    no)   lock_flag="" ;;
    *)    warn "--use-lock must be auto|yes|no (got: ${USE_LOCK}); defaulting to auto"; [ -f uv.lock ] && lock_flag="--locked" ;;
  esac

  info "Syncing dependencies (uv sync ${lock_flag})…"
  uv sync ${lock_flag}
  ok "Environment synced"
}

# ------------- Pre-commit hooks -------------
setup_precommit() {
  if [ "${RUN_PRE_COMMIT}" != "1" ]; then
    warn "Skipping pre-commit install (requested)"
    return 0
  fi
  if ! have pre-commit; then
    info "Installing pre-commit via uv tool…"
    uv tool install pre-commit
  fi
  pre-commit install
  ok "pre-commit hooks installed"
  info "Running pre-commit on all files (may take a minute on first run)…"
  # Use uvx to ensure the same version that installed hooks; exit nonzero if failures.
  uvx pre-commit run --all-files
  ok "pre-commit finished"
}

# ------------- Generate PATH_MAP for remote editors (optional) -------------
generate_path_map() {
  if [ "${GENERATE_PATH_MAP}" != "1" ]; then
    return 0
  fi
  mkdir -p docs/_build
  local pm="docs/_build/path_map.txt"
  if [ -f "${pm}" ]; then
    ok "PATH_MAP already exists at ${pm}"
    return 0
  fi

  # Heuristic: Dev Containers often mount the repo at /workspaces/<name>;
  # Codespaces/VS Code Remote use similar layouts. Provide sensible defaults.
  local container_prefix=""
  local editor_prefix=""
  if [ -d "/workspace" ]; then
    container_prefix="/workspace"
    editor_prefix="/workspaces/${REPO_NAME}"
  elif [ -d "/workspaces/${REPO_NAME}" ]; then
    container_prefix="/workspaces/${REPO_NAME}"
    editor_prefix="/workspaces/${REPO_NAME}"
  else
    # Fallback: map repo root to itself (useful for local shells)
    container_prefix="$(pwd)"
    editor_prefix="$(pwd)"
  fi

  {
    echo "${container_prefix} => ${editor_prefix}"
  } > "${pm}"

  ok "Generated ${pm} mapping: ${container_prefix} => ${editor_prefix}"
  if [ -z "${EDITOR_URI_TEMPLATE:-}" ]; then
    warn "EDITOR_URI_TEMPLATE not set; using default when generating links."
    export EDITOR_URI_TEMPLATE="${EDITOR_URI_TEMPLATE_DEFAULT}"
  fi
}

# ------------- Diagnostics -------------
print_summary() {
  cat <<EOF

${BOLD}Environment ready.${RESET}
  Python     : $(python -V 2>&1 || true)
  uv         : $(uv --version 2>/dev/null | head -n1 || echo "n/a")
  Location   : $(pwd)
  OS         : ${OS}
  Venv       : .venv (created by uv)

Next steps:
  - Lint/format : uv run ruff format && uv run ruff check --fix
  - Type-check  : uv run pyrefly check && uv run mypy --config-file mypy.ini
  - Tests       : uv run pytest -q
  - Artifacts   : make artifacts && git diff --exit-code
  - Docs (open) : site/_build/html/index.html  (or Agent Portal at site/_build/agent/index.html)

Tip: re-run hooks anytime with: uvx pre-commit run --all-files
EOF
}

# ------------- Main flow -------------
ensure_uv
ensure_python "${PY_VER_DEFAULT}"
sync_env
setup_precommit
generate_path_map
print_summary
